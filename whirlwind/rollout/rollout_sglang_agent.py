"""SGLang rollout node for agent traffic (EXPERIMENTAL).

A sibling of :mod:`whirlwind.rollout.rollout_sglang` for the agentic mode
(docs/agentic_multiturn.md). On top of the base node it adds, all driven by
:class:`whirlwind.config_agentic.AgentRolloutConfig`:

* **chat conversations + tools** — a ``PromptBatch`` carrying
  ``conversations`` (per-prompt message lists with system / user / assistant /
  tool roles) and ``tools`` (OpenAI-style schemas) is rendered with the
  model's own chat template, so multi-turn and tool-calling structure reaches
  the model natively. Every string is special-token-escaped first, so an
  observation cannot forge a turn boundary;
* **server-side tool-call parsing** (``tool_call_parser``) with SGLang's
  FunctionCallParser registry; parsed calls ride along as
  ``RolloutResult.tool_calls`` while the raw completion stays canonical for
  training;
* **template control** (``apply_chat_template``, ``chat_template_kwargs``),
  **radix-cache** and **speculative-decoding** engine settings;
* **constrained decoding** (``PromptBatch.sampling_constraint``): a request
  may restrict its completions to an SGLang structured-output grammar
  (``ebnf``, ``regex``, ``json_schema`` or ``structural_tag``), e.g. to
  enforce the chat template's tool-call format;
* **concurrent serving** (``n_rpc_workers``): agents send one small request
  per (episode, turn), so the node runs several RPC workers and lets SGLang
  batch them. Weight updates, sleep and wake wait for in-flight generations
  and hold new ones back, so an adapter swap never overlaps a request.

Run it like the base node::

    python3 -m whirlwind.rollout.rollout_sglang_agent --config configs/rollout/agentic_engine.yaml
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import types
from collections.abc import AsyncIterator
from typing import Any

from whirlwind.config_agentic import AgentRolloutConfig
from whirlwind.messages import PromptBatch, RolloutResult
from whirlwind.rollout.rollout_sglang import RolloutNode, main

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat-template safety: neutralize special-token markers in untrusted text
# ---------------------------------------------------------------------------
# Anything interpolated into a chat-template conversation (problem statements,
# bash command outputs, model-emitted commands) must never carry a literal
# special-token or tool-structure marker: an observation containing
# "<|im_end|>" would inject a fake turn boundary into the templated text and
# the special token itself into the token stream; "<tool_call>"/"</?think>" likewise
# forge tool-call / reasoning structure. Escaping backslash-prefixes every
# marker, which (a) breaks exact-match tokenization of the special token and
# (b) leaves the template's own structure markers unambiguous. Idempotent:
# already-escaped markers (even user-pre-backslashed) normalize to exactly
# one backslash, so caller-side and engine-side layers can stack.
_CHAT_MARKER_RE = re.compile(
    r"\\?(<\|[A-Za-z_]+\|>|</?tool_call>|</?think>|</?assistant>)"
)


def escape_special_tokens(text: str) -> str:
    r"""Backslash-escape chat special-token / tool-structure markers in text.

    >>> escape_special_tokens("out: <|im_end|> done")
    'out: \\<|im_end|> done'
    """
    return _CHAT_MARKER_RE.sub(r"\\\1", text)


def escape_conversation(messages: Any) -> Any:
    """Deep special-token escape of a conversation (contents + tool_call args).

    Walks dicts/lists and escapes every string value — message contents, tool
    names, and tool_call ``arguments`` payloads. The engine applies this as a
    fail-closed final layer before templating; agents should escape untrusted
    text as they build conversations (the escape is idempotent, so both layers
    stack without double-escaping).
    """
    if isinstance(messages, str):
        return escape_special_tokens(messages)
    if isinstance(messages, list):
        return [escape_conversation(m) for m in messages]
    if isinstance(messages, dict):
        return {k: escape_conversation(v) for k, v in messages.items()}
    return messages


# SGLang structured-output sampling params a request may set (one at a time).
_CONSTRAINT_KEYS = frozenset({"ebnf", "regex", "json_schema", "structural_tag"})

# apply_chat_template kwargs the engine always controls; a config
# chat_template_kwargs entry colliding with these fails closed at startup.
_TEMPLATE_KWARG_RESERVED = frozenset({"tokenize", "add_generation_prompt", "tools"})


def _merge_template_kwargs(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the config's chat_template_kwargs (reserved keys are rejected)."""
    if not extra:
        return {}
    clash = _TEMPLATE_KWARG_RESERVED.intersection(extra)
    if clash:
        raise ValueError(
            "chat_template_kwargs may not override engine-controlled keys: "
            f"{sorted(clash)}"
        )
    return dict(extra)


def apply_chat_template_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Render a full message list with the tokenizer's chat template.

    Native multi-turn structure (system / user / assistant / tool roles) and
    OpenAI-style tool schemas. Every string in the conversation is
    special-token-escaped first (idempotent, so agent-side escaping stacks):
    an unescaped "<|im_end|>" inside an observation would otherwise inject a
    fake turn boundary and a special token into the token stream.
    """
    safe = escape_conversation(messages)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    kwargs.update(_merge_template_kwargs(template_kwargs))
    return tokenizer.apply_chat_template(safe, **kwargs)


def _parse_tool_calls(
    completions: list[str],
    tools: list[dict[str, Any]] | None,
    parser_name: str,
) -> list[list[dict[str, Any]]]:
    """Parse structured tool calls out of completions with SGLang's parser registry.

    The tool-call grammar is a property of the chat template this node
    applies, so parsing belongs here: every external harness gets structured
    calls instead of reimplementing grammar detection. Returns one list of
    ``{"name", "arguments"}`` dicts per completion (``[]`` = no call).
    Fail-closed: unparseable arguments become ``{}``, and a parser crash
    downgrades that completion to no-call.
    """
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    # Duck-typed stand-ins for SGLang's Tool protocol objects.
    def _tool(t: dict[str, Any]):
        f = t.get("function", {})
        return types.SimpleNamespace(
            type=t.get("type", "function"),
            function=types.SimpleNamespace(
                name=f.get("name", ""), parameters=f.get("parameters", {})
            ),
        )

    sglang_tools = [_tool(t) for t in (tools or [])]
    out: list[list[dict[str, Any]]] = []
    for text in completions:
        calls: list[dict[str, Any]] = []
        try:
            parser = FunctionCallParser(sglang_tools, parser_name)  # detectors are stateful
            _, items = parser.parse_non_stream(text)
            for item in items:
                try:
                    raw = item.parameters
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
                except (ValueError, TypeError):
                    args = {}
                calls.append({"name": item.name, "arguments": args})
        except Exception:
            logger.warning(
                "tool_call_parser=%r failed on one completion; treating it as no-call",
                parser_name,
            )
        out.append(calls)
    return out


class _ServeGate:
    """Reader/writer gate: generations share it, engine-state changes
    (weight update, sleep, wake) take it exclusively. A waiting writer blocks
    new readers, so an update cannot starve behind a stream of requests."""

    def __init__(self) -> None:
        self._cond: asyncio.Condition | None = None  # bound lazily to the running loop
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @contextlib.asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        cond = self._condition()
        async with cond:
            await cond.wait_for(lambda: not self._writer and self._writers_waiting == 0)
            self._readers += 1
        try:
            yield
        finally:
            async with cond:
                self._readers -= 1
                cond.notify_all()

    @contextlib.asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        cond = self._condition()
        async with cond:
            self._writers_waiting += 1
            try:
                await cond.wait_for(lambda: not self._writer and self._readers == 0)
            finally:
                self._writers_waiting -= 1
                cond.notify_all()  # readers held back by this writer may proceed
            self._writer = True
        try:
            yield
        finally:
            async with cond:
                self._writer = False
                cond.notify_all()


class AgentRolloutNode(RolloutNode):
    """SGLang rollout node with conversations, tools and tool-call parsing."""

    config: AgentRolloutConfig

    def __init__(self, config: AgentRolloutConfig) -> None:
        _merge_template_kwargs(config.chat_template_kwargs)  # fail at startup
        super().__init__(config)
        self._gate = _ServeGate()

    def _engine_kwargs(self) -> dict:
        kwargs = super()._engine_kwargs()
        kwargs["disable_radix_cache"] = bool(self.config.disable_radix_cache)
        if self.config.speculative_algorithm:
            kwargs["speculative_algorithm"] = self.config.speculative_algorithm
            if self.config.speculative_num_draft_tokens is not None:
                kwargs["speculative_num_draft_tokens"] = (
                    self.config.speculative_num_draft_tokens
                )
        return kwargs

    def set_up(self) -> None:
        super().set_up()
        override = self.config.apply_chat_template
        if override is not None:
            self._use_chat_template = bool(override)
            logger.info(
                "Rollout: chat templating forced %s by config",
                "on" if override else "off (raw prompts)",
            )
        extra = _merge_template_kwargs(self.config.chat_template_kwargs)
        if extra and self._tokenizer is not None:
            # Inject the kwargs into every template call (flat and conversation
            # paths alike); explicit per-call kwargs still win.
            orig = self._tokenizer.apply_chat_template

            def _apply(conversation, *args, **kwargs):
                return orig(conversation, *args, **{**extra, **kwargs})

            self._tokenizer.apply_chat_template = _apply  # type: ignore[method-assign]
            logger.info("Rollout: chat_template_kwargs %s", extra)
        assert self._node is not None
        # Gate every handler that touches the engine; re-exposing replaces the
        # base registration. The bound handlers keep their own serdes, so the
        # wrappers pass the raw RPC arguments through.
        def _gated(fn: Any, mode: str) -> Any:
            async def handler(*args: Any, **kwargs: Any) -> Any:
                gate = self._gate.shared() if mode == "shared" else self._gate.exclusive()
                async with gate:
                    return await fn(*args, **kwargs)

            return handler

        self._node.expose("generate")(_gated(self._generate, "shared"))
        self._node.expose("update_weights")(_gated(self._update_weights, "exclusive"))
        self._node.expose("sleep")(_gated(self._sleep, "exclusive"))
        self._node.expose("wake")(_gated(self._wake, "exclusive"))

    async def start(self) -> None:
        await super().start()
        assert self._node is not None
        extra = max(1, int(self.config.n_rpc_workers)) - 1  # the node runs one itself
        for _ in range(extra):
            self._node._tasks.append(  # noqa: SLF001 — cancelled by Node.stop()
                self._node.loop.create_task(self._node.process_rpc_requests())
            )
        logger.info("Rollout: %d concurrent RPC worker(s)", extra + 1)

    def _extra_sampling_params(self, batch: PromptBatch) -> dict:
        constraint = batch.sampling_constraint or {}
        unknown = set(constraint) - _CONSTRAINT_KEYS
        if unknown or len(constraint) > 1:
            raise ValueError(
                f"sampling_constraint takes one of {sorted(_CONSTRAINT_KEYS)}; got {sorted(constraint)}"
            )
        return dict(constraint)

    def _template_prompts(self, batch: PromptBatch) -> list[str]:
        if batch.conversations is None:
            return super()._template_prompts(batch)
        if not self._use_chat_template:
            raise RuntimeError(
                "PromptBatch carries conversations but chat templating is off "
                "(apply_chat_template=false, or no template in the tokenizer) — "
                "refusing to flatten structured conversations to raw text"
            )
        return [
            apply_chat_template_messages(self._tokenizer, conv, batch.tools)
            for conv in batch.conversations
        ]

    def _postprocess_result(
        self, batch: PromptBatch, completions: list[str], result: RolloutResult
    ) -> None:
        parser_name = self.config.tool_call_parser
        if parser_name and batch.tools is not None and batch.conversations is not None:
            result.tool_calls = _parse_tool_calls(completions, batch.tools, parser_name)


if __name__ == "__main__":
    main(AgentRolloutNode, AgentRolloutConfig)
