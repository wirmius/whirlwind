"""Reflect-and-retry GSM8K agent — a small multi-turn task (EXPERIMENTAL).

A self-contained multi-turn agent for the agentic mode
(docs/agentic_multiturn.md): it solves GSM8K problems, and when an attempt is
wrong the environment asks the model to reflect and try again ("Your answer is
incorrect. Reflect on your solution and try to answer the previous question
again."), up to ``--max-turns`` turns per episode. It needs no containers or
tools, which makes it the quickest end-to-end check of multi-turn training:
``TASK=reflect examples/experimental_agentic.sh`` runs it against the agentic
orchestrator (:mod:`whirlwind.training.training_agentic`).

Episode / group semantics::

    group   = K episodes of the same problem (``--group-size``), whose returns
              are z-scored against each other (K >= 2)
    episode = 1..max_turns turns; an episode that is solved early stops early,
              so a group holds episodes of different lengths
    turn    = one user observation + one assistant completion

Wire protocol: one ``generate_external`` call per (episode, turn) with
``group_size=1``, the episode's identity riding in ``metadata["ww_tag"]``::

    {"task": "gsm8k-reflect", "problem": <sha256[:16] of the prompt>,
     "group": <id>, "group_size": K, "episode": <0..K-1>, "turn": <t>,
     "max_turns": T, "terminal": <t == T>, "reference": <answer>}

When an episode ends, the agent reports its return for EVERY turn's
``rollout_id`` in a single ``report_reward`` call (the agentic orchestrator
counts an episode as arrived once its turns are popped, so they must become
ready together). The orchestrator regroups turns by the tag and gives every
turn its episode's group-relative advantage.

Two carriers (``--carrier``):

    flat      the conversation so far is flattened into one user prompt
              (problem, then each previous answer + the retry message) — works
              with base models and plain prompt templating;
    messages  the real chat: system / user / assistant turns as
              ``conversations`` — needs the agentic rollout node with a chat
              template (``rollout_sglang_agent``).

Reward (terminal, one scalar per episode)::

    correct            -> discount ** (turn - 1)   (default discount 1.0)
    incorrect (budget) -> 0.0

First-try correctness is logged separately as ``reflect/first_try_acc``.

Usage (with the cluster already up)::

    python3 -m whirlwind.training.sim_reflect_agent \
        --group-size 4 --max-turns 3 --waves 50
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from whirlwind.messages import (
    BatchRequest,
    BatchResult,
    ExternalGenerateRequest,
    ExternalGenerateResult,
    ReportRewardRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.training.sim_external_agent import _load_accuracy_regex
from whirlwind.whirlcoms import Node, RpcError

logger = logging.getLogger(__name__)

DEFAULT_RETRY_TEMPLATE = (
    "Your answer is incorrect. Reflect on your solution and try to answer "
    "the previous question again."
)

CARRIERS = ("flat", "messages")


@dataclass
class ReflectEpisode:
    """Mutable per-episode state (one problem, one member of the group)."""

    group_id: int
    episode: int
    problem: str
    reference: str
    max_turns: int
    problem_sha: str
    group_size: int = 1
    turn: int = 1
    terminal: bool = False
    correct: bool = False
    first_try_correct: bool = False
    reward: float | None = None
    rollout_ids: list[str] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)  # assistant text, per turn

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def render_prompt(self, retry_template: str) -> str:
        """The conversation so far as one flat user prompt.

        Turn 1 is the bare problem (the same traffic as ``sim_external_agent``);
        turn t>1 appends each previous attempt and the retry message.
        Deterministic, so a trajectory is reconstructible from
        ``(problem, attempts)`` alone.
        """
        if self.turn == 1:
            return self.problem
        parts = [self.problem]
        for attempt in self.attempts:
            parts.append(f"Your previous answer:\n{attempt.strip()}")
            parts.append(retry_template)
        return "\n\n".join(parts)

    def render_messages(self, retry_template: str, system_prompt: str | None) -> list[dict[str, Any]]:
        """The conversation so far as chat messages (the ``messages`` carrier)."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.problem})
        for attempt in self.attempts:
            messages.append({"role": "assistant", "content": attempt})
            messages.append({"role": "user", "content": retry_template})
        return messages

    # ------------------------------------------------------------------
    # Tag (metadata["ww_tag"])
    # ------------------------------------------------------------------

    def make_tag(self) -> dict[str, Any]:
        """Request-time tag. ``terminal`` means "last turn by budget"; an
        early success ends the episode through the reward report."""
        return {
            "ww_tag": {
                "task": "gsm8k-reflect",
                "problem": self.problem_sha,
                "group": self.group_id,
                "group_size": self.group_size,
                "episode": self.episode,
                "turn": self.turn,
                "max_turns": self.max_turns,
                "terminal": self.turn >= self.max_turns,
                "reference": self.reference,
            }
        }

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def record_turn(self, rollout_id: str, completion: str, correct: bool, discount: float) -> None:
        """Advance the episode by one graded turn."""
        self.rollout_ids.append(rollout_id)
        self.attempts.append(completion)
        if self.turn == 1:
            self.first_try_correct = correct
        if correct or self.turn >= self.max_turns:
            self.terminal = True
            self.correct = correct
            self.reward = (discount ** (self.turn - 1)) if correct else 0.0
        else:
            self.turn += 1

    @property
    def n_turns(self) -> int:
        return len(self.attempts)


def grade_gsm8k(completion: str, reference: str, pattern: re.Pattern[str]) -> bool:
    """Same accuracy rule as the reward node's ``accuracy`` type / the sim."""
    ref_match = pattern.search(reference)
    if not ref_match:
        return False
    ref_answer = ref_match.group(1).replace(",", "")
    comp_match = pattern.search(completion)
    return bool(comp_match and comp_match.group(1).replace(",", "") == ref_answer)


class SimReflectAgent:
    """Drives the rollout proxy with reflect-and-retry multi-turn traffic."""

    def __init__(
        self,
        *,
        node_id: str = "sim-reflect-agent",
        proxy_node: str = "rollout",
        dataloader_node: str = "dataloader",
        group_size: int = 4,
        max_turns: int = 3,
        accuracy_regex: str = r"#####?\s*(\d[\d,]*)",
        carrier: str = "flat",
        discount: float = 1.0,
        retry_template: str = DEFAULT_RETRY_TEMPLATE,
        problems_per_wave: int = 0,
        waves: int = 0,
        sleep_s: float = 0.0,
        client_id: str = "sim-reflect",
    ) -> None:
        if group_size < 2:
            logger.warning(
                "group_size=%d: a one-episode group has no baseline, so every "
                "advantage is zero; use >= 2 for anything but a smoke test",
                group_size,
            )
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if carrier not in CARRIERS:
            raise ValueError(f"carrier must be one of {CARRIERS}")
        if not (0.0 < discount <= 1.0):
            raise ValueError("discount must be in (0, 1]")
        self.node_id = node_id
        self.proxy_node = proxy_node
        self.dataloader_node = dataloader_node
        self.group_size = max(1, group_size)
        self.max_turns = max_turns
        self.accuracy_regex = accuracy_regex
        self.carrier = carrier
        self.discount = discount
        self.retry_template = retry_template
        self.problems_per_wave = problems_per_wave  # 0 = whole fetched batch
        self.waves = waves  # 0 = until stopped
        self.sleep_s = sleep_s
        self.client_id = client_id
        self._node: Node | None = None
        self._group_counter = 0
        self._pattern = re.compile(self.accuracy_regex)

    @property
    def is_running(self) -> bool:
        return self._node.is_running if self._node is not None else False

    async def start(self) -> None:
        self._node = Node(self.node_id, auto_process=True)
        await self._node.start()
        logger.info(
            "Sim reflect agent '%s' started (proxy=%s, K=%d, max_turns=%d, carrier=%s)",
            self.node_id,
            self.proxy_node,
            self.group_size,
            self.max_turns,
            self.carrier,
        )

    async def stop(self) -> None:
        if self._node is not None:
            await self._node.stop()

    # ------------------------------------------------------------------
    # Per-(episode, turn) serving — one call, self-describing tag
    # ------------------------------------------------------------------

    async def _serve_one(self, ep: ReflectEpisode, system_prompt: str | None) -> ExternalGenerateResult:
        assert self._node is not None
        prompt = ep.render_prompt(self.retry_template)
        if self.carrier == "messages":
            # The system prompt travels inside the conversation.
            request = ExternalGenerateRequest(
                prompts=[prompt],
                conversations=[ep.render_messages(self.retry_template, system_prompt)],
                group_size=1,
                client_id=self.client_id,
                metadata=ep.make_tag(),
            )
        else:
            request = ExternalGenerateRequest(
                prompts=[prompt],
                system_prompt=system_prompt,
                group_size=1,
                client_id=self.client_id,
                metadata=ep.make_tag(),
            )
        return await self._node.call_typed(
            self.proxy_node, "generate_external", ExternalGenerateResult, request
        )

    # ------------------------------------------------------------------
    # The agent loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        assert self._node is not None
        wave = 0
        entries_served = 0
        while self.is_running and (self.waves <= 0 or wave < self.waves):
            t0 = time.monotonic()
            try:
                episodes, system_prompt = await self._new_wave(wave)
                if not episodes:
                    logger.warning("wave %d: no problems fetched — sleeping 5s", wave)
                    await asyncio.sleep(5.0)
                    continue
                await self._drive_wave(episodes, system_prompt)
            except RpcError as exc:
                # Mid-wave RPC failure: already-served turns of pending
                # episodes never get rewards — they TTL out. That is the
                # fail-closed path (and exercises it); log loudly and move on.
                logger.error("wave %d failed: %s — pending turns will TTL out", wave, exc)
                wave += 1
                continue

            terminal = [ep for ep in episodes if ep.terminal]
            assert len(terminal) == len(episodes), "drive_wave must run episodes to completion"
            n_correct = sum(1 for ep in terminal if ep.correct)
            n_first = sum(1 for ep in terminal if ep.first_try_correct)
            mean_turns = sum(ep.n_turns for ep in terminal) / len(terminal)
            wave_entries = sum(ep.n_turns for ep in terminal)
            entries_served += wave_entries
            wave += 1
            elapsed = time.monotonic() - t0
            self._node.call_bg_no_return(
                "tracker",
                "log_metrics",
                {
                    "metrics": {
                        "reflect/waves": float(wave),
                        "reflect/episodes": float(len(terminal)),
                        "reflect/entries_served": float(entries_served),
                        "reflect/final_acc": n_correct / len(terminal),
                        "reflect/first_try_acc": n_first / len(terminal),
                        "reflect/mean_turns": mean_turns,
                        "reflect/wave_time_s": elapsed,
                    },
                    "node_id": self.node_id,
                },
            )
            logger.info(
                "wave %d: %d episodes, %d turns (mean %.2f), final_acc=%.3f, "
                "first_try_acc=%.3f (%.1fs)",
                wave,
                len(terminal),
                wave_entries,
                mean_turns,
                n_correct / len(terminal),
                n_first / len(terminal),
                elapsed,
            )

            if self.sleep_s > 0:
                await asyncio.sleep(self.sleep_s)

        logger.info(
            "Sim reflect agent done: %d waves, %d entries served", wave, entries_served
        )

    async def _new_wave(self, wave: int) -> tuple[list[ReflectEpisode], str | None]:
        """Fetch problems the way any data consumer would; expand to K episodes
        per problem (one group per problem). Returns the episodes and the
        batch's few-shot system prompt (chat template applied engine-side)."""
        assert self._node is not None
        raw = await self._node.call(
            self.dataloader_node,
            "next_batch",
            _to_dict(BatchRequest(dataset_name="train", step=wave)),
        )
        if raw is None:
            raise RpcError("dataloader returned None")
        batch = _from_dict(BatchResult, raw)
        if not batch.ok:
            raise RpcError(f"dataloader error: {batch.error}")
        prompts = batch.to_prompt_batch()
        problems = prompts.prompts
        if self.problems_per_wave > 0:
            problems = problems[: self.problems_per_wave]
        references = prompts.responses if prompts.responses else [""] * len(problems)

        episodes: list[ReflectEpisode] = []
        for problem, reference in zip(problems, references, strict=True):
            self._group_counter += 1
            problem_sha = hashlib.sha256(problem.encode()).hexdigest()[:16]
            for episode in range(self.group_size):
                episodes.append(
                    ReflectEpisode(
                        group_id=self._group_counter,
                        episode=episode,
                        problem=problem,
                        reference=reference,
                        max_turns=self.max_turns,
                        problem_sha=problem_sha,
                        group_size=self.group_size,
                    )
                )
        return episodes, prompts.system_prompt

    async def _drive_wave(self, episodes: list[ReflectEpisode], system_prompt: str | None) -> None:
        """Run every episode to termination, turn-synchronously: each round
        serves all pending episodes' current turns concurrently (one RPC
        each), grades, advances, and reports rewards for episodes that
        finished this round."""
        assert self._node is not None
        pending = list(episodes)
        while pending:
            results = await asyncio.gather(
                *(self._serve_one(ep, system_prompt) for ep in pending)
            )
            newly_terminal: list[ReflectEpisode] = []
            for ep, result in zip(pending, results, strict=True):
                rollout_id = result.rollout_ids[0]
                completion = result.completions[0]
                correct = grade_gsm8k(completion, ep.reference, self._pattern)
                ep.record_turn(rollout_id, completion, correct, self.discount)
                if ep.terminal:
                    newly_terminal.append(ep)
            if newly_terminal:
                await self._report_terminal(newly_terminal)
            pending = [ep for ep in pending if not ep.terminal]

    async def _report_terminal(self, terminal: list[ReflectEpisode]) -> None:
        """Report each finished episode's return for every one of its turns,
        all in one call (so an episode's turns become ready together)."""
        assert self._node is not None
        rollout_ids: list[str] = []
        rewards: list[float] = []
        for ep in terminal:
            assert ep.reward is not None
            rollout_ids.extend(ep.rollout_ids)
            rewards.extend([ep.reward] * len(ep.rollout_ids))
        reply = await self._node.call(
            self.proxy_node,
            "report_reward",
            _to_dict(ReportRewardRequest(rollout_ids=rollout_ids, rewards=rewards)),
        )
        if not (isinstance(reply, dict) and reply.get("ok")):
            logger.warning("report_reward not ok: %r", reply)


async def _main(args: argparse.Namespace) -> None:
    agent = SimReflectAgent(
        node_id=args.node_id,
        proxy_node=args.proxy,
        dataloader_node=args.dataloader,
        group_size=args.group_size,
        max_turns=args.max_turns,
        accuracy_regex=_load_accuracy_regex(args.reward_config),
        carrier=args.carrier,
        discount=args.discount,
        retry_template=args.retry_template,
        problems_per_wave=args.problems_per_wave,
        waves=args.waves,
        sleep_s=args.sleep,
        client_id=args.client_id,
    )
    await agent.start()
    try:
        await agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reflect-and-retry GSM8K agent for the agentic mode (EXPERIMENTAL)"
    )
    parser.add_argument("--node-id", default="sim-reflect-agent")
    parser.add_argument("--proxy", default="rollout", help="rollout proxy node id")
    parser.add_argument("--dataloader", default="dataloader", help="dataloader node id")
    parser.add_argument(
        "--group-size", type=int, default=4, help="episodes per problem (the GRPO group)"
    )
    parser.add_argument(
        "--max-turns", type=int, default=3, help="turn budget per episode (retry loop cap)"
    )
    parser.add_argument(
        "--reward-config",
        default=None,
        help="reward config YAML to read the accuracy regex from (default: GSM8K pattern)",
    )
    parser.add_argument(
        "--carrier",
        choices=CARRIERS,
        default="flat",
        help="flat: the conversation flattened into one user prompt (base models); "
        "messages: real chat turns (needs the agentic rollout node with a chat template)",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=1.0,
        help="per-extra-turn reward discount (1.0 = final-answer accuracy only)",
    )
    parser.add_argument(
        "--retry-template",
        default=DEFAULT_RETRY_TEMPLATE,
        help="the follow-up user message sent after an incorrect attempt",
    )
    parser.add_argument(
        "--problems-per-wave",
        type=int,
        default=0,
        help="problems per wave (0 = whole fetched batch; each problem -> K episodes)",
    )
    parser.add_argument("--waves", type=int, default=0, help="stop after N waves (0 = until stopped)")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between waves")
    parser.add_argument("--client-id", default="sim-reflect")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(_main(args))
