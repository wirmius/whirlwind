"""SWE-Gym agent — containerized bash episodes for the agentic mode (EXPERIMENTAL).

A multi-turn tool-using agent for the agentic mode (docs/agentic_multiturn.md).
It drives SWE-Gym task instances (``SWE-Gym/SWE-Gym-Lite``) in their
prebuilt container environments
(``xingyaoww/sweb.eval.x86_64.<sanitized-instance>:latest``, run with podman):
one ``generate_external`` call per (episode, turn), the episode's identity in
``metadata["ww_tag"]``, a ``touch`` heartbeat per turn, and swebench-style
test verification at the end of the episode as the reward.

Episode / group semantics (the same tag grammar as ``sim_reflect_agent``)::

    group   = N episodes of the same instance (``--episodes-per-instance``)
    episode = 1..max_turns turns against a fresh container's /testbed
    turn    = one model action (a bash command) + one observation

Each episode gets its own container (fresh from the image, copy-on-write).
The loop is bash-only with a linear history: the trajectory is the prompt
sequence, and every action is an independent exec.

Reward (terminal, one scalar per episode, swebench "resolved")::

    agent diff collected from the container working tree
    -> working tree reset -> test_patch applied -> agent diff re-applied
    -> FAIL_TO_PASS all pass AND PASS_TO_PASS no regressions  -> 1.0
    anything else (patch does not apply, timeouts, failures)  -> 0.0

Fail-closed by construction: the reset discards agent edits to test files (no
reward hacking through tests), unapplicable diffs and suite timeouts score 0.

Two prompt styles: ``chat`` (instruct models: the chat template renders a
system / user / assistant / tool conversation with native ``bash`` and
``finish`` tool calls — needs ``rollout_sglang_agent``) and ``fewshot`` (base
models: contiguous in-prompt examples ending at ``Command:``). A ``toy`` task
family reuses one pulled image with small synthetic fixes, for wiring checks.

Requirements: podman, the ``datasets`` package, and the instance images
(pulled on first use, several GB each). Small base models solve essentially
no real instances; use an instruct model of a few billion parameters or more
to see a learning signal.

Usage (with the agentic cluster already up)::

    python3 -m whirlwind.training.sim_swe_agent \
        --repo getmoto/moto --n-instances 8 --episodes-per-instance 4 \
        --max-turns 10 --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any

from whirlwind.messages import (
    ExternalGenerateRequest,
    ExternalGenerateResult,
    ReportRewardRequest,
    TouchRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.rollout.rollout_sglang_agent import escape_special_tokens
from whirlwind.whirlcoms import Node, RpcError

logger = logging.getLogger(__name__)

DATASET_ID = "SWE-Gym/SWE-Gym-Lite"
IMAGE_PREFIX = "xingyaoww/sweb.eval.x86_64"
TAG_VERBATIM = "1.0"

# Prompt styles. "chat": the model's default chat template drives everything —
# multi-turn conversation (system / user / assistant / tool roles), native
# tool calls (bash + finish), tool responses as tool-role messages. Requires
# an instruct-class model and apply_chat_template on the engine side.
# "fewshot": contiguous in-prompt examples with a bare 'Command:' continuation
# — the style base models actually pattern-complete (engine raw mode).
STYLE_CHAT = "chat"
STYLE_FEWSHOT = "fewshot"

# Repetition guard: a byte-identical command sent REPEAT_NUDGE_AFTER
# times consecutively is not executed — the model gets a nudge tool
# result instead; one more identical send terminates the episode
# (budget protection; looping trajectories are poor training data).
REPEAT_NUDGE_AFTER = 3
REPEAT_TERMINATE_AFTER = 4
REPEAT_NUDGE_TEXT = (
    "This exact command was already run and produced the same result. "
    "Do not repeat it. Read the previous output carefully, change your "
    "approach, or call finish."
)

CHAT_SYSTEM_PROMPT = (
    "You are an autonomous software engineer working in a repository checkout "
    "at /testbed. Use the bash tool to inspect files, edit code, and run "
    "commands — exactly one command per tool call; its combined output is "
    "returned to you as the tool result. Work step by step until the task is "
    "done, then call the finish tool (you may also finish by responding with "
    "just: END). Never call finish before you have run at least one bash "
    "command — always investigate the repository first."
)

# OpenAI-style tool schemas rendered by the chat template (Qwen3 default
# template emits the tool docs plus the hermes tool_call grammar).
BASH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in the repository at /testbed and return "
            "its combined stdout/stderr (truncated for length)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

FINISH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call when the task is complete; ends the episode.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous software engineer working in a repository checkout "
    "at /testbed. Solve the issue described below by modifying the code.\n"
    "Respond with exactly ONE bash command per message (no explanations, no "
    "markdown fences) — it will be executed in /testbed and you will see its "
    "output. Use the available tools: grep, find, sed, python, git diff, and "
    "the project's own commands. When you believe the issue is fixed, or you "
    "cannot make progress, respond with just: END"
)

# Observation truncation budget (chars): keep head+tail so stack traces and
# summaries survive while dumps/pager noise is cut (mini-swe-agent policy).
OBS_HEAD_CHARS = 1500
OBS_TAIL_CHARS = 500

# Verification budget (seconds, inside the container).
VERIFY_TIMEOUT_S = 600.0
EXEC_TIMEOUT_S = 120.0


def image_name(instance_id: str) -> str:
    """Docker Hub ref for a SWE-Gym instance image (sanitized: __ -> _s_)."""
    return f"{IMAGE_PREFIX}.{instance_id.replace('__', '_s_').lower()}:latest"


def container_name(run_id: str, instance_id: str, episode: int) -> str:
    return f"{run_id}-{instance_id.replace('__', '-').lower()}-e{episode}"


@dataclass
class SweInstance:
    """One SWE-Gym task instance (agent-owned curriculum, by design)."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    # Optional container setup (toy family): commands run after container
    # creation, BEFORE the agent's first turn (e.g. pre-create a broken file).
    setup_cmds: list[str] = field(default_factory=list)
    # Image override (toy family reuses one pulled SWE-Gym image); defaults
    # to the instance's own prebuilt image.
    image: str = ""

    def __post_init__(self) -> None:
        if not self.image:
            self.image = image_name(self.instance_id)


@dataclass
class SweEpisode:
    """Mutable per-episode state (one instance, one member of the group)."""

    instance: SweInstance
    group_id: int
    episode: int
    max_turns: int
    run_id: str
    group_size: int = 1
    turn: int = 1
    terminal: bool = False
    ended_early: bool = False
    reward: float | None = None
    rollout_ids: list[str] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)  # (action, obs)
    n_cmds_failed: int = 0
    # Repetition-guard state: last executed/sent command + consecutive
    # identical count (reset on any different action).
    last_action: str = ""
    n_identical: int = 0
    n_nudged: int = 0
    # Prompt style: "chat" (chat-template conversations + native tool calls)
    # or "fewshot" (contiguous in-prompt examples, 'Command:' continuation —
    # the style base models pattern-complete; engine raw mode).
    prompt_style: str = STYLE_CHAT
    fewshot: str = ""

    # ------------------------------------------------------------------
    # Prompt (linear history — trajectory == messages, training-ready)
    # ------------------------------------------------------------------
    def render_prompt(self, system_prompt: str) -> str:
        if self.prompt_style == "fewshot":
            # Contiguous pattern-completion text: examples, issue, then the
            # running Command:/Output: history, ending at 'Command:' so the
            # model completes exactly one command line.
            parts = [self.fewshot.rstrip(), f"Issue: {self.instance.problem_statement}"]
            for action, obs in self.history:
                parts.append(f"Command: {action}\nOutput: {obs}")
            parts.append("Command:")
            return "\n\n".join(parts)
        raise ValueError(
            f"render_prompt is only for the fewshot style (got {self.prompt_style!r}); "
            "chat-style episodes build conversations (build_chat_messages)"
        )

    # Chat-style conversation history (STYLE_CHAT). Trajectory == messages:
    # each turn generation prompt is exactly these messages (the template
    # appends the assistant generation prompt). Untrusted text is escaped
    # as it enters; the engine re-escapes idempotently (layers stack).
    messages: list[dict[str, Any]] = field(default_factory=list)

    def seed_chat_messages(self, system_prompt: str) -> None:
        # Seed: system role + the issue as the (single) user turn.
        self.messages = [
            {"role": "system", "content": escape_special_tokens(system_prompt)},
            {"role": "user", "content": escape_special_tokens(
                self.instance.problem_statement)},
        ]

    def append_tool_round(
        self, call_id: str, name: str, arguments: Any, obs: str
    ) -> None:
        # Append one assistant tool-call + tool-result round to the history.
        # function.arguments must be a real dict: some chat templates iterate
        # arguments.items() and fail on a JSON string ("Can only get item
        # pairs from a mapping"); hermes-style templates accept either form.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        self.messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }],
        })
        self.messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": escape_special_tokens(obs),
        })

    # ------------------------------------------------------------------
    # Turn accounting
    # ------------------------------------------------------------------
    def advance_turn(self) -> None:
        """One round consumed — executed, malformed, or unparseable.

        A burned round is still a round: it consumed the model's turn and a
        buffer entry, so it consumes the turn budget like any other. Otherwise
        a model that never produces a parseable call would retry turn 1
        forever and flood the buffer.
        """
        if self.turn >= self.max_turns:
            self.terminal = True
        else:
            self.turn += 1

    # ------------------------------------------------------------------
    # Tag (metadata["ww_tag"] — every buffer entry is self-describing)
    # ------------------------------------------------------------------
    def make_tag(self) -> dict[str, Any]:
        return {
            "ww_tag": {
                "task": "swe-gym-lite",
                "problem": self.instance.instance_id,
                "group": self.group_id,
                "group_size": self.group_size,
                "episode": self.episode,
                "turn": self.turn,
                "max_turns": self.max_turns,
                "terminal": self.turn >= self.max_turns,
                "reference": self.instance.instance_id,
            }
        }


def extract_action(completion: str) -> str | None:
    """Model output -> bash command, or None = end the episode.

    Tolerates markdown fences and stray prose around the command; an empty
    extraction or a bare END ends the episode.
    """
    text = completion.strip()
    if not text:
        return None
    fence = re.search(r"```(?:bash|sh|shell)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text or text.split()[0].upper() in ("END", "END."):
        return None
    # Strip a leading shell-prompt or "$ " if the model imitated one.
    text = re.sub(r"^[$>]\s*", "", text, count=1)
    return text


# Qwen3 tool-call structure markers, built by concatenation so the
# literal special sequences never appear in source (they corrupt tooling).
_TOOL_CALL_OPEN = "<tool" + "_call>"
_TOOL_CALL_CLOSE = "</tool" + "_call>"
# XML function grammar (the qwen3_coder-style tool-call format), built by
# concatenation like the hermes markers above.
_FN_OPEN = "<function="
_FN_CLOSE = "</function" + ">"
_PARAM_OPEN = "<parameter="
_PARAM_CLOSE = "</parameter" + ">"


def parse_tool_call(completion: str) -> dict[str, Any] | None:
    # First tool call in a completion -> {name, arguments}; None = no call.
    # A tool_call wrapper may hold EITHER grammar (both fail-closed):
    #   hermes JSON (Qwen3/3.5 instruct):
    #       <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    #   XML (qwen3_coder-style templates):
    #       <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
    # Hermes is tried first and wins when valid; anything else falls through
    # to the XML parser on the SAME payload — the wrapper alone says nothing
    # about the payload grammar.
    # Tolerates truncated closing markers and arguments given either as a
    # dict or as a JSON-encoded string (Qwen3 emits the string form).
    # Fail-closed: any structural surprise returns None (nothing executed).
    start = completion.find(_TOOL_CALL_OPEN)
    if start < 0:
        return _parse_xml_tool_call(completion)
    rest = completion[start + len(_TOOL_CALL_OPEN):]
    end = rest.find(_TOOL_CALL_CLOSE)
    payload = rest if end < 0 else rest[:end]
    call = _parse_hermes_call(payload)
    if call is not None:
        return call
    return _parse_xml_tool_call(payload)


def _parse_hermes_call(payload: str) -> dict[str, Any] | None:
    # JSON tool-call body: {"name": ..., "arguments": {...}}. None = not a
    # valid hermes call (caller falls through to the XML grammar). Braces
    # inside an XML command value (sed 's/{a}/{b}/') land here, fail JSON
    # validation, and correctly fall through.
    brace_open = payload.find("{")
    brace_close = payload.rfind("}")
    if brace_open < 0 or brace_close <= brace_open:
        return None
    try:
        obj = json.loads(payload[brace_open:brace_close + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict) or not isinstance(name, str):
        return None
    return {"name": name, "arguments": args}


def _parse_xml_tool_call(completion: str) -> dict[str, Any] | None:
    # <function=NAME> <parameter=K>V</parameter> ... </function> grammar.
    # Fail-closed like the hermes parser: structural surprises -> None.
    open_at = completion.find(_FN_OPEN)
    if open_at < 0:
        return None
    name_start = open_at + len(_FN_OPEN)
    name_end = completion.find(">", name_start)
    if name_end < 0 or name_end == name_start:
        return None
    name = completion[name_start:name_end].strip()
    if not name:
        return None
    close_at = completion.find(_FN_CLOSE, name_end)
    body = completion[name_end + 1:close_at if close_at >= 0 else len(completion)]
    args: dict[str, Any] = {}
    pos = 0
    while True:
        p_open = body.find(_PARAM_OPEN, pos)
        if p_open < 0:
            break
        k_start = p_open + len(_PARAM_OPEN)
        k_end = body.find(">", k_start)
        if k_end < 0:
            return None
        key = body[k_start:k_end].strip()
        v_open = body.find(_PARAM_CLOSE, k_end)
        if v_open < 0 or not key:
            return None
        value = body[k_end + 1:v_open]
        args[key] = value.strip("\n")
        pos = v_open + len(_PARAM_CLOSE)
    return {"name": name, "arguments": args}


def truncate_obs(text: str, head: int = OBS_HEAD_CHARS, tail: int = OBS_TAIL_CHARS) -> str:
    if len(text) <= head + tail + 64:
        return text
    return (
        text[:head]
        + f"\n[... output truncated: {len(text)} chars total ...]\n"
        + text[-tail:]
    )


# ---------------------------------------------------------------------------
# Container task runner (podman; any OCI runtime with run/exec/rm works)
# ---------------------------------------------------------------------------


class ContainerRunner:
    """Async podman wrapper: fresh container per episode, exec per turn."""

    def __init__(self, exec_timeout_s: float = EXEC_TIMEOUT_S) -> None:
        self.exec_timeout_s = exec_timeout_s

    async def _run(self, *argv: str, timeout: float, stdin: bytes | None = None) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, f"[exec timed out after {timeout:.0f}s]"
        return proc.returncode or 0, out.decode(errors="replace")

    async def create(self, name: str, image: str) -> None:
        rc, out = await self._run(
            "podman", "run", "-d", "--name", name, image, "sleep", "infinity",
            timeout=120.0,
        )
        if rc != 0:
            raise RuntimeError(f"podman run {name} failed (rc={rc}): {out[-400:]}")

    async def exec(self, name: str, command: str, timeout: float | None = None) -> tuple[int, str]:
        return await self._run(
            "podman", "exec", "-w", "/testbed", name, "bash", "-c", command,
            timeout=timeout or self.exec_timeout_s,
        )

    async def copy_in(self, name: str, local_path: str, dest: str) -> None:
        rc, out = await self._run(
            "podman", "cp", local_path, f"{name}:{dest}", timeout=60.0
        )
        if rc != 0:
            raise RuntimeError(f"podman cp into {name} failed: {out[-200:]}")

    async def remove(self, name: str) -> None:
        await self._run("podman", "rm", "-f", name, timeout=60.0)


def build_verify_script(instance: SweInstance) -> str:
    """The swebench-resolution check, as one in-container bash script.

    Fail-closed exits: 3 = test_patch unapplicable (environment broken),
    4 = agent diff unapplicable, 5/6 = F2P/P2P failures, 0+RESOLVED = 1.0.

    Order and reset semantics (fixed after the oracle test caught the
    index-restore bug): ``git add -A`` stages the agent's tree, so
    ``git checkout -- .`` restores FROM THE INDEX — a no-op. The reset must
    be ``git reset --hard HEAD`` (index AND worktree). The collected diff
    excludes any file the reference test_patch touches: agent edits to
    test files are DISCARDED (swebench semantics — the reference tests are
    authoritative; reward hacking via test edits is impossible), instead
    of failing the whole episode on a patch overlap.
    """
    f2p = " ".join(shlex.quote(t) for t in instance.fail_to_pass)
    p2p = " ".join(shlex.quote(t) for t in instance.pass_to_pass)
    # Toy instances have no regression suite: an empty P2P list must SKIP the
    # pytest call, not run the entire repo's test collection by accident.
    p2p_cmd = (
        f"timeout {int(VERIFY_TIMEOUT_S)} $PY -m pytest -q {p2p} "
        f"> /tmp/p2p.out 2>&1 || {{ echo P2P_FAIL; exit 6; }}"
        if p2p
        else "echo P2P_EMPTY"
    )
    return f"""set -u
cd /testbed
PY=/opt/miniconda3/envs/testbed/bin/python
# 1) collect the agent's diff, EXCLUDING files the reference tests touch
TEST_FILES=$(grep -E '^diff --git' /tmp/test.patch | awk '{{print $3}}' | sed 's|^a/||' | sort -u)
EXCL=""
for f in $TEST_FILES; do EXCL="$EXCL :(exclude)$f"; done
git add -A
git diff HEAD $EXCL > /tmp/agent.patch
# 2) hard reset: index AND worktree to the pristine base (checkout -- . after
#    add -A is a no-op — it restores from the index)
git reset --hard HEAD > /dev/null
git clean -fdq
# 3) apply the reference test patch (adds/updates the tests)
git apply /tmp/test.patch || {{ echo NO_TEST_PATCH; exit 3; }}
# 4) re-apply the agent's source diff (skip if it made no changes)
if [ -s /tmp/agent.patch ]; then
  git apply /tmp/agent.patch || {{ echo NO_AGENT_PATCH; exit 4; }}
fi
# 5) run the resolution suites
timeout {int(VERIFY_TIMEOUT_S)} $PY -m pytest -q -x {f2p} > /tmp/f2p.out 2>&1 || {{ echo F2P_FAIL; exit 5; }}
{p2p_cmd}
echo RESOLVED
exit 0"""


VERIFY_EXIT_REWARD = {3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 124: 0.0, 0: 1.0}

# ---------------------------------------------------------------------------
# Toy task family — solvable by a tiny non-aligned base model (few-shot
# pattern completion), running through the IDENTICAL production machinery:
# same containers/tags/diff-collection/reset/verification/buffer. Purpose:
# non-zero rewards end-to-end without a Coder-class model. Three families:
#   transcribe — create a file with exact content (echo ... > f)
#   empty      — create an empty file (touch f)
#   fixconst   — replace a file's content (setup pre-creates the broken file)
# ---------------------------------------------------------------------------

# The image all toy instances run in (already pulled; has git + the testbed
# conda env the verify script expects).
TOY_IMAGE = "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-5699:latest"

TOY_SYSTEM_PROMPT = (
    "You are an autonomous software engineer working in a repository checkout "
    "at /testbed. Solve the issue described below by modifying the code.\n"
    "Respond with exactly ONE bash command per message (no explanations, no "
    "markdown fences). When you are done, respond with just: END\n\n"
    "Example 1:\n"
    "Issue: Create a file named notes.txt containing exactly: buy milk\n"
    "Command: echo \"buy milk\" > notes.txt\n\n"
    "Example 2:\n"
    "Issue: The file greeting.py is empty. Write into it exactly: print('hi')\n"
    "Command: echo \"print('hi')\" > greeting.py\n\n"
    "Example 3:\n"
    "Issue: The file settings.py contains RATE = 0. Replace the file content "
    "with exactly: RATE = 5\n"
    "Command: echo \"RATE = 5\" > settings.py\n\n"
    "Example 4:\n"
    "Issue: Create an empty file named placeholder.tmp\n"
    "Command: touch placeholder.tmp\n"
)

# The few-shot BLOCK that goes into the user text (base models read THIS, not
# the system message) — includes an END-after-success example so episodes
# terminate instead of issuing extra commands that could break a solved state.
TOY_FEWSHOT = (
    "Example 1:\n"
    "Issue: Create a file named notes.txt containing exactly: buy milk\n"
    "Command: echo \"buy milk\" > notes.txt\n\n"
    "Example 2:\n"
    "Issue: The file greeting.py is empty. Write into it exactly: print('hi')\n"
    "Command: echo \"print('hi')\" > greeting.py\n\n"
    "Example 3:\n"
    "Issue: The file settings.py contains RATE = 0. Replace the file content "
    "with exactly: RATE = 5\n"
    "Command: echo \"RATE = 5\" > settings.py\n\n"
    "Example 4:\n"
    "Issue: Create an empty file named placeholder.tmp\n"
    "Command: touch placeholder.tmp\n\n"
    "Example 5:\n"
    "Issue: Create a file named final.txt containing exactly: done\n"
    "Command: echo \"done\" > final.txt\n"
    "Output: \n"
    "Command: END\n"
)

# Few-shot block for REAL SWE-Gym episodes with base models: teaches the
# workflow shape (locate -> edit -> run tests -> END), not file creation.
SWE_FEWSHOT = (
    "Example 1:\n"
    "Issue: The function add in utils.py returns the difference of its "
    "arguments instead of the sum. Fix it.\n"
    "Command: grep -rn \"def add\" .\n"
    "Output: ./utils.py:12:def add(a, b):\n"
    "Command: sed -i 's/return a - b/return a + b/' utils.py\n"
    "Output: \n"
    "Command: python -m pytest tests/test_utils.py -q\n"
    "Output: 2 passed in 0.5s\n"
    "Command: END\n\n"
    "Example 2:\n"
    "Issue: parse_config in config.py crashes when the file contains blank "
    "lines. Make it skip them.\n"
    "Command: grep -rn \"def parse_config\" .\n"
    "Output: ./config.py:30:def parse_config(path):\n"
    "Command: sed -i 's/lines = f.readlines()/lines = [l for l in f if l.strip()]/' config.py\n"
    "Output: \n"
    "Command: python -m pytest tests/test_config.py -q\n"
    "Output: 5 passed in 0.8s\n"
    "Command: END\n"
)


def _toy_test_patch(test_file: str, body_lines: list[str]) -> str:
    """A valid 'git apply'-able patch adding one test file."""
    body = "\n".join("+" + line for line in body_lines)
    return (
        f"diff --git a/{test_file} b/{test_file}\n"
        f"new file mode 100644\n"
        f"--- /dev/null\n"
        f"+++ b/{test_file}\n"
        f"@@ -0,0 +1,{len(body_lines)} @@\n"
        f"{body}\n"
    )


def make_toy_instances(n: int, image: str = TOY_IMAGE, seed: int = 1234) -> list[SweInstance]:
    """Deterministic toy instances a tiny base model can genuinely solve."""
    import random

    rng = random.Random(seed)
    kinds = ("transcribe", "empty", "fixconst")
    instances: list[SweInstance] = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        val = rng.randint(100, 999)
        tname = f"test_toy_{i:02d}"
        fname = f"taskfile_{i:02d}.txt"
        test_name = f"test_toy_{i:02d}_{kind}"
        if kind == "transcribe":
            content = f"answer is {val}"
            problem = (
                f"Create a file named {fname} containing exactly: {content}"
            )
            body = [
                f"def {test_name}():",
                f"    assert open('{fname}').read().strip() == '{content}'",
            ]
            setup: list[str] = []
        elif kind == "empty":
            problem = f"Create an empty file named {fname}"
            body = [
                "import os",
                f"def {test_name}():",
                f"    assert os.path.exists('{fname}') and os.path.getsize('{fname}') == 0",
            ]
            setup = []
        else:  # fixconst
            py = f"config_{i:02d}.py"
            problem = (
                f"The file {py} contains the line RATE = 0. Replace the file "
                f"content with exactly: RATE = {val}"
            )
            body = [
                f"def {test_name}():",
                f"    assert 'RATE = {val}' in open('{py}').read()",
            ]
            setup = [f"echo \"RATE = 0\" > /testbed/{py}"]
        instances.append(
            SweInstance(
                instance_id=f"toy-{i:02d}-{kind}",
                repo="toy/toy",
                base_commit="",
                problem_statement=problem,
                test_patch=_toy_test_patch(f"{tname}.py", body),
                fail_to_pass=[f"{tname}.py::{test_name}"],
                pass_to_pass=[],
                setup_cmds=setup,
                image=image,
            )
        )
    return instances


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class SimSweAgent:
    """Drives the rollout proxy with containerized SWE-Gym episodes."""

    def __init__(
        self,
        *,
        node_id: str = "sim-swe-agent",
        proxy_node: str = "rollout",
        repo: str = "getmoto/moto",
        n_instances: int = 8,
        episodes_per_instance: int = 2,
        max_turns: int = 10,
        concurrency: int = 16,
        system_prompt: str | None = None,
        exec_timeout_s: float = EXEC_TIMEOUT_S,
        client_id: str = "sim-swe",
        run_id: str = "swe-agent",
        task_family: str = "swe-gym",
        toy_image: str = TOY_IMAGE,
        prompt_style: str | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if task_family not in ("swe-gym", "toy"):
            raise ValueError(f"unknown task_family {task_family!r}")
        # Prompt style: explicit > task-family default (toy -> fewshot for
        # base models, swe-gym -> chat: native template + tool calls).
        if prompt_style is None:
            prompt_style = STYLE_FEWSHOT if task_family == "toy" else STYLE_CHAT
        if prompt_style not in (STYLE_CHAT, STYLE_FEWSHOT):
            raise ValueError(f"unknown prompt_style {prompt_style!r}")
        if episodes_per_instance < 2:
            logger.warning(
                "episodes_per_instance=%d: a one-episode group has no baseline, "
                "so every advantage is zero — use >= 2",
                episodes_per_instance,
            )
        self.node_id = node_id
        self.proxy_node = proxy_node
        self.repo = repo
        self.n_instances = n_instances
        self.episodes_per_instance = max(1, episodes_per_instance)
        self.max_turns = max_turns
        self.concurrency = max(1, concurrency)
        self.task_family = task_family
        self.toy_image = toy_image
        self.prompt_style = prompt_style
        self.system_prompt = system_prompt or (
            CHAT_SYSTEM_PROMPT if prompt_style == STYLE_CHAT
            else TOY_SYSTEM_PROMPT
        )
        # Tool schemas rendered into the chat template (chat style only).
        self.tools: list[dict[str, Any]] | None = (
            [BASH_TOOL_SCHEMA, FINISH_TOOL_SCHEMA]
            if prompt_style == STYLE_CHAT else None
        )
        self.client_id = client_id
        self.run_id = re.sub(r"[^a-z0-9-]", "-", run_id.lower())
        self._node: Node | None = None
        self._runner = ContainerRunner(exec_timeout_s=exec_timeout_s)
        self._group_counter = 0
        # Curriculum telemetry: per-instance solve rates (the zero-gradient
        # filter — all-0/all-1 groups contribute nothing once training is real).
        self.instance_stats: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._node = Node(self.node_id, auto_process=True)
        await self._node.start()
        logger.info(
            "Sim SWE agent '%s' started (proxy=%s, repo=%s, %d instances x %d "
            "episodes, max_turns=%d, concurrency=%d)",
            self.node_id, self.proxy_node, self.repo, self.n_instances,
            self.episodes_per_instance, self.max_turns, self.concurrency,
        )

    async def stop(self) -> None:
        if self._node is not None:
            await self._node.stop()

    # ------------------------------------------------------------------
    # Instances (agent-owned curriculum; no harness dataloader by design)
    # ------------------------------------------------------------------
    def load_instances(self) -> list[SweInstance]:
        if self.task_family == "toy":
            instances = make_toy_instances(self.n_instances, image=self.toy_image)
            for inst in instances:
                logger.info(
                    "toy instance %s: %s", inst.instance_id,
                    inst.problem_statement[:70],
                )
            return instances
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, split="train")
        rows = [r for r in ds if r["repo"] == self.repo]
        if not rows:
            raise ValueError(f"no instances for repo {self.repo!r} in {DATASET_ID}")
        # Fastest verification first (smallest suites) — keeps episodes cheap.
        rows.sort(key=lambda r: len(r["FAIL_TO_PASS"]) + len(r["PASS_TO_PASS"]))
        rows = rows[: self.n_instances]
        instances = [
            SweInstance(
                instance_id=r["instance_id"],
                repo=r["repo"],
                base_commit=r["base_commit"],
                problem_statement=r["problem_statement"],
                test_patch=r["test_patch"],
                fail_to_pass=list(r["FAIL_TO_PASS"] or []),
                pass_to_pass=list(r["PASS_TO_PASS"] or []),
            )
            for r in rows
        ]
        for inst in instances:
            logger.info(
                "instance %s: F2P=%d P2P=%d image=%s",
                inst.instance_id, len(inst.fail_to_pass), len(inst.pass_to_pass),
                inst.image,
            )
        return instances

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    async def run(self) -> None:
        assert self._node is not None
        instances = self.load_instances()
        episodes: list[SweEpisode] = []
        for inst in instances:
            self._group_counter += 1
            self.instance_stats.setdefault(inst.instance_id, [])
            for e in range(self.episodes_per_instance):
                episodes.append(
                    SweEpisode(
                        instance=inst,
                        group_id=self._group_counter,
                        episode=e,
                        max_turns=self.max_turns,
                        run_id=self.run_id,
                        group_size=self.episodes_per_instance,
                        prompt_style=self.prompt_style,
                        fewshot=(
                            (SWE_FEWSHOT if self.task_family == "swe-gym" else TOY_FEWSHOT)
                            if self.prompt_style == STYLE_FEWSHOT else ""
                        ),
                    )
                )
        sem = asyncio.Semaphore(self.concurrency)
        t0 = time.monotonic()
        results = await asyncio.gather(
            *(self._run_episode(ep, sem) for ep in episodes), return_exceptions=True
        )
        n_exc = sum(1 for r in results if isinstance(r, Exception))
        for r in results:
            if isinstance(r, Exception):
                logger.error("episode crashed: %s", r, exc_info=r)

        # Curriculum telemetry.
        logger.info("=== episode summary (%.0fs) ===", time.monotonic() - t0)
        n_resolved = 0
        for inst_id, rewards in sorted(self.instance_stats.items()):
            r_mean = sum(rewards) / len(rewards) if rewards else 0.0
            n_resolved += sum(rewards)
            logger.info(
                "instance %s: solved %d/%d (mean reward %.2f)",
                inst_id, int(sum(rewards)), len(rewards), r_mean,
            )
        total = len([r for r in results if not isinstance(r, Exception)])
        logger.info(
            "Sim SWE agent done: %d episodes (%d crashed), %d resolved, "
            "solve rate %.3f",
            total, n_exc, int(n_resolved),
            (n_resolved / total) if total else 0.0,
        )

    # ------------------------------------------------------------------
    # One episode (independent loop — episodes are NOT turn-synchronized;
    # container exec timing varies wildly and the proxy batches anyway)
    # ------------------------------------------------------------------
    @staticmethod
    def _repeat_guard(ep: SweEpisode, action: str) -> str | None:
        """Track consecutive identical commands; None | 'nudge' | 'terminate'.

        The FIRST send of a command always executes; the second too (a legit
        retry after an intervening state change is a DIFFERENT command by
        construction — identical-consecutive means nothing happened between).
        From REPEAT_NUDGE_AFTER consecutive sends the command is not executed
        and the model receives a nudge; at REPEAT_TERMINATE_AFTER the episode
        ends (fail-closed budget protection).
        """
        if action and action == ep.last_action:
            ep.n_identical += 1
        else:
            ep.last_action = action
            ep.n_identical = 1
            return None
        if ep.n_identical >= REPEAT_TERMINATE_AFTER:
            return "terminate"
        if ep.n_identical >= REPEAT_NUDGE_AFTER:
            ep.n_nudged += 1
            return "nudge"
        return None

    async def _run_episode(self, ep: SweEpisode, sem: asyncio.Semaphore) -> None:
        async with sem:
            assert self._node is not None
            ctr = container_name(ep.run_id, ep.instance.instance_id, ep.episode)
            await self._runner.create(ctr, ep.instance.image)
            try:
                # Ship the reference test patch up front (applied only at
                # verification time — the agent never sees the tests).
                import tempfile, os

                with tempfile.NamedTemporaryFile(
                    "w", suffix=".patch", delete=False
                ) as f:
                    f.write(ep.instance.test_patch)
                    patch_path = f.name
                try:
                    await self._runner.copy_in(ctr, patch_path, "/tmp/test.patch")
                finally:
                    os.unlink(patch_path)
                # Toy-family container setup (pre-create the broken state).
                for cmd in ep.instance.setup_cmds:
                    rc, out = await self._runner.exec(ctr, cmd)
                    if rc != 0:
                        raise RuntimeError(
                            f"setup cmd failed ({cmd[:60]}): {out[-200:]}"
                        )

                if ep.prompt_style == STYLE_CHAT:
                    ep.seed_chat_messages(self.system_prompt)

                while not ep.terminal:
                    if ep.prompt_style == STYLE_CHAT:
                        req = ExternalGenerateRequest(
                            prompts=[f"[chat {ep.instance.instance_id} turn {ep.turn}]"],
                            conversations=[list(ep.messages)],
                            tools=self.tools,
                            group_size=1,
                            metadata=ep.make_tag(),
                            client_id=self.client_id,
                        )
                    else:
                        req = ExternalGenerateRequest(
                            prompts=[ep.render_prompt(self.system_prompt)],
                            group_size=1,
                            metadata=ep.make_tag(),
                            client_id=self.client_id,
                        )
                    raw = await self._generate(req, first=(ep.turn == 1))
                    if raw is None:
                        raise RpcError("proxy returned None")
                    res = _from_dict(ExternalGenerateResult, raw)
                    rid = res.rollout_ids[0]
                    completion = res.completions[0]
                    ep.rollout_ids.append(rid)

                    # Heartbeat: keep the whole episode's entries alive.
                    if len(ep.rollout_ids) > 1:
                        tr = await self._node.call(
                            self.proxy_node, "touch",
                            _to_dict(TouchRequest(rollout_ids=ep.rollout_ids)),
                        )
                        if not (isinstance(tr, dict) and tr.get("ok")):
                            logger.warning("touch not ok for %s: %r", ctr, tr)

                    if ep.prompt_style == STYLE_CHAT:
                        # Engine-parsed structured calls (the engine's
                        # tool_call_parser, e.g. qwen3_coder) take
                        # precedence; the text parser is the fallback.
                        # Wire shape (messages.ExternalGenerateResult): one
                        # list of {"name", "arguments"} dicts PER
                        # COMPLETION. This episode's turn = completion 0's
                        # FIRST call; an empty list means "no call" and
                        # falls back to the text parser.
                        structured = None
                        if res.tool_calls and res.tool_calls[0]:
                            structured = res.tool_calls[0][0]
                        tc = (
                            structured
                            if structured is not None
                            else parse_tool_call(completion)
                        )
                        if tc is None:
                            stripped = completion.strip()
                            if stripped == "END":
                                logger.info(
                                    "turn %d/%d %s e%d: plain-text END",
                                    ep.turn, ep.max_turns,
                                    ep.instance.instance_id, ep.episode,
                                )
                                # Explicit plain-text finish (the only
                                # legitimate no-call finish; the finish turn
                                # is a trained entry).
                                ep.terminal = True
                                ep.ended_early = True
                                continue
                            # Unparseable completion — NOT a finish. Thinking
                            # models produce these often (a reasoning-only
                            # reply that ends right after the think block, or
                            # a generation cap inside it). Fail-closed round:
                            # the model sees its own reply + the error;
                            # nothing runs; it retries next turn.
                            ep.n_cmds_failed += 1
                            logger.info(
                                "turn %d/%d %s e%d: UNPARSED completion "
                                "(%d chars, head=%r)",
                                ep.turn, ep.max_turns,
                                ep.instance.instance_id, ep.episode,
                                len(completion), completion[:120],
                            )
                            ep.append_tool_round(
                                f"call-{ep.turn}", "bash", "{}",
                                "[tool error: no bash tool call found in "
                                "your reply — call the bash tool with a "
                                "command (one per reply), or reply END to "
                                "finish]",
                            )
                            # The burned round consumes a turn (fail-closed
                            # is not free) BEFORE continuing — otherwise the
                            # episode retries this turn forever.
                            ep.advance_turn()
                            continue
                        name, args = tc["name"], tc["arguments"]
                        if name == "finish":
                            logger.info(
                                "turn %d/%d %s e%d: finish tool (head=%r)",
                                ep.turn, ep.max_turns,
                                ep.instance.instance_id, ep.episode,
                                completion[:100],
                            )
                            ep.terminal = True
                            ep.ended_early = True
                            continue
                        if name != "bash" or not isinstance(args.get("command"), str):
                            # Malformed call: fail-closed round — the model
                            # sees its own call + the error; nothing execs.
                            ep.n_cmds_failed += 1
                            logger.info(
                                "turn %d/%d %s e%d: MALFORMED call "
                                "(name=%r, args keys=%r)",
                                ep.turn, ep.max_turns,
                                ep.instance.instance_id, ep.episode,
                                name, list(args.keys()),
                            )
                            ep.append_tool_round(
                                f"call-{ep.turn}", name, json.dumps(args),
                                "[tool error: expected bash(command=str)]",
                            )
                        else:
                            action = args["command"]
                            guard = self._repeat_guard(ep, action)
                            if guard == "terminate":
                                logger.info(
                                    "episode %s e%d: repetition guard terminated "
                                    "after %d identical sends",
                                    ep.instance.instance_id, ep.episode,
                                    ep.n_identical,
                                )
                                ep.terminal = True
                                ep.ended_early = True
                                continue
                            if guard == "nudge":
                                logger.info(
                                    "turn %d/%d %s e%d: REPETITION NUDGE (identical x%d)",
                                    ep.turn, ep.max_turns, ep.instance.instance_id,
                                    ep.episode, ep.n_identical,
                                )
                                ep.history.append((action, REPEAT_NUDGE_TEXT))
                                ep.append_tool_round(
                                    f"call-{ep.turn}", "bash",
                                    json.dumps({"command": action}),
                                    REPEAT_NUDGE_TEXT,
                                )
                            else:
                                rc, out = await self._runner.exec(ctr, action)
                                if rc != 0:
                                    ep.n_cmds_failed += 1
                                obs = truncate_obs(out)
                                ep.history.append((action, obs))
                                logger.info(
                                    "turn %d/%d %s e%d: %s (rc=%d)",
                                    ep.turn, ep.max_turns, ep.instance.instance_id,
                                    ep.episode, action[:100], rc,
                                )
                                ep.append_tool_round(
                                    f"call-{ep.turn}", "bash",
                                    json.dumps({"command": action}), obs,
                                )
                    else:
                        # fewshot: one command line after (Command:);
                        # past the first line is hallucinated output.
                        completion = completion.splitlines()[0] if completion.splitlines() else ""
                        action = extract_action(completion)
                        if action is None:
                            ep.terminal = True
                            ep.ended_early = True
                            continue
                        guard = self._repeat_guard(ep, action)
                        if guard == "terminate":
                            logger.info(
                                "episode %s e%d: repetition guard terminated "
                                "after %d identical sends",
                                ep.instance.instance_id, ep.episode,
                                ep.n_identical,
                            )
                            ep.terminal = True
                            ep.ended_early = True
                            continue
                        if guard == "nudge":
                            logger.info(
                                "turn %d/%d %s e%d: REPETITION NUDGE (identical x%d)",
                                ep.turn, ep.max_turns, ep.instance.instance_id,
                                ep.episode, ep.n_identical,
                            )
                            ep.history.append((action, REPEAT_NUDGE_TEXT))
                        else:
                            rc, out = await self._runner.exec(ctr, action)
                            if rc != 0:
                                ep.n_cmds_failed += 1
                            obs = truncate_obs(out)
                            ep.history.append((action, obs))
                            logger.info(
                                "turn %d/%d %s e%d: %s (rc=%d)",
                                ep.turn, ep.max_turns, ep.instance.instance_id,
                                ep.episode, action[:100], rc,
                            )

                    ep.advance_turn()

                # Terminal verification -> reward -> report.
                ep.reward = await self._verify(ctr, ep.instance)
                self.instance_stats.setdefault(ep.instance.instance_id, []).append(ep.reward)
                await self._report(ep)
                logger.info(
                    "episode %s e%d done: turns=%d reward=%.1f%s",
                    ep.instance.instance_id, ep.episode,
                    len(ep.rollout_ids), ep.reward,
                    " (early END)" if ep.ended_early else "",
                )
            finally:
                await self._runner.remove(ctr)

    async def _generate(
        self, req: ExternalGenerateRequest, *, first: bool
    ) -> dict[str, Any] | None:
        """generate_external with bringup tolerance: the first turn of an
        episode retries through cluster bringup (engine init ~40s); later
        turns fail fast (a live-cluster failure is a real failure)."""
        assert self._node is not None
        attempts = 8 if first else 1
        for attempt in range(attempts):
            try:
                return await self._node.call(
                    self.proxy_node, "generate_external", _to_dict(req)
                )
            except (RpcError, Exception) as exc:  # noqa: BLE001 - bringup races
                if attempt + 1 >= attempts:
                    raise
                logger.warning(
                    "generate_external failed (attempt %d/%d, bringup?): %s",
                    attempt + 1, attempts, exc,
                )
                await asyncio.sleep(15.0)
        return None  # unreachable

    async def _verify(self, ctr: str, instance: SweInstance) -> float:
        rc, out = await self._runner.exec(
            ctr, build_verify_script(instance), timeout=VERIFY_TIMEOUT_S + 120.0
        )
        marker = out.strip().splitlines()[-1] if out.strip() else "<no output>"
        logger.info(
            "verify %s: rc=%d marker=%s", instance.instance_id, rc, marker[:80]
        )
        if rc in VERIFY_EXIT_REWARD:
            if rc == 0 and "RESOLVED" not in out:
                logger.warning("verify rc=0 without RESOLVED marker: %s", out[-200:])
                return 0.0
            return VERIFY_EXIT_REWARD[rc]
        logger.warning("verify unexpected rc=%d: %s", rc, out[-300:])
        return 0.0

    async def _report(self, ep: SweEpisode) -> None:
        assert self._node is not None
        if not ep.rollout_ids:
            return
        reply = await self._node.call(
            self.proxy_node,
            "report_reward",
            _to_dict(
                ReportRewardRequest(
                    rollout_ids=ep.rollout_ids,
                    rewards=[ep.reward] * len(ep.rollout_ids),
                )
            ),
        )
        if not (isinstance(reply, dict) and reply.get("ok")):
            logger.error("report_reward not ok for %s: %r", ep.instance.instance_id, reply)
        elif reply.get("missing"):
            # Fail-closed visibility: TTL/popped ids must be loud.
            logger.error(
                "report_reward: %d ids missing for %s (TTL-expired/popped): %s",
                len(reply["missing"]), ep.instance.instance_id, reply["missing"][:8],
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--node-id", default="sim-swe-agent")
    p.add_argument("--proxy-node", default="rollout")
    p.add_argument("--repo", default="getmoto/moto")
    p.add_argument("--task-family", default="swe-gym", choices=("swe-gym", "toy"))
    p.add_argument(
        "--prompt-style", default=None, choices=(STYLE_CHAT, STYLE_FEWSHOT),
        help="chat = native template + tool calls (instruct models); "
             "fewshot = contiguous examples (base models). Default: by family.",
    )
    p.add_argument("--toy-image", default=TOY_IMAGE)
    p.add_argument("--n-instances", type=int, default=8)
    p.add_argument("--episodes-per-instance", type=int, default=2)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--exec-timeout", type=float, default=EXEC_TIMEOUT_S)
    p.add_argument("--run-id", default="swe-agent")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def _main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    agent = SimSweAgent(
        node_id=args.node_id,
        proxy_node=args.proxy_node,
        repo=args.repo,
        n_instances=args.n_instances,
        episodes_per_instance=args.episodes_per_instance,
        max_turns=args.max_turns,
        concurrency=args.concurrency,
        exec_timeout_s=args.exec_timeout,
        run_id=args.run_id,
        prompt_style=args.prompt_style,
        task_family=args.task_family,
        toy_image=args.toy_image,
    )
    await agent.start()
    try:
        await agent.run()
    finally:
        await agent.stop()


def main() -> None:
    asyncio.run(_main(_parse_args()))


if __name__ == "__main__":
    main()
