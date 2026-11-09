"""OS tool-use agent — procedurally generated file-system tasks (EXPERIMENTAL).

A small tool-calling task for the agentic mode (docs/agentic_multiturn.md):
the model answers questions about a computer ("which directory holds
report_7.csv?", "what time is it?", "what code is in notes/key.txt?") and can
only find out by calling tools. Each task comes with its own small,
procedurally generated file tree. Two backends (``--backend``):

    podman   one container (``--image``, default python:3.12-slim) shared by
             all tasks: their trees are bind-mounted read-only at /tasks, and
             each tool call runs as a real command (ls, find, cat, date) with
             the task's directory as working directory and the task's own time
             zone; paths map so every task sees its tree at /home/user. No
             network, no capabilities, read-only root, pid and memory limits.
    memory   the tree and a fixed virtual clock held in Python — no
             containers, deterministic (tests, machines without podman).

Tools (OpenAI-style schemas, rendered by the model's chat template)::

    list_dir(path)    entries of a directory ("name/" for subdirectories)
    find_file(name)   absolute paths of the files with exactly this name
    read_file(path)   a file's contents
    get_time()        the current date, time and weekday

Task kinds: ``time``, ``date``, ``weekday`` (one ``get_time`` call),
``find_dir`` (``find_file``), ``read_file`` (``read_file``), ``find_read``
(``find_file`` then ``read_file``) and ``count_files`` (``list_dir``). An
episode ends when the model replies without a tool call — that reply is the
answer — or when the turn budget runs out (the last tool result says so).
Relative paths resolve against ``/home/user``. With podman the clock is real:
time/date/weekday answers are graded against the task's local time during the
episode.

Reward, graded on the first line of the answer: 1.0 when it is exactly the
answer; ``--prefix-reward`` (0.5) when its first word is the answer but the
line runs on; otherwise ``--evidence-reward`` (default 0) when a tool result
contained the needed information, else 0. ``os/success`` counts exact answers
only. Pretrained base models (not instruction-tuned) rarely produce the chat
template's ``<tool_call>`` special tokens, stop cleanly, or answer at all at
first, so the chat opens with two solved example exchanges, a bare JSON call
at the start of a reply counts as a call (``os/wrapped_call_frac`` tracks the
proper format), and the partial rewards give RL a gradient from the start.
``--constrain-calls`` instead enforces the format with constrained decoding
(an EBNF grammar from the tool schemas, see :func:`tool_call_grammar`): every
turn is then exactly one ``<tool_call>`` in the template's format, naming a
real tool with its parameters, and the final answer is a call of the extra
``answer(text)`` tool.

Every turn is one ``generate_external`` call carrying the chat so far as
``conversations`` plus the tool schemas, tagged in ``metadata["ww_tag"]``; the
engine's tool-call parser (``tool_call_parser``) returns structured calls,
with a text parser as fallback. Training tasks come from a fixed pool of
``--n-tasks`` (default 128) generated from ``--seed``; every
``--eval-every`` waves the agent also plays ``--n-eval-tasks`` held-out tasks
(a different seed) through the proxy's internal ``generate`` endpoint, which
never enters the training buffer, and logs ``os_eval/*``.

Usage (with the agentic cluster already up)::

    python3 -m whirlwind.training.sim_os_agent --backend podman --group-size 8 --tasks-per-wave 32
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import posixpath
import random
import re
import signal
import shutil
import string
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from whirlwind.messages import (
    ExternalGenerateRequest,
    ExternalGenerateResult,
    PromptBatch,
    ReportRewardRequest,
    RolloutResult,
    TouchRequest,
    _to_dict,
)
from whirlwind.training.sim_swe_agent import parse_tool_call
from whirlwind.whirlcoms import Node, RpcError

logger = logging.getLogger(__name__)

HOME = "/home/user"
TASK_KINDS = ("time", "date", "weekday", "find_dir", "read_file", "find_read", "count_files")
CLOCK_KINDS = ("time", "date", "weekday")
_CLOCK_FORMATS = {"time": "%H:%M", "date": "%Y-%m-%d", "weekday": "%A"}
TIME_FORMAT = "%Y-%m-%d %H:%M:%S (%A)"  # get_time's output
MAX_OBS_CHARS = 4000  # tool output is truncated beyond this
BACKENDS = ("podman", "memory")
DEFAULT_IMAGE = "docker.io/library/python:3.12-slim"
CONTAINER_LABEL = "whirlwind.sim_os_agent"
MAX_CALLS_PER_TURN = 4
MAX_ANSWER_CHARS = 120  # longer replies never score (no "list everything" answers)
LAST_TURN_HINT = "[This is your last turn: reply with the answer now.]"

SYSTEM_PROMPT = (
    "You are an assistant with access to a computer through tools. The user's "
    "home directory is /home/user. Use the tools to look up what you need, one "
    "step at a time. When you know the answer, reply with the answer only."
)
# Under --constrain-calls every turn is a tool call; the answer is a call too.
ANSWER_SYSTEM_PROMPT = (
    "You are an assistant with access to a computer through tools. The user's "
    "home directory is /home/user. Use the tools to look up what you need, one "
    "step at a time. When you know the answer, call the answer tool with the "
    "answer only."
)


def _tool(name: str, description: str, params: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


def _final(answer: str, answer_tool: bool, call_id: str) -> dict[str, Any]:
    if not answer_tool:
        return {"role": "assistant", "content": answer}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "answer", "arguments": {"text": answer}}}
        ],
    }


def demo_messages(answer_tool: bool = False) -> list[dict[str, Any]]:
    """System prompt plus two solved example exchanges (in-context demonstrations).

    A base model has seen the tool-call format but rarely produces it
    unprompted; two worked exchanges (one call then the answer; find a file,
    read it, then the answer) give it a non-zero starting point for RL. The
    examples' clock and files are not any task's. With ``answer_tool`` the
    answers are ``answer`` tool calls (the constrained-decoding protocol).
    """
    path = "/home/user/work/plan_1.txt"
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT if answer_tool else SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "What time is it right now? Reply with the time in 24-hour "
            "HH:MM format and nothing else.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "demo_0", "type": "function", "function": {"name": "get_time", "arguments": {}}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_0",
            "name": "get_time",
            "content": "2024-05-02 09:12:44 (Thursday)",
        },
        _final("09:12", answer_tool, "demo_a0"),
        {
            "role": "user",
            "content": "There is a file named plan_1.txt somewhere under /home/user. "
            "What access code does it contain? Reply with the code only.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "demo_1",
                    "type": "function",
                    "function": {"name": "find_file", "arguments": {"name": "plan_1.txt"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "demo_1", "name": "find_file", "content": path},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "demo_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": path}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "demo_2",
            "name": "read_file",
            "content": "meeting notes\naccess code: Q4ZT7B\n",
        },
        _final("Q4ZT7B", answer_tool, "demo_a1"),
    ]


TOOLS: list[dict[str, Any]] = [
    _tool(
        "list_dir",
        "List the files and subdirectories of a directory (subdirectories end with '/').",
        {"path": "Absolute path of the directory, e.g. /home/user/docs."},
    ),
    _tool(
        "find_file",
        "Search the home directory for files with exactly this name; returns their paths.",
        {"name": "The file name, e.g. notes_3.txt."},
    ),
    _tool(
        "read_file",
        "Return the contents of a file.",
        {"path": "Absolute path of the file."},
    ),
    _tool("get_time", "Return the current date, time and day of the week.", {}),
]

def _ebnf_literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


ANSWER_TOOL: dict[str, Any] = _tool(
    "answer",
    "Give the final answer to the user's question; this ends the task.",
    {"text": "The answer only, e.g. 09:27 or /home/user/docs."},
)


def tool_call_grammar(tools: list[dict[str, Any]]) -> str:
    """EBNF for one assistant turn under ``--constrain-calls``: exactly one tool call.

    The call is the chat template's hermes format, byte for byte as the
    template renders it (``<tool_call>\\n{"name": ..., "arguments": {...}}
    \\n</tool_call>``), naming one of ``tools`` with exactly its parameters
    (single-line JSON strings; their length is capped by the engine's
    ``max_gen_len`` — a bounded repeat like ``{0,200}`` makes the grammar
    matcher's per-token masks several hundred times slower). Every turn being a call
    (the final answer through the ``answer`` tool) is what makes the format
    enforceable: a base model offered a free-text alternative routes its call
    intent into the text instead. Used as ``sampling_constraint={"ebnf": ...}``
    it also ends the turn right after ``</tool_call>``.
    """
    calls = []
    for tool in tools:
        fn = tool["function"]
        parts = [_ebnf_literal('<tool_call>\n{"name": "' + fn["name"] + '", "arguments": {')]
        for i, param in enumerate(fn["parameters"].get("properties", {})):
            parts.append(_ebnf_literal(("" if i == 0 else ", ") + '"' + param + '": '))
            parts.append("str")
        parts.append(_ebnf_literal("}}\n</tool_call>"))
        calls.append(" ".join(parts))
    return "\n".join(
        [
            "root ::= " + " | ".join(calls),
            'str ::= "\\"" ([^"\\\\\\n] | "\\\\" ["\\\\/bfnrt])* "\\""',
        ]
    )


_TOP_DIRS = (
    "docs", "projects", "notes", "logs", "music", "photos", "backup", "src",
    "data", "reports", "archive", "downloads",
)
_SUB_DIRS = ("alpha", "beta", "gamma", "old", "drafts", "misc", "work", "personal", "q3", "2025")
_WORDS = (
    "apple", "river", "stone", "maple", "comet", "delta", "ember", "falcon",
    "garnet", "harbor", "iris", "jade", "kepler", "lotus", "meadow", "nova",
    "onyx", "pepper", "quartz", "raven", "sable", "tundra", "umber", "violet",
    "willow", "yarrow", "zephyr", "budget", "invoice", "report", "summary",
)
_EXTS = ("txt", "csv", "log", "md", "json")


# ---------------------------------------------------------------------------
# The simulated computer
# ---------------------------------------------------------------------------


def call_error(name: str, args: dict[str, Any]) -> str | None:
    """Why a tool call is malformed (unknown tool, missing argument), else None."""
    known = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}
    if name not in known:
        return f"error: unknown tool {name!r}; available: {', '.join(sorted(known))}"
    if any(k not in args for k in known[name]):
        return f"error: missing or invalid arguments for {name}: {args!r}"
    return None


@dataclass
class VirtualFS:
    """A task's directory tree and clock: the in-memory tool backend, and the
    spec the podman backend materializes."""

    files: dict[str, str]  # absolute path -> contents
    dirs: set[str]  # absolute directory paths (HOME and its ancestors included)
    now: datetime  # the memory backend's fixed clock
    tz_offset_h: int = 0  # the podman backend's time zone (hours east of UTC)

    def resolve(self, path: Any) -> str:
        p = str(path or "").strip().strip("'\"`")
        if p in ("", "~"):
            return HOME
        if p.startswith("~/"):
            p = HOME + p[1:]
        elif not p.startswith("/"):
            p = posixpath.join(HOME, p)
        return posixpath.normpath(p)

    def list_dir(self, path: Any) -> str:
        p = self.resolve(path)
        if p in self.files:
            return f"error: not a directory: {p}"
        if p not in self.dirs:
            return f"error: no such directory: {p}"
        entries = sorted(
            [posixpath.basename(d) + "/" for d in self.dirs if posixpath.dirname(d) == p and d != p]
            + [posixpath.basename(f) for f in self.files if posixpath.dirname(f) == p]
        )
        return "\n".join(entries) if entries else "(empty directory)"

    def find_file(self, name: Any) -> str:
        n = posixpath.basename(str(name or "").strip().strip("'\"`"))
        hits = sorted(f for f in self.files if posixpath.basename(f) == n)
        return "\n".join(hits) if hits else f"no files named {n!r}"

    def read_file(self, path: Any) -> str:
        p = self.resolve(path)
        if p in self.files:
            return self.files[p]
        if p in self.dirs:
            return f"error: is a directory: {p}"
        return f"error: no such file: {p}"

    def get_time(self) -> str:
        return self.now.strftime(TIME_FORMAT)

    def call(self, name: str, arguments: Any) -> tuple[str, bool]:
        """Run one tool call; returns ``(observation, well_formed)``."""
        args = arguments if isinstance(arguments, dict) else {}
        err = call_error(name, args)
        if err is not None:
            return err, False
        if name == "get_time":
            return self.get_time(), True
        if name == "list_dir":
            return self.list_dir(args["path"]), True
        if name == "find_file":
            return self.find_file(args["name"]), True
        return self.read_file(args["path"]), True


def posix_tz(offset_h: int) -> str:
    """A POSIX TZ string for a fixed UTC offset (no tzdata needed in the image)."""
    return f"SIM{-offset_h:+d}"


def clock_answer(kind: str, when: datetime) -> str:
    return when.strftime(_CLOCK_FORMATS[kind])


# ---------------------------------------------------------------------------
# The podman backend
# ---------------------------------------------------------------------------


class PodmanSandbox:
    """One podman container shared by all tasks; the tools run as real commands in it.

    Every task's tree is written to a host directory, one subdirectory per
    task, bind-mounted read-only at ``/tasks``. A tool call runs as ``podman
    exec -w /tasks/<task>/home/user -e TZ=<the task's zone>`` with an argv
    list (never a shell). Paths are mapped from the model's view
    (``/home/user/...`` or any other absolute path, normalized first) into the
    task's root and back in the output, so each task looks like its own
    machine and cannot reach another task's files. The container has no
    network, no capabilities, a read-only root file system and pid and memory
    limits; it carries the label ``whirlwind.sim_os_agent=<label>``, and
    :meth:`remove_all` removes it (also a stale one from a crashed run with
    the same label) together with the host directory.
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        label: str = "sim-os",
        concurrency: int = 32,
        exec_timeout_s: float = 10.0,
        podman: str = "podman",
    ) -> None:
        self.image = image
        self.label = re.sub(r"[^A-Za-z0-9_.-]", "-", label) or "sim-os"
        self.name = f"ww-os-{self.label}"
        self.concurrency = max(1, concurrency)
        self.exec_timeout_s = exec_timeout_s
        self.podman = podman
        self._roots: dict[str, str] = {}  # task_id -> the task's root inside the container
        self._host_dir: str | None = None
        self._sem: asyncio.Semaphore | None = None

    async def _run(
        self, *args: str, stdin: bytes | None = None, timeout: float = 60.0
    ) -> tuple[int, str, str]:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                self.podman,
                *args,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return 124, "", f"timed out after {timeout:.0f}s"
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")

    async def start(self, tasks: list[OsTask]) -> None:
        """Write every task's tree and start the shared container."""
        await self.remove_all()  # leftovers of a crashed run with this label
        self._host_dir = tempfile.mkdtemp(prefix="whirlwind-os-tasks-")
        os.chmod(self._host_dir, 0o755)
        for i, task in enumerate(tasks):
            sub = f"t{i}"
            for d in task.fs.dirs:
                os.makedirs(os.path.join(self._host_dir, sub, d.lstrip("/")), mode=0o755, exist_ok=True)
            for path, content in task.fs.files.items():
                with open(os.path.join(self._host_dir, sub, path.lstrip("/")), "w", encoding="utf-8") as f:
                    f.write(content)
            self._roots[task.task_id] = f"/tasks/{sub}"
        rc, _, err = await self._run(
            "run", "-d", "--rm", "--name", self.name,
            "--label", f"{CONTAINER_LABEL}={self.label}",
            "--network", "none", "--cap-drop", "all",
            "--security-opt", "no-new-privileges", "--read-only",
            "--pids-limit", "512", "--memory", "512m",
            "-v", f"{self._host_dir}:/tasks:ro",
            self.image, "sleep", "infinity",
            timeout=300.0,
        )  # fmt: skip
        if rc != 0:
            await self.remove_all()
            raise RuntimeError(f"podman run failed: {err.strip()}")

    def _inside(self, task: OsTask, path: Any) -> str:
        """A path as the model gave it, mapped into the task's root."""
        return posixpath.normpath(self._roots[task.task_id] + task.fs.resolve(path))

    def _outside(self, task: OsTask, text: str) -> str:
        """Command output with the task's root mapped back to ``/``."""
        root = self._roots[task.task_id]
        return re.sub(re.escape(root) + r"(?=/|\b|$)", "", text)

    async def call(self, task: OsTask, name: str, arguments: Any) -> tuple[str, bool]:
        """Run one tool call for ``task``; ``(observation, well_formed)``."""
        args = arguments if isinstance(arguments, dict) else {}
        err = call_error(name, args)
        if err is not None:
            return err, False
        home = self._roots[task.task_id] + HOME
        if name == "get_time":
            argv = ["date", "+" + TIME_FORMAT]
        elif name == "list_dir":
            argv = ["ls", "-1p", "--", self._inside(task, args["path"])]
        elif name == "find_file":
            base = posixpath.basename(str(args["name"]).strip().strip("'\"`"))
            argv = ["find", home, "-type", "f", "-name", base]
        else:
            argv = ["cat", "--", self._inside(task, args["path"])]
        rc, out, stderr = await self._run(
            "exec", "-w", home, "-e", f"TZ={posix_tz(task.fs.tz_offset_h)}", self.name, *argv,
            timeout=self.exec_timeout_s,
        )
        out = self._outside(task, out)
        stderr = self._outside(task, stderr).replace("''", "'/'")  # the root itself
        if rc != 0 and not out:
            return f"error: {stderr.strip() or f'exit status {rc}'}"[:MAX_OBS_CHARS], True
        if name == "list_dir":
            entries = sorted(ln for ln in out.splitlines() if ln)
            return ("\n".join(entries) if entries else "(empty directory)"), True
        if name == "find_file":
            hits = sorted(ln for ln in out.splitlines() if ln)
            return ("\n".join(hits) if hits else f"no files named {base!r}"), True
        if name == "get_time":
            return out.strip(), True
        return out[:MAX_OBS_CHARS], True

    async def remove_all(self) -> None:
        rc, out, _ = await self._run(
            "ps", "-aq", "--filter", f"label={CONTAINER_LABEL}={self.label}"
        )
        ids = out.split() if rc == 0 else []
        if ids:
            await self._run("rm", "-f", "-t", "0", *ids, timeout=300.0)
        if self._host_dir is not None:
            shutil.rmtree(self._host_dir, ignore_errors=True)
            self._host_dir = None
        self._roots.clear()


@dataclass
class OsTask:
    task_id: str
    kind: str
    question: str
    answer: str
    fs: VirtualFS
    # Appears in a tool observation once the information needed for the
    # answer has been retrieved (drives the optional evidence reward).
    evidence: str = ""


def _code(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def _make_fs(rng: random.Random) -> VirtualFS:
    dirs = {"/", "/home", HOME}
    for top in rng.sample(_TOP_DIRS, rng.randint(3, 5)):
        d = f"{HOME}/{top}"
        dirs.add(d)
        for sub in rng.sample(_SUB_DIRS, rng.randint(0, 2)):
            dirs.add(f"{d}/{sub}")
    leaf_dirs = sorted(d for d in dirs if d.startswith(HOME))
    files: dict[str, str] = {}
    names: set[str] = set()
    n_files = rng.randint(6, 12)
    while len(files) < n_files:
        name = f"{rng.choice(_WORDS)}_{rng.randint(1, 99)}.{rng.choice(_EXTS)}"
        if name in names:
            continue
        names.add(name)
        word = rng.choice(_WORDS)
        files[f"{rng.choice(leaf_dirs)}/{name}"] = f"{word} {rng.choice(_WORDS)} {rng.randint(1, 999)}\n"
    now = datetime(2025, 1, 1) + timedelta(seconds=rng.randint(0, 3 * 365 * 24 * 3600))
    return VirtualFS(files=files, dirs=dirs, now=now)


def make_task(rng: random.Random, task_id: str, kind: str | None = None) -> OsTask:
    """One procedurally generated task (its own file tree and clock)."""
    kind = kind or rng.choice(TASK_KINDS)
    fs = _make_fs(rng)
    path = rng.choice(sorted(fs.files))
    name = posixpath.basename(path)
    if kind == "time":
        q = "What time is it right now? Reply with the time in 24-hour HH:MM format and nothing else."
        a = clock_answer("time", fs.now)
    elif kind == "date":
        q = "What is today's date? Reply in YYYY-MM-DD format and nothing else."
        a = clock_answer("date", fs.now)
    elif kind == "weekday":
        q = "Which day of the week is it today? Reply with the name of the day only."
        a = clock_answer("weekday", fs.now)
    elif kind == "find_dir":
        q = (
            f"Which directory contains the file {name}? Reply with the absolute "
            "path of the directory and nothing else."
        )
        a = posixpath.dirname(path)
    elif kind in ("read_file", "find_read"):
        code = _code(rng)
        fs.files[path] = fs.files[path] + f"access code: {code}\n"
        if kind == "read_file":
            q = f"What is the access code written in {path}? Reply with the code only."
        else:
            q = (
                f"There is a file named {name} somewhere under /home/user. What "
                "access code does it contain? Reply with the code only."
            )
        a = code
    elif kind == "count_files":
        with_files = sorted({posixpath.dirname(f) for f in fs.files})
        d = rng.choice(with_files)
        q = (
            f"How many files (not counting subdirectories) are directly inside {d}? "
            "Reply with a number only."
        )
        a = str(sum(1 for f in fs.files if posixpath.dirname(f) == d))
    else:
        raise ValueError(f"unknown task kind {kind!r}")
    if kind in ("time", "date", "weekday"):
        evidence = fs.get_time()
    elif kind == "find_dir":
        evidence = path
    elif kind in ("read_file", "find_read"):
        evidence = f"access code: {a}"
    else:  # count_files: the directory's own listing
        evidence = fs.list_dir(d)
    # The podman backend's time zone, from the task id (the RNG stream above,
    # and therefore every task, stays the same as without it).
    fs.tz_offset_h = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 23 - 10
    return OsTask(task_id=task_id, kind=kind, question=q, answer=a, fs=fs, evidence=evidence)


def make_tasks(n: int, seed: int, kinds: tuple[str, ...] = TASK_KINDS) -> list[OsTask]:
    """``n`` tasks, the kinds cycled evenly, reproducible from ``seed``."""
    rng = random.Random(seed)
    return [make_task(rng, f"s{seed}-{i}", kinds[i % len(kinds)]) for i in range(n)]


# ---------------------------------------------------------------------------
# Replies: tool calls and answers
# ---------------------------------------------------------------------------

_SPECIAL_RE = re.compile(r"<\|[A-Za-z_]+\|>")
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def clean_reply(text: str) -> str:
    """The visible answer text of a completion (no think block or special tokens)."""
    return _SPECIAL_RE.sub("", _THINK_RE.sub("", text)).strip()


_BARE_CALL_WINDOW = 32  # a bare JSON call must start within this many chars


def _bare_json_call(text: str) -> dict[str, Any] | None:
    """A tool call written as a bare JSON object at the start of the reply."""
    stripped = text.lstrip()
    i = stripped.find("{")
    if i < 0 or i > _BARE_CALL_WINDOW:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(stripped[i:])
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}


def reply_calls(
    completion: str, structured: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], bool]:
    """Tool calls of one completion and whether they used the template's format.

    Order: the engine's parsed calls, the text parser (``<tool_call>``-wrapped
    hermes JSON or XML), then a bare JSON call at the start of the reply. Base
    models rarely produce the wrapper's special tokens before training, so the
    environment accepts the bare form too; ``os/wrapped_call_frac`` tracks
    how many calls use the proper format.
    """
    if structured:
        return list(structured)[:MAX_CALLS_PER_TURN], True
    call = parse_tool_call(completion)
    if call is not None:
        return [call], True
    bare = _bare_json_call(clean_reply(completion))
    return ([bare], False) if bare is not None else ([], False)


def _norm(s: str) -> str:
    s = s.strip().strip("`'\"").strip().rstrip(".,;:!").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip("/") if len(s) > 1 else s


def grade(reply: str, expected: str) -> str:
    """How well a reply states the expected answer: "exact", "prefix" or "none".

    Only the reply's first non-empty line counts. "exact" (the line must be
    at most ``MAX_ANSWER_CHARS`` long): the line — or the text after
    "answer:"/"answer is", or a backticked span in it — equals the answer
    after normalization (case, quotes, a trailing period or slash). "prefix":
    the line's first word is the answer but the line runs on (a base model
    that has not yet learned to stop: "14:49 sure").
    """
    lines = [ln for ln in clean_reply(reply).splitlines() if ln.strip()]
    if not lines:
        return "none"
    line = lines[0].strip()
    want = _norm(expected)
    if len(line) <= MAX_ANSWER_CHARS:
        candidates = [line]
        m = re.search(r"answer(?: is)?\s*[:=]?\s*(.+)$", line, re.IGNORECASE)
        if m:
            candidates.append(m.group(1))
        candidates += re.findall(r"`([^`]+)`", line)
        if any(_norm(c) == want for c in candidates):
            return "exact"
    # Every answer is a single token, so "prefix" compares the first word.
    if _norm(line.split()[0]) == want:
        return "prefix"
    return "none"


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


@dataclass
class OsEpisode:
    """One attempt at a task: the chat so far plus bookkeeping."""

    task: OsTask
    group_id: int
    group_size: int
    episode: int
    max_turns: int
    fewshot: bool = True
    answer_tool: bool = False  # the final answer is an ``answer`` tool call
    prefix_reward: float = 0.5
    evidence_reward: float = 0.0
    live_clock: bool = False  # podman: clock answers come from the real clock
    started_at: float = field(default_factory=time.time)
    turn: int = 1
    terminal: bool = False
    correct: bool = False
    partial: bool = False
    answered: bool = False
    found: bool = False
    reward: float = 0.0
    n_calls: int = 0
    n_valid_calls: int = 0
    n_wrapped_calls: int = 0
    final_reply: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)  # tool calls, for logging
    rollout_ids: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            system = ANSWER_SYSTEM_PROMPT if self.answer_tool else SYSTEM_PROMPT
            head = (
                demo_messages(self.answer_tool)
                if self.fewshot
                else [{"role": "system", "content": system}]
            )
            self.messages = head + [{"role": "user", "content": self.task.question}]

    def make_tag(self) -> dict[str, Any]:
        return {
            "ww_tag": {
                "task": "os-tools",
                "kind": self.task.kind,
                "problem": self.task.task_id,
                "group": self.group_id,
                "group_size": self.group_size,
                "episode": self.episode,
                "turn": self.turn,
                "max_turns": self.max_turns,
                "terminal": self.turn >= self.max_turns,
            }
        }

    def expected_answers(self) -> list[str]:
        """The accepted answers: the task's, or — for clock questions on a real
        clock — the task's local time at any minute of the episode so far."""
        if not (self.live_clock and self.task.kind in CLOCK_KINDS):
            return [self.task.answer]
        tz = timezone(timedelta(hours=self.task.fs.tz_offset_h))
        t, end, out = self.started_at - 60.0, time.time(), []
        while t <= end + 60.0:
            a = clock_answer(self.task.kind, datetime.fromtimestamp(t, tz))
            if a not in out:
                out.append(a)
            t += 60.0
        return out

    def _is_evidence(self, name: str, obs: str, ok: bool) -> bool:
        if self.task.kind in CLOCK_KINDS:
            return name == "get_time" and ok and not obs.startswith("error:")
        return bool(self.task.evidence) and self.task.evidence in obs

    def step(
        self,
        completion: str,
        calls: list[dict[str, Any]],
        wrapped: bool = True,
        results: list[tuple[str, bool]] | None = None,
    ) -> None:
        """Apply one model reply: record its tool calls and their ``results``
        (``(observation, well_formed)`` each; run on the in-memory tree when
        omitted), or grade it as the answer (the reply itself, or the text of
        an ``answer`` call under the answer-tool protocol)."""
        answer = next((c for c in calls if c.get("name") == "answer"), None)
        if answer is not None and self.answer_tool:
            name, args = _name_args(answer)
            self.n_calls += 1
            self.n_valid_calls += int("text" in args)
            self.n_wrapped_calls += int(wrapped)
            self.trace.append(
                {"turn": self.turn, "name": name, "arguments": args, "ok": "text" in args,
                 "wrapped": wrapped, "result": ""}
            )
            completion, calls = str(args.get("text", "")), []
        if not calls:
            self.answered = True
            self.final_reply = completion
            matches = [grade(completion, a) for a in self.expected_answers()]
            self.correct = "exact" in matches
            self.partial = not self.correct and "prefix" in matches
            self.terminal = True
            self._set_reward()
            return
        if results is None:
            results = [self.task.fs.call(*_name_args(c)) for c in calls]
        tool_calls: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        for i, (call, (obs, ok)) in enumerate(zip(calls, results, strict=True)):
            name, args = _name_args(call)
            self.found = self.found or self._is_evidence(name, obs, ok)
            self.n_calls += 1
            self.n_valid_calls += int(ok)
            self.n_wrapped_calls += int(wrapped)
            self.trace.append(
                {"turn": self.turn, "name": name, "arguments": args, "ok": ok,
                 "wrapped": wrapped, "result": obs[:200]}
            )
            cid = f"call_{self.turn}_{i}"
            tool_calls.append(
                {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
            )
            messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": obs})
        if self.turn + 1 == self.max_turns:
            # Budget feedback, as agent frameworks give it: the next turn is the last.
            messages[-1]["content"] = f"{messages[-1]['content']}\n{LAST_TURN_HINT}"
        self.messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        self.messages.extend(messages)
        if self.turn >= self.max_turns:
            self.terminal = True  # out of turns without an answer
            self._set_reward()
        else:
            self.turn += 1

    def _set_reward(self) -> None:
        """1.0 for an exact answer, ``prefix_reward`` for the answer followed
        by more text, else ``evidence_reward`` if the tools retrieved the
        needed information, else 0.0."""
        if self.correct:
            self.reward = 1.0
        elif self.partial:
            self.reward = self.prefix_reward
        elif self.found:
            self.reward = self.evidence_reward
        else:
            self.reward = 0.0


def _name_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    args = call.get("arguments")
    return str(call.get("name", "")), (args if isinstance(args, dict) else {})


def episode_record(ep: OsEpisode, **extra: Any) -> dict[str, Any]:
    """One episode as a JSON-able record (``--trajectory-log``)."""
    return {
        **extra,
        "task": ep.task.task_id,
        "kind": ep.task.kind,
        "question": ep.task.question,
        "answer": ep.expected_answers()[-1],
        "calls": ep.trace,
        "final_reply": clean_reply(ep.final_reply)[:300],
        "correct": ep.correct,
        "partial": ep.partial,
        "found": ep.found,
        "reward": ep.reward,
        "turns": ep.turn,
    }


def episode_stats(episodes: list[OsEpisode], prefix: str) -> dict[str, float]:
    n = max(1, len(episodes))
    calls = sum(ep.n_calls for ep in episodes)
    stats = {
        f"{prefix}/success": sum(ep.correct for ep in episodes) / n,
        f"{prefix}/success_or_prefix": sum(ep.correct or ep.partial for ep in episodes) / n,
        f"{prefix}/answered_frac": sum(ep.answered for ep in episodes) / n,
        f"{prefix}/found_frac": sum(ep.found for ep in episodes) / n,
        f"{prefix}/mean_reward": sum(ep.reward for ep in episodes) / n,
        f"{prefix}/no_tool_answer_frac": sum(
            ep.answered and ep.n_calls == (1 if ep.answer_tool else 0) for ep in episodes
        ) / n,
        f"{prefix}/calls_per_episode": calls / n,
        f"{prefix}/valid_call_frac": sum(ep.n_valid_calls for ep in episodes) / max(1, calls),
        f"{prefix}/wrapped_call_frac": sum(ep.n_wrapped_calls for ep in episodes) / max(1, calls),
        f"{prefix}/mean_turns": sum(ep.turn for ep in episodes) / n,
    }
    for kind in sorted({ep.task.kind for ep in episodes}):
        eps = [ep for ep in episodes if ep.task.kind == kind]
        stats[f"{prefix}/success/{kind}"] = sum(ep.correct for ep in eps) / len(eps)
    return stats


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class SimOsAgent:
    """Drives the agent proxy with OS tool-use episodes."""

    def __init__(
        self,
        *,
        node_id: str = "sim-os-agent",
        proxy_node: str = "rollout",
        n_tasks: int = 128,
        seed: int = 0,
        group_size: int = 8,
        tasks_per_wave: int = 32,
        max_turns: int = 4,
        fewshot: bool = True,
        prefix_reward: float = 0.5,
        evidence_reward: float = 0.0,
        waves: int = 0,
        eval_every: int = 5,
        n_eval_tasks: int = 64,
        eval_seed: int = 1_000_003,
        client_id: str = "sim-os",
        backend: str = "memory",
        image: str = DEFAULT_IMAGE,
        container_label: str = "",
        container_concurrency: int = 32,
        trajectory_log: str = "",
        constrain_calls: bool = False,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if group_size < 2:
            logger.warning(
                "group_size=%d: a one-episode group has no baseline, so every "
                "advantage is zero; use >= 2 for anything but a smoke test",
                group_size,
            )
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if eval_seed == seed:
            raise ValueError("eval_seed must differ from seed (held-out tasks)")
        self.node_id = node_id
        self.proxy_node = proxy_node
        self.tasks = make_tasks(n_tasks, seed)
        self.eval_tasks = make_tasks(n_eval_tasks, eval_seed) if n_eval_tasks > 0 else []
        self.group_size = max(1, group_size)
        self.tasks_per_wave = max(1, min(tasks_per_wave, len(self.tasks)))
        self.max_turns = max_turns
        self.episode_kwargs = {
            "fewshot": fewshot,
            "prefix_reward": prefix_reward,
            "evidence_reward": evidence_reward,
            "live_clock": backend == "podman",
        }
        self.backend = backend
        self.trajectory_log = trajectory_log
        # Constrained decoding: every turn is exactly one template-format tool
        # call; the final answer goes through the ``answer`` tool.
        self.tools = TOOLS + [ANSWER_TOOL] if constrain_calls else TOOLS
        self.constraint = {"ebnf": tool_call_grammar(self.tools)} if constrain_calls else None
        self.episode_kwargs["answer_tool"] = constrain_calls
        self.sandboxes = (
            PodmanSandbox(
                image=image,
                label=container_label or client_id,
                concurrency=container_concurrency,
            )
            if backend == "podman"
            else None
        )
        self.waves = waves  # 0 = until stopped
        self.eval_every = eval_every  # 0 = no evaluation
        self.client_id = client_id
        self._rng = random.Random(seed)
        self._order: list[int] = []
        self._eval_episodes: list[OsEpisode] = []
        self._node: Node | None = None
        self._group_counter = 0

    @property
    def is_running(self) -> bool:
        return self._node.is_running if self._node is not None else False

    async def start(self) -> None:
        if self.sandboxes is not None:
            t0 = time.monotonic()
            await self.sandboxes.start(self.tasks + self.eval_tasks)
            logger.info(
                "Started the task container %s from %s with %d task trees (label %s=%s) in %.1fs",
                self.sandboxes.name,
                self.sandboxes.image,
                len(self.tasks) + len(self.eval_tasks),
                CONTAINER_LABEL,
                self.sandboxes.label,
                time.monotonic() - t0,
            )
        self._node = Node(self.node_id, auto_process=True)
        await self._node.start()
        logger.info(
            "Sim OS agent '%s' started (proxy=%s, backend=%s, constrained calls=%s, %d tasks, "
            "%d held-out, K=%d, %d tasks/wave, max_turns=%d, rewards: prefix %.2f, evidence %.2f)",
            self.node_id,
            self.proxy_node,
            self.backend,
            self.constraint is not None,
            len(self.tasks),
            len(self.eval_tasks),
            self.group_size,
            self.tasks_per_wave,
            self.max_turns,
            self.episode_kwargs["prefix_reward"],
            self.episode_kwargs["evidence_reward"],
        )

    async def stop(self) -> None:
        # Containers first: they outlive this process otherwise.
        if self.sandboxes is not None:
            await self.sandboxes.remove_all()
        if self._node is not None:
            try:  # deregistering waits on the cluster; never hang the exit on it
                await asyncio.wait_for(self._node.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("node stop timed out; exiting anyway")

    async def _run_tools(
        self, task: OsTask, calls: list[dict[str, Any]]
    ) -> list[tuple[str, bool]] | None:
        """Tool results for one reply: in the task's container, or ``None``
        (the episode runs them on its in-memory tree)."""
        if self.sandboxes is None or not calls or any(c.get("name") == "answer" for c in calls):
            return None  # in-memory tree, or the turn ends the episode
        return list(
            await asyncio.gather(
                *(self.sandboxes.call(task, *_name_args(c)) for c in calls)
            )
        )

    def _write_trajectories(self, episodes: list[OsEpisode], **extra: Any) -> None:
        if not self.trajectory_log:
            return
        with open(self.trajectory_log, "a", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(episode_record(ep, **extra)) + "\n")

    def _log(self, metrics: dict[str, float]) -> None:
        assert self._node is not None
        self._node.call_bg_no_return(
            "tracker", "log_metrics", {"metrics": metrics, "node_id": self.node_id}
        )

    def _next_tasks(self) -> list[OsTask]:
        out: list[OsTask] = []
        while len(out) < self.tasks_per_wave:
            if not self._order:
                self._order = list(range(len(self.tasks)))
                self._rng.shuffle(self._order)
            out.append(self.tasks[self._order.pop()])
        return out

    # ------------------------------------------------------------------
    # Training episodes (generate_external; rewards reported per episode)
    # ------------------------------------------------------------------

    async def _run_episode(self, ep: OsEpisode) -> None:
        assert self._node is not None
        while not ep.terminal:
            res = await self._node.call_typed(
                self.proxy_node,
                "generate_external",
                ExternalGenerateResult,
                ExternalGenerateRequest(
                    prompts=[ep.task.question],
                    conversations=[ep.messages],
                    tools=self.tools,
                    sampling_constraint=self.constraint,
                    group_size=1,
                    client_id=self.client_id,
                    metadata=ep.make_tag(),
                ),
            )
            ep.rollout_ids.append(res.rollout_ids[0])
            if len(ep.rollout_ids) > 1:  # keep the episode's earlier turns alive
                await self._node.call(
                    self.proxy_node, "touch", _to_dict(TouchRequest(rollout_ids=ep.rollout_ids))
                )
            structured = res.tool_calls[0] if res.tool_calls else None
            completion = res.completions[0]
            calls, wrapped = reply_calls(completion, structured)
            ep.step(completion, calls, wrapped, await self._run_tools(ep.task, calls))
        reply = await self._node.call(
            self.proxy_node,
            "report_reward",
            _to_dict(
                ReportRewardRequest(
                    rollout_ids=ep.rollout_ids, rewards=[ep.reward] * len(ep.rollout_ids)
                )
            ),
        )
        if not (isinstance(reply, dict) and reply.get("ok")):
            logger.warning("report_reward not ok for %s: %r", ep.task.task_id, reply)

    async def _train_wave(self) -> list[OsEpisode]:
        episodes: list[OsEpisode] = []
        for task in self._next_tasks():
            self._group_counter += 1
            for e in range(self.group_size):
                episodes.append(
                    OsEpisode(
                        task=task,
                        group_id=self._group_counter,
                        group_size=self.group_size,
                        episode=e,
                        max_turns=self.max_turns,
                        **self.episode_kwargs,
                    )
                )
        results = await asyncio.gather(
            *(self._run_episode(ep) for ep in episodes), return_exceptions=True
        )
        failed = [r for r in results if isinstance(r, Exception)]
        if failed:
            # Their turns never get a reward; the groups time out in the loop.
            logger.error("%d/%d episodes failed: %s", len(failed), len(episodes), failed[0])
        return [ep for ep, r in zip(episodes, results, strict=True) if not isinstance(r, Exception)]

    # ------------------------------------------------------------------
    # Held-out evaluation (internal generate: never enters the buffer)
    # ------------------------------------------------------------------

    async def evaluate(self) -> dict[str, float]:
        assert self._node is not None
        pending = [
            OsEpisode(
                task=t,
                group_id=0,
                group_size=1,
                episode=0,
                max_turns=self.max_turns,
                **self.episode_kwargs,
            )
            for t in self.eval_tasks
        ]
        done = list(pending)
        while pending:
            result = await self._node.call_typed(
                self.proxy_node,
                "generate",
                RolloutResult,
                PromptBatch(
                    prompts=[ep.task.question for ep in pending],
                    conversations=[ep.messages for ep in pending],
                    tools=self.tools,
                    sampling_constraint=self.constraint,
                    log_prefix="val/os_tools",
                ),
            )
            tool_calls = result.tool_calls or [[] for _ in pending]
            parsed = [
                reply_calls(completion, structured)
                for completion, structured in zip(result.completions, tool_calls, strict=True)
            ]
            outputs = await asyncio.gather(
                *(self._run_tools(ep.task, calls) for ep, (calls, _) in zip(pending, parsed))
            )
            for ep, completion, (calls, wrapped), out in zip(
                pending, result.completions, parsed, outputs, strict=True
            ):
                ep.step(completion, calls, wrapped, out)
            pending = [ep for ep in pending if not ep.terminal]
        self._eval_episodes = done
        return episode_stats(done, "os_eval")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        assert self._node is not None
        wave = 0
        while self.is_running and (self.waves <= 0 or wave < self.waves):
            if self.eval_tasks and self.eval_every > 0 and wave % self.eval_every == 0:
                try:
                    t0 = time.monotonic()
                    stats = await self.evaluate()
                    self._write_trajectories(self._eval_episodes, split="eval", wave=wave)
                    stats["os_eval/wave"] = float(wave)
                    self._log(stats)
                    logger.info(
                        "eval @ wave %d: success=%.3f (or prefix %.3f) found=%.3f "
                        "valid_calls=%.3f calls/ep=%.2f (%.1fs)",
                        wave,
                        stats["os_eval/success"],
                        stats["os_eval/success_or_prefix"],
                        stats["os_eval/found_frac"],
                        stats["os_eval/valid_call_frac"],
                        stats["os_eval/calls_per_episode"],
                        time.monotonic() - t0,
                    )
                except RpcError as exc:
                    logger.error("evaluation at wave %d failed: %s", wave, exc)
            t0 = time.monotonic()
            try:
                episodes = await self._train_wave()
            except RpcError as exc:
                logger.error("wave %d failed: %s", wave, exc)
                wave += 1
                continue
            wave += 1
            if not episodes:
                continue
            stats = episode_stats(episodes, "os")
            self._write_trajectories(episodes, split="train", wave=wave)
            stats["os/waves"] = float(wave)
            stats["os/wave_time_s"] = time.monotonic() - t0
            self._log(stats)
            logger.info(
                "wave %d: %d episodes, success=%.3f (or prefix %.3f) found=%.3f "
                "reward=%.3f valid_calls=%.3f wrapped_calls=%.3f calls/ep=%.2f "
                "no-tool answers=%.2f (%.1fs)",
                wave,
                len(episodes),
                stats["os/success"],
                stats["os/success_or_prefix"],
                stats["os/found_frac"],
                stats["os/mean_reward"],
                stats["os/valid_call_frac"],
                stats["os/wrapped_call_frac"],
                stats["os/calls_per_episode"],
                stats["os/no_tool_answer_frac"],
                stats["os/wave_time_s"],
            )
        logger.info("Sim OS agent done: %d waves", wave)


async def _main(args: argparse.Namespace) -> None:
    agent = SimOsAgent(
        node_id=args.node_id,
        proxy_node=args.proxy,
        n_tasks=args.n_tasks,
        seed=args.seed,
        group_size=args.group_size,
        tasks_per_wave=args.tasks_per_wave,
        max_turns=args.max_turns,
        fewshot=not args.no_fewshot,
        prefix_reward=args.prefix_reward,
        evidence_reward=args.evidence_reward,
        waves=args.waves,
        eval_every=args.eval_every,
        n_eval_tasks=args.n_eval_tasks,
        eval_seed=args.eval_seed,
        client_id=args.client_id,
        backend=args.backend,
        image=args.image,
        container_label=args.container_label or os.environ.get("WW_RUN_ID", ""),
        container_concurrency=args.container_concurrency,
        trajectory_log=args.trajectory_log,
        constrain_calls=args.constrain_calls,
    )
    # SIGTERM/SIGINT cancel the run, so the finally below removes the containers.
    main_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, main_task.cancel)  # type: ignore[union-attr]
    try:
        await agent.start()
        await agent.run()
    except asyncio.CancelledError:
        logger.info("Sim OS agent stopping (signal)")
    finally:
        await agent.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OS tool-use agent for the agentic mode (EXPERIMENTAL)"
    )
    parser.add_argument("--node-id", default="sim-os-agent")
    parser.add_argument("--proxy", default="rollout", help="rollout proxy node id")
    parser.add_argument("--n-tasks", type=int, default=128, help="training task pool size")
    parser.add_argument("--seed", type=int, default=0, help="seed of the training tasks")
    parser.add_argument("--group-size", type=int, default=8, help="episodes per task (the GRPO group)")
    parser.add_argument("--tasks-per-wave", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=4, help="turn budget per episode")
    parser.add_argument(
        "--no-fewshot", action="store_true", help="omit the two solved example exchanges"
    )
    parser.add_argument(
        "--prefix-reward", type=float, default=0.5,
        help="reward when the answer line starts with the answer but runs on",
    )
    parser.add_argument(
        "--evidence-reward", type=float, default=0.0,
        help="reward when a tool result contained the needed information (no correct answer)",
    )
    parser.add_argument("--waves", type=int, default=0, help="stop after N waves (0 = until stopped)")
    parser.add_argument("--eval-every", type=int, default=5, help="evaluate every N waves (0 = never)")
    parser.add_argument("--n-eval-tasks", type=int, default=64, help="held-out tasks per evaluation")
    parser.add_argument("--eval-seed", type=int, default=1_000_003, help="seed of the held-out tasks")
    parser.add_argument("--client-id", default="sim-os")
    parser.add_argument(
        "--backend", choices=BACKENDS, default="podman",
        help="podman: real containers, one per task; memory: in-memory simulation",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="container image (podman backend)")
    parser.add_argument(
        "--container-label", default="",
        help="label value of this run's containers (default: $WW_RUN_ID, else the client id)",
    )
    parser.add_argument(
        "--container-concurrency", type=int, default=32, help="concurrent podman commands"
    )
    parser.add_argument(
        "--constrain-calls", action="store_true",
        help="constrained decoding: every turn is one template-format <tool_call>; "
        "answers go through an extra answer(text) tool",
    )
    parser.add_argument(
        "--trajectory-log", default="",
        help="append every episode (tool calls, results, reply, reward) to this JSONL file",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(_main(args))
