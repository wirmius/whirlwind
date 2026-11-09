# Agentic multi-turn GRPO (experimental)

> **Status: experimental.** Built on the [async proxy-rollout
> mode](async_proxy_rollout.md) and covered by GPU-free tests
> (`tests/test_agentic.py`); see [Results](#results) for the end-to-end runs.
> The SWE agent is less tested. APIs and configs may change.

The async mode trains on single-turn traffic: every `generate_external` call
is its own GRPO group. Agents work in **episodes**: they generate, act on the
output (run a tool, check an answer), feed the result back, and generate
again, and the reward arrives only at the end. This mode trains on such
episodes. An agent tags each call with the episode it belongs to; the
training loop regroups the served turns into episodes, compares the episodes
of the same task against each other, and trains every turn with its
episode's advantage.

All of it lives in sibling modules next to the async mode, which stays
unchanged:

| piece | module | adds |
|---|---|---|
| configs | `whirlwind/config_agentic.py` | `AgentRolloutConfig`, `AgentProxyConfig`, `AgenticTrainingConfig` |
| engine node | `whirlwind/rollout/rollout_sglang_agent.py` | chat conversations + tools, server-side tool-call parsing, special-token escaping, template / radix-cache / speculative-decoding knobs, concurrent serving |
| proxy | `whirlwind/rollout/rollout_proxy_agent.py` | `touch` heartbeats, `pause_for_update` / `resume_after_update`, control-lane workers |
| training loop | `whirlwind/training/training_agentic.py` | `AgenticGRPOOrchestrator`: episode grouping, batch alignment, shared-GPU update choreography |
| example agents | `whirlwind/training/sim_os_agent.py`, `sim_reflect_agent.py`, `sim_swe_agent.py` | tool calls on procedurally generated file-system / clock tasks; reflect-and-retry GSM8K (no tools); containerized SWE-Gym bash episodes |

## Episode protocol

An agent serves **one `generate_external` call per (episode, turn)** with
`group_size=1` and describes the turn in `metadata["ww_tag"]`:

```python
{"task": "gsm8k-reflect",   # any string; part of the group key
 "problem": "3f2a...",       # problem id; part of the group key
 "group": 17,                # group id, unique per (client, task, problem)
 "group_size": 4,            # episodes in the group
 "episode": 2,               # 0 .. group_size-1
 "turn": 1,                  # 1-based
 "max_turns": 3, "terminal": False, "reference": "..."}   # informational
```

- **Group key.** `(client_id, task, problem, group)`. A group is complete
  once all `group_size` episodes have shown up in the buffer.
- **Rewards.** When an episode ends, report its return for **every** turn's
  `rollout_id` in **one** `report_reward` call, so its turns become ready
  together. Episodes that never report are dropped with their group after
  `group_complete_timeout_s`.
- **Heartbeats.** A turn's reward is pending until its episode ends. For
  long episodes, call `touch(rollout_ids)` on the proxy every turn: it resets
  those entries' reward TTL, so `reward_ttl_s` measures how long an agent has
  been silent rather than how long its episode is.
- **Conversations and tools.** Send the history either flattened into the
  prompt or as `conversations` (system / user / assistant / tool messages)
  plus optional OpenAI-style `tools`. With `tool_call_parser` set on the
  engine, `ExternalGenerateResult.tool_calls` carries the parsed calls (one
  list per completion); the raw completion stays canonical for training.
  Every string is special-token-escaped before templating, so an observation
  cannot forge a turn boundary.
- **Constrained decoding.** `sampling_constraint` on the request (one of
  `ebnf`, `regex`, `json_schema`, `structural_tag`) restricts the
  completions to an SGLang structured-output grammar, e.g. to enforce the
  tool-call format. Only the agentic engine node serves it; the base node
  rejects such requests instead of ignoring the constraint.

Untagged traffic (or tags without `group_size`) trains exactly like in the
async mode, so single-turn and multi-turn clients can share a run.

The full call/report contract for single-turn clients is in
[agent_connector_spec.md](agent_connector_spec.md); the two example agents
are reference multi-turn clients.

## How a batch is trained

1. The loop pops every ready entry from the proxy and **stages** it under its
   group key until the group is complete.
2. Complete groups leave the stage oldest first, until `min_train_batch`
   **turns** are collected (at most `max_train_batch`, never splitting a
   group). Groups that fall more than `max_weight_version_lag` versions
   behind while staged are dropped (`agentic/groups_dropped_stale`).
3. Each episode's return (the mean of its turns' rewards — agents report the
   same value for each) is z-scored against the other episodes of its group
   with the configured `scale_rewards`, and every turn gets that advantage.
4. Each turn trains as one sequence: its prompt is the conversation up to
   that turn (exactly the tokens the engine saw) and its completion is the
   model's reply. The loss, importance sampling and staleness handling are
   those of the async mode.
5. The model node needs a batch that divides into `micro_batch_size ×
   fsdp_size`. The loop pads the batch with copies of its shortest turns,
   marked `extra["ww_pad"]`, with zero advantage. With `kl_coef: 0` they add
   no gradient; they enlarge the dr_grpo denominator by fewer than
   `micro_batch_size × fsdp_size` samples (`agentic/pad_entries`).

Logged per step: `agentic/groups`, `agentic/episodes`,
`agentic/turns_per_episode`, `agentic/groups_with_signal_frac` (groups whose
episodes did not all score the same), `agentic/mean_return`,
`agentic/staged_groups`, `agentic/staged_entries`, `agentic/groups_trained`,
`agentic/groups_dropped_timeout`, `agentic/groups_dropped_stale`.

## Serving

The agent engine node runs `n_rpc_workers` RPC workers: agents send one small
request per live episode, and SGLang batches concurrent requests itself.
Generations share the engine; adapter swaps, sleep and wake wait for
in-flight generations and hold new ones back, so a swap never overlaps a
request. The radix cache is flushed on every adapter swap (prefix KV computed
under the old adapter is never reused). With LoRA, SGLang supports only
`speculative_algorithm: "NGRAM"`.

The proxy's `n_rpc_workers` should cover the number of live episodes, and
its `n_control_workers` keep `report_reward`, `touch`, `pop_ready_batch` and
pause/resume responsive while every serving worker is busy.

## Two layouts

- **Dedicated GPUs (default).** The engine owns its GPU(s) and never sleeps;
  the updater trains on its own. `engine_sleep_during_update: false`.
- **Shared GPU.** Set `engine_sleep_during_update: true` in the training
  config, and in the engine config `enable_memory_saver: true`,
  `enable_weights_cpu_backup: true`, `disable_cuda_graph: true` and a
  `mem_fraction_static` that leaves room for the updater
  (`configs/rollout/agentic_engine_tools_shared.yaml`,
  `configs/training/agentic_os_tools_shared.yaml`). Before each
  optimizer step the loop calls the proxy's `pause_for_update`: new generates
  park, in-flight ones finish (or the pause fails after
  `pause_drain_timeout_s` instead of sleeping under live requests), and the
  engine sleeps. After the step the adapter is exported, the engine is woken
  (`wake`, requests stay parked), the adapter is pushed, and
  `resume_after_update` releases the parked requests, so nothing is served
  between the wake and the swap. A failed pause, wake or resume shuts the
  cluster down, since a paused engine would otherwise park every agent
  forever.

## Running it

```bash
ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_agentic.sh               # OS tool use, 100 steps
ENGINE_GPU=0 MODEL_GPU=0 bash examples/experimental_agentic.sh               # the same on one GPU
STEPS=3 ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_agentic.sh       # smoke
TASK=reflect ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_agentic.sh  # reflect-and-retry GSM8K
```

The script mirrors the async example but starts `rollout_sglang_agent`,
`rollout_proxy_agent`, `training_agentic` and one of the agents below.

**OS tool use** (`TASK=os_tools`, the default; `sim_os_agent.py`). Each task
is a question about a computer — "which directory contains kepler_61.json?",
"what time is it?", "what access code is written in
/home/user/data/lotus_11.txt?", "how many files are in /home/user/logs?" —
with its own procedurally generated file tree. The model can only find out by
calling the tools `list_dir`, `find_file`, `read_file` and `get_time`; it
ends an episode by replying without a tool call.

The tools run in a podman container (`OS_BACKEND=podman`, the default): one
container from `OS_IMAGE` (default `python:3.12-slim`, pulled on first use)
serves every task. The agent writes all task trees to a host directory (one
subdirectory per task) that is bind-mounted read-only at `/tasks`, and runs
each call as a real command (`ls`, `find`, `cat`, `date`, passed as argv,
never through a shell) with `podman exec -w /tasks/<task>/home/user -e
TZ=<the task's UTC offset>`. Paths are mapped both ways, so each task looks
like its own machine with its tree at `/home/user`: the model's paths are
normalized and placed under the task's directory, and the task directory is
stripped from the output, so no task can reach another's files. The
container has no network, no capabilities, a read-only root and pid and
memory limits; it carries the label `whirlwind.sim_os_agent=<WW_RUN_ID>`,
which the example's cleanup also uses, and the agent removes it and the host
directory on exit. With the real clock, time questions are graded against
the task's local time during the episode. `OS_BACKEND=memory` simulates the
same tools in Python instead (fixed virtual clock; no podman needed). The
example writes every episode — tool calls with their arguments and results,
the final reply and the reward — to `$LOG_DIR/os_trajectories.jsonl`. Training uses a fixed pool of 128 tasks (seven
kinds, one to two tool calls each), 8 episodes per task and up to 4 turns; a
tool result tells the model when the next turn is its last. Every few waves
the agent plays 64 held-out tasks through the proxy's internal `generate`
endpoint, which never enters the training buffer, and logs `os_eval/*`.

Reward is graded on the first line of the final reply: 1.0 when it is exactly
the answer, 0.5 when its first word is the answer but the line runs on, and
otherwise `EVIDENCE_REWARD` (0.2 in the example, 0 by default) when one of
the episode's tool results contained the needed information.
`os/success` and `os_eval/success` count exact answers only.

A pretrained base model (not instruction-tuned) needs help to get started.
Before any training, Qwen3-0.6B-Base calls tools with well-formed JSON
arguments, but it never produces the chat template's `<tool_call>` special
tokens, rarely stops after its answer, and tends to keep calling tools
instead of answering. The chat therefore opens with two solved example
exchanges, and there are two ways to deal with the format:

- **Constrained decoding** (`OS_CONSTRAIN=1`, the default; `--constrain-calls`).
  Every turn must be exactly one tool call in the template's format, enforced
  by an EBNF grammar built from the tool schemas
  (`sim_os_agent.tool_call_grammar`) and passed per request as
  `sampling_constraint={"ebnf": ...}`; the agentic engine node hands it to
  SGLang's grammar backend. The final answer is a call of an extra
  `answer(text)` tool. The model then only chooses the tool, its arguments
  and the answer text; the tags, the JSON skeleton and the end of the turn
  are fixed. Making every turn a call is what makes the format enforceable:
  with a free-text alternative in the grammar, the base model puts its call
  into the text instead. On the 128 training tasks the untrained model makes
  100% template-format calls and answers 73% exactly under the grammar,
  against 0% and under 1% without it.
- **Lenient parsing** (`OS_CONSTRAIN=0`). A bare JSON call at the start of a
  reply counts as a call (`os/wrapped_call_frac` tracks the proper format),
  and the partial rewards give GRPO a gradient: unconstrained, the base model
  retrieves the needed information in about 70% of episodes but answers
  exactly in under 1%. In a 49-step run it learned to put the answer first
  (held-out 5% → 55-66%) but never the `<tool_call>` format (0% throughout).

Under the grammar, SGLang reports the sampled tokens' logprobs from the
masked distribution, so forced tokens (the tags) carry an engine IS ratio
far below 1 and add almost no gradient; the format is enforced at serving
time rather than learned.

**Reflect-and-retry GSM8K** (`TASK=reflect`; `sim_reflect_agent.py`). The
agent fetches GSM8K problems from the dataloader, plays `GROUP_SIZE` (4)
episodes per problem with up to `MAX_TURNS` (3) turns each, and after a wrong
answer asks the model to reflect and try again. An episode scores 1 if its
final answer is correct. `CARRIER=flat` (the default) puts the history in
one user prompt; `CARRIER=messages` sends real chat turns. The agent logs
`reflect/final_acc`, `reflect/first_try_acc` and `reflect/mean_turns`;
validation is the usual single-turn GSM8K accuracy.

**SWE-Gym** (`sim_swe_agent.py`) needs podman, the `datasets` package and the
SWE-Gym instance images (several GB each, pulled on first use), and an
instruct model with a tool-call format (`tool_call_parser` in the engine
config, e.g. `qwen25` for hermes-style calls). Start the same nodes as the
example without its agent, then:

```bash
python3 -m whirlwind.training.sim_swe_agent --repo getmoto/moto \
    --n-instances 8 --episodes-per-instance 4 --max-turns 10 --concurrency 16
```

`--task-family toy` runs small synthetic fixes in one image, to check the
wiring before pulling real instance images.

## Knobs

| knob | where | effect |
|---|---|---|
| `group_episodes` | training | regroup tagged turns into episodes (off = plain async behavior) |
| `group_complete_timeout_s` | training | drop groups whose episodes do not all arrive in time |
| `min_train_batch`, `max_train_batch` | training | turns per optimizer step (groups are never split) |
| `max_weight_version_lag` | training | staleness bound K; agentic default 8, since episodes span several updates |
| `engine_sleep_during_update`, `pause_drain_timeout_s` | training | shared-GPU layout |
| `n_rpc_workers` | engine, proxy | concurrent requests served |
| `n_control_workers`, `backend_wait_timeout_s` | proxy | control-lane workers; how long a call waits for the engine to come up |
| `reward_ttl_s` | proxy | pending-reward lifetime, reset by `touch` |
| `apply_chat_template`, `chat_template_kwargs` | engine | force templating on/off; extra template kwargs (e.g. `enable_thinking`) |
| `tool_call_parser` | engine | SGLang FunctionCallParser name for server-side tool-call parsing |
| `sampling_constraint` | request (`ExternalGenerateRequest`, `PromptBatch`) | per-request structured-output grammar for the completions |
| `disable_radix_cache`, `speculative_algorithm`, `speculative_num_draft_tokens` | engine | prefix cache; NGRAM speculative decoding |

## Results

OS tool use, Qwen3-0.6B-Base (pretrained, not instruction-tuned) with a LoRA
adapter, tools in a podman container, engine and updater on one GPU each,
128 training tasks, 8 episodes per task, 64 held-out tasks.

**Constrained decoding** (the default; 100 optimizer steps, about 45 min):

| held-out tasks | before training | after 100 steps |
|---|---|---|
| exact answers, all kinds | 72% | 100% |
| find the directory of a file | 0% | 93% |
| count the files in a directory | 22% | 100% |
| time / weekday | 90% / 89% | 100% / 100% |
| date, read a file, find then read a file | 100% | 100% |
| calls in the template's `<tool_call>` format | 100% | 100% |

The grammar fixes the format from the first step; RL then teaches the task
kinds the untrained model gets wrong — answering with the directory instead
of the file's path, counting the listing — while tool calls per episode drop
from 2.6 to 2.4.

**Without the grammar** (lenient parsing; a run stopped by an interruption at
step 49 of 100): calls in the `<tool_call>` format stayed at 0% throughout,
the right tool with the exact right argument at about 45% of calls, and
exact answers at 0-11% on the held-out tasks; what improved was putting the
answer first (answer lines starting with the right value: 5% → 55-66%).

Every episode — calls, arguments, real command output, final reply, reward —
is in `$LOG_DIR/os_trajectories.jsonl`.

## Limitations

- **Terminal rewards only.** Every turn of an episode gets the same
  advantage; there is no per-turn credit assignment.
- **Turns train as separate sequences.** A turn's prompt repeats the
  conversation before it, so long episodes cost roughly quadratic tokens, and
  episodes with more turns carry more weight in the batch.
- **Group completeness is by arrival.** An episode counts as present once any
  of its turns is popped; agents must report all of an episode's turns in one
  call (the example agents do).
- **LoRA only, no restart safety**, as in the async mode.
- **The SWE agent** is a reference client: small models solve essentially no
  real instances, so it checks the machinery rather than demonstrating
  learning.
