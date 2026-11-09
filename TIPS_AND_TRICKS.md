# Tips and tricks

The rules we learned building and validating Whirlwind against TRL's
`GRPOTrainer` (Qwen3-0.6B-Base on GSM8K, two to four GPUs, 65 to 200
optimizer steps). Each entry gives the rule, the mechanism, and how to check
it in this codebase. Read §1–§4 before touching precision, loss scaling,
sleep/wake or engine-logprob handling; §8 lists the diagnostic switches.

Numbers are GSM8K validation reward on Qwen3-0.6B-Base unless stated
otherwise. Seed noise alone is 2–3 points, so differences inside the
0.58–0.66 band are parity.

## 1. Precision and dtype

**1.1 Keep an fp32 optimizer master under FSDP2 mixed precision:
`torch_dtype: "float32"` with `fsdp_mixed_precision: true`.** FSDP2 stores
the *sharded* parameter in its load-time dtype and the optimizer steps that
tensor; `param_dtype=bf16` only casts the all-gathered compute copy. GRPO's
z-scored advantages produce AdamW updates of about 1e-5 per parameter, roughly
12× below the bf16 ULP at |w| ≈ 0.02, so a bf16 master rounds them to zero.
Observed: 99% of a layer's elements bit-frozen over 51 steps, reward flat
(0.148 → 0.161) versus 0.152 → 0.487 with an fp32 master. The model node
warns at init when the master is bf16. DDP tolerates a bf16 master because it
accumulates differently; do not generalize across sharding strategies.

**1.2 Validate precision choices under the real training signal.** An SFT
sanity run passes with a bf16 master because cross-entropy gradients clear
the ULP; GRPO's tiny updates do not. A proxy with larger gradients proves
nothing about RL-scale updates.

**1.3 The reported loss is a noisy reporter; the gradient is the signal.**
Under bf16 compute with importance sampling, the loss scalar swings 40–128%
across micro-batch sizes on the *same* weights and batch while `grad_norm`
stays within 10–13%. The chain: per-token logprob noise → sequence sum →
`exp` → a ratio multiplying a surrogate whose mean is ≈ 0. Judge training by
`grad_norm` and validation reward, never by loss swings.

**1.4 Only upcasting the logits before log-softmax helps (`WW_LOGITS_FP32=1`,
TRL's `cast_lm_head_to_fp32`); upcasting inside the loss (`WW_IS_FP32`,
`WW_LOSS_FP32`) is a bit-identical no-op.** The bf16 corruption enters in the
forward; the loss path already runs in fp32 by promotion. The residual noise
is batch-shape dependent (different micro-batch sizes → different matmul tile
shapes → different rounding) and only an fp32 forward would remove it.
Accept a ~10% loss/grad difference between micro-batch sizes as this effect.
`WW_LOGITS_FP32` doubles logits memory.

**1.5 Precision "config matches" must include dtype fields**, not just
learning rate, loss type and IS flags.

**1.6 On some GPU and driver combinations the cuDNN fused-attention backward
returns NaN at the first post-update step.** `WW_SDP_DISABLE_CUDNN=1` falls
back to the flash kernel; `WW_ATTN_IMPL=eager` bypasses SDPA entirely;
`WW_DETECT_ANOMALY=1` localizes the first NaN gradient (2–3× slower).

**1.7 CPU and GPU bf16 do not behave the same.** Raw AdamW state in bf16 NaNs
on CPU around step 19 where the GPU tolerates it, and the stock
`clip_grad_norm_` has no mesh backend for CPU DTensors. The CPU-offload path
therefore carries three adapters: a custom fp32-internal Adam step, a custom
clip, and an optimizer-state device sync. Re-derive every op when you move a
path to CPU.

## 2. Loss scale, microbatching, gradient accumulation

**2.1 Loss denominators are per-rank: `data.batch_size // world_size` (and the
token total `// world_size`) in the microbatch path.** FSDP2 reduces gradients
with `ReduceOp.AVG`, so each rank must normalize by its own share. The
microbatch split in `_use_grad_accum_microbatch_size` runs *before*
`_fsdp_using_function` scatters the batch to ranks, so `data` there is the
global batch; with the undivided global count the microbatched gradient was
`world_size`× smaller than the `micro_batch_size: -1` path (a two-rank probe
measured 0.5001×). An older layout sharded *before* the split, where dividing
again doubled the gradient ("learns but lands low": 0.50 vs 0.62); if you move
the decorator, re-derive this. Both paths must produce the same gradient at any
world size, and the single-GPU unit tests cannot see either mistake (§6.7).
Re-check the reduce op if you change `fsdp_grad_reduce_dtype`.

**2.2 Per-microbatch losses of fixed-denominator reductions (`dr_grpo`,
`bnpo`, `ce`) are summed, never averaged.** Each microbatch loss is already
divided by the full-batch denominator; averaging them made the reported loss
micro-batch-size dependent (32× too small at mb=4) while the gradient, which
`loss.backward()` computes per microbatch, was correct the whole time.

**2.3 Never compare absolute loss values across frameworks, accumulation
settings or micro-batch sizes.** With 16 accumulation steps the reported loss
was exactly 16× TRL's at an identical gradient. AdamW is invariant to a
constant loss scale; compare `grad_norm` and validation reward.

**2.4 Topology changes change `world_size`, and with it the scale.** A
"dr_grpo hurts under colocate" finding was this confound: colocate ran at
world size 2 with the bug active, non-colocate at world size 1 where it was
inert. Any A/B that varies topology varies world size.

**2.5 Compare like learning-rate schedules.** A 65-step cosine run has
annealed to ~0 by step 64 while a 200-step run is near peak LR at step 64
(0.52 vs 0.57 at the same step, same code). Short runs are diagnostics, not
baselines; `lr_schedule_steps` keeps the long schedule in a short run.

**2.6 `max_grad_norm: 1000.0` is clipping-off.** Fine for the validated
mainline, where norms stay around 0.05–0.1; set 1.0 for anything that can
spike (per-token reward shaping produced gradient-norm medians of 1.5–45 with
spikes into the thousands).

**2.7 `micro_batch_size: -1` bypasses the microbatch loop** (one forward and
backward over the rank's shard). It is the "is microbatching the problem?"
A/B, needs about 50 GB of bf16 logits at 128 completions × 1024 tokens ×
151k vocabulary (100 GB with `WW_LOGITS_FP32`), and only fits the manual node.

**2.8 Chunked lm_head bounds peak logits memory but not the autograd graph.**
At 8k-token completions every chunk's fp32 logits stay alive until that
microbatch's backward: ≈ tokens per forward × 151k × 12 bytes, about 54 GB
at mb=4. mb=2 was clean. 1024-token runs never hit it.

**2.9 `max_completion_length` (model config) is the dr_grpo denominator cap
and must equal the rollout's `max_gen_len`**; config loading warns when they
disagree. Gradient accumulation is orchestrator-owned:
`TrainStepRequest.accum_steps` carries it to the model node, which scales the
loss before backward; the HF-Trainer node pins its own accumulation to 1.

## 3. Colocated sleep/wake and FSDP2 state

**3.1 Never reassign FSDP2's `_sharded_param_data`.** It aliases the storage
of `sharded_param._local_tensor`: the forward all-gathers from the first, the
optimizer writes the second. Moving the shard to CPU by reassignment breaks
the alias, the forward reads an increasingly stale copy while the rollout
samples the live policy, `old − sampling` widens and the engine IS ratio
collapses within ~15 steps (0.14 → 0.02, reward stuck at 0.18). It looks like
a precision bug and is a read/write desync. Sleep therefore offloads
**optimizer state only**; if parameters must leave the GPU use
`fsdp_cpu_offload: true` (FSDP's `CPUOffloadPolicy`, which never reassigns).
Both reach 0.63 at 64 steps where the reassigning path reached 0.18–0.24.

**3.2 "The HF stack survives it" does not make a path safe.** The HF-Trainer
node tolerated the reassigning sleep because Accelerate's fuller
initialization re-establishes the alias; raw `fully_shard` plus raw AdamW did
not. Verify under your own init path.

**3.3 SGLang colocate needs `enable_memory_saver: true` (in step with the
training config's `memory_management_enabled`), `disable_cuda_graph: true`
(CUDA-graph replay crashes with FSDP2 co-residence) and
`enable_weights_cpu_backup: true` (otherwise resume loses the weights and
the engine emits gibberish).**

**3.4 Phase order: sleep the model before waking the rollout.** The engine
cannot reclaim its KV pool until the trainer has freed optimizer state. The
reward phase runs with the engine asleep; a reward node that needs generation
must wake, generate and sleep the rollout itself, in `try/finally`.
`steps_per_generation > 1` is refused under colocate because that choreography
is not wired for the window loop.

**3.5 vLLM LoRA plus colocate is blocked** (the sleep allocator conflicts with
the LoRA memory pool). Use SGLang for colocated LoRA; a pool with one engine
per GPU sidesteps SGLang's `dp_size == 1` LoRA assertion. LoRA r=16, α=32 on
q and v matched full fine-tuning at 64 steps with 130× smaller adapter sync
and about 9× faster optimizer steps at GAS 1.

**3.6 An fp32 master costs nothing measurable at 0.6B.** The CPU↔GPU copy is
0.5% of an optimizer step that is 90% generation. Storing bf16 to halve the
copy breaks the master (§1.1). An fp32-master CPU-offload run OOMs in
colocate because the all-gathered fp32 parameters double the GPU footprint;
the working combination is bf16 compute parameters with fp32-internal Adam.

**3.7 Cache the HF config at init.** Calling `AutoConfig.from_pretrained` on
every weight export issued hundreds of Hub requests per run. Run with
`HF_HUB_OFFLINE=1` once the cache is complete.

**3.8 Check the FSDP2 wrap count in the log.** Per-layer wrapping needs the
decoder-layer class; a discovery that matches nothing does not fail, it
shards the whole model as one root unit, and activation checkpointing (which
wraps the same units) silently does nothing. Memory and step time then look
plausible but wrong. The model node finds decoder layers by a structural scan
and logs `FSDP2 wrap: N decoder-layer unit(s) + root`; N should equal the
model's layer count, and a zero logs a ROOT-ONLY warning. Recheck it after a
transformers upgrade. Activation checkpointing also needs the model's KV cache
off: the backward recompute of each checkpointed layer otherwise appends its
keys to the forward's cache a second time and attention fails on the doubled
length. The model node sets `config.use_cache = False` at load (the trainer
never generates).

## 4. Rollout engines and logprobs

**4.1 Engine logprobs must be computed after temperature scaling, and the
trainer's `temperature` must equal the rollout's.** vLLM's raw mode computes
log-softmax at T=1 while the trainer samples at T=0.7: a constant ≈ 0.07 per
token offset, `exp(40)` over a 570-token completion, the sequence-mask cap
trips on every sequence and the loss is zero. `logprobs_mode:
"processed_logprobs"` fixes it; SGLang divides by temperature before the
log-softmax. Config loading warns on a temperature mismatch.

**4.2 Missing engine logprobs are NaN sentinels, never 0.0.** `old − 0.0`
annihilates the ratio (a zero-logprob run scored 0.10). Both engines NaN-pad;
the model node neutralizes NaN to ratio 1 and asserts length alignment
between the logprob list and the completion tokens. Keep the assert.

**4.3 vLLM's `processed_logprobs` are post-top-p and renormalized**, so the
trainer's full-softmax `old` sits below them by `log(kept mass)` ≈ −0.022 per
token at top_p 0.9. Under `sequence_mask` this collapses the ratio on long
completions (step-1 means 0.06–0.16); TRL shows the same (0.067) and recovers
as completions shorten. Do **not** "fix" it by truncating the trainer's
logprobs to match: that restored ratio ≈ 1 and removed the accidental
length-based curriculum, and the run stopped learning. Fingerprint: a
`signed_mean` of about −0.02 with the deficit on uncertain tokens and exact
agreement on certain ones.

**4.4 SGLang returns full-vocabulary logprobs, so its engine IS ratio is
≈ 1.** Renormalizing them to the top-p set (as vLLM's processed logprobs
are) reproduces the vLLM collapse. Which logprob flavor an engine returns is a
first-class experimental variable; record it with every run.

**4.5 SGLang's stock `log(softmax(x))` differs from vLLM's fused fp32
log-softmax by a stable −0.014 nats per token on identical context**, 6×
more on the first generated token. The bias is weight-independent, so it
never self-corrects, and `exp(0.014 × 500)` is three orders of magnitude over
a 500-token completion under sequence-level IS. Remedies: vLLM for IS-on
runs, or SGLang's `rl_on_policy_target: "fsdp"` (stable log-softmax, but it
forces deterministic sampling and is 5–10× slower). Do not patch SGLang's
decode sampler in place: the in-tree fp32 patch broke CUDA-graph capture.

**4.6 Compare engine logprobs only at positions whose whole preceding
context is identical.** "Same token at the same position" conflates
numerical divergence with context divergence; 85% of the apparent
divergence was the latter. Force-decode one engine's tokens through the
other for a definitive measurement.

**4.7 Sequence-level IS is fragile on long completions.** A +0.006 per-token
bias over ~550 thinking tokens is `exp(3.3)` ≈ 27, far above a cap of 3,
so sequences are masked to zero and steps die; at ~7000 tokens per-sequence
ratios ranged 0.19–2.8. Use `engine_is_mode: token_mask` or `token_truncate`
for long-context runs; sequence-level caps work only when the per-token bias
is ≈ 0, and the cap never prevents collapse from the low end.

**4.8 A low engine IS ratio is length-driven and benign by itself.** Both
this codebase and TRL settle around 0.5 and never return to 1.0, and TRL
learns with ratios of 0.14–0.27. `old − sampling = log(M)` with `M` the
captured top-p mass: an entropy rise saturates `M` at the top-p floor and the
ratio falls from the low end while `over_cap_frac` stays 0%. A ratio collapse
is an entropy symptom of the update rule, not an IS-math bug.

**4.9 Engine IS correction is roughly neutral at this scale** (controlled
pairs: +0.04 / +0.03 / −0.01 / +0.03 at steps 64–200). Keep the TRL default
(on); treat "IS off" as a diagnostic arm, not a cure.

**4.10 Engine logprobs stay quarantined from the PPO ratio.** `coef_1` uses a
fresh detached forward (≈ 1 at GAS 1, clipping dormant); engine logprobs
enter only as the bounded, detached multiplier on the per-token loss, exactly
as in TRL. Do not fold them into the ratio.

**4.11 Alignment is prepend-zero, and prompt length must not count right
padding.** An index-shift sweep minimizes at shift 0; the padding-length bug
only activates at `micro_batch_size > 1`. `tests/test_trl_padding_alignment.py`
pins the alignment against a no-pad reference.

**4.12 Weight export is not a divergence source.** Weights round-trip
bit-identically through export and reload, and trained weights shift engine
logprobs as expected. When trainer and engine disagree, look at the logprob
computation, not the sync.

**4.13 Use the fused full-vocabulary `log_softmax`** over bf16 logits rather
than a hand-rolled logsumexp (per-token error 0.0018 vs 0.016).

**4.14 Normalize SGLang's `finish_reason`**: it is a dict
(`{"type": "length", ...}`), and comparing it to a string silently disabled
truncation masking and the clipped-ratio metric in every run. Fingerprint:
`completions/clipped_ratio` = 0 while `mean_length` ≈ `max_gen_len`.

**4.15 GRPO advantages group by `entry_id`** (the `rollout_per_prompt` copies of
one prompt), with a Bessel-corrected NaN-aware std and `scale_rewards: group` by
default, all matching TRL. `whirlwind/reward/advantages.py` holds the one
implementation the synchronous reward node and the async updater share; keep
new reward variants on it.

**4.16 Measure a base model's tool-use rate before training on tool calls.**
A pretrained (not instruction-tuned) checkpoint barely produces the chat
template's special tokens: Qwen3-0.6B-Base never emitted `<tool_call>` and
rarely `<|im_end|>`, even with demonstrations in the prompt, while writing
well-formed JSON calls and retrieving the right information most of the time.
With strict parsing and exact-match rewards every group scores 0 and GRPO has
no gradient. Run the untrained policy on the task first and count calls,
answers and exact successes; then add demonstrations, accept the format the
model does produce (and log how often it uses the proper one), and give
partial credit for the intermediate skills (`sim_os_agent.py`). Or enforce
the format with constrained decoding: a grammar that makes every turn one
template-format tool call (the answer through an `answer` tool) took the same
untrained model from 0% to 73% exact answers. Two catches: a grammar with a
free-text alternative does not help (the model writes its call as text), and
bounded repeats (`{0,200}`) make xgrammar's per-token masks hundreds of times
slower than `*`; cap lengths with `max_gen_len` instead.

## 5. Importance-sampling dynamics and diagnostics

**5.1 A healthy IS-on run and a collapsing one fork in the first 15–30
optimizer steps**, long before validation shows it. Healthy: `seq_sum`
−5.6 → −1.1 and ratio 0.14 → 0.44 by step 30. Dying: `seq_sum` −5.6 → −18,
ratio → 0.001. `grad_norm` alone does not separate them early; the
per-step `[STEP-DECOMP]` trace (`WW_PRINT_STEP1_DIAG=1`) does. A bounded
`signed_mean` of about −0.005 per token is healthy; drifting to −0.07 is the
collapse.

**5.2 Sequence-mask IS collapses the gradient on any staleness by design.**
Replaying another run's rollouts through a trainer is on-policy only at step
1; by step 15 the replayed ratio was 0.02 vs 0.18, by step 47 0.009 vs 0.22.
Only a step-1 comparison with identical weights and data tests trainer
fidelity (gradient within 4%, cosine 0.996–1.0). A script that judges
"trainer bug" from end-of-run weight distance is wrong.

**5.3 The two model nodes share the loss path and are bit-identical stage by
stage** given the same weights, batch and micro-batch size (2048 comparisons,
zero differences). A manual-vs-HF difference comes from inputs or from the
backward/reduce path (gradients differ at 1e-5), never from `_compute_loss`.
Verify with `WW_IS_DIAG_DIR` dumps on a replayed round.

**5.4 TRL's `sampling_logp_difference/mean` is `mean(|old − sampling|)`;
ours is signed.** An "opposite signs" finding was this convention mismatch.

**5.5 Iterate diagnostic keys in sorted order before cross-rank
`all_reduce`.** Hash-ordered iteration differs per process and cross-wires
collectives; impossible values such as `is_ratio/mean < 0` or
`uncapped_min > uncapped_max` are the symptom. Only visible at world size > 1.

## 6. RPC, timeouts, long-context runs

**6.1 `WHIRLWIND_RPC_TIMEOUT` (default 300 s) must exceed the longest single
RPC.** An 8k validation generation takes about 18 minutes: set 3600. The
orchestrator's node
is constructed with `fail_on_timeout=True`, so a timeout raises `RpcError` and
the cluster shuts down instead of dying on `AttributeError` from a `None`
reply; elsewhere use `call_typed` or guard the result.

**6.2 Raise `WHIRLWIND_HEARTBEAT_TIMEOUT` (default 15 s; 600 at 8k) for
long-context runs.** A multi-million-token validation payload blocks the
trainer loop in synchronous parsing past the heartbeat budget, the registry
drops the trainer mid-RPC and the run wedges. `WHIRLWIND_HEARTBEAT_FATAL_MISSES`
(default 12) decides when a service node gives up on a vanished orchestrator
and exits instead of hanging for hours.

**6.3 `WHIRLWIND_NCCL_TIMEOUT` (default 600 s) must cover the longest wait of
a standby rank.** At data-parallel 2 with 8k context the lead rank awaits a
15–20 minute validation rollout while its peers sit in a collective; the
default watchdog kills them.

**6.4 `validation_log_table: false` for long-context runs.** Tables serialize
every completion (30–60 MB at 8k) through W&B's slow path and stall the
tracker RPC past every watchdog.

**6.5 `allow_auto_truncate: true` (rollout) whenever prompt plus `max_gen_len`
can exceed the context window**; SGLang otherwise rejects the request and the
run dies on the first generate.

**6.6 Every FSDP rank runs the same collectives in the same order.** Rank 0
serves RPC and scatters; the others wait in `_standby`. A rank-dependent
branch deadlocks or corrupts. A handler exception on rank 0 still joins the
gather with an error sentinel and releases the lock.

**6.7 World-size bugs are invisible on one GPU** (`// 1` is a no-op). Validate
numerics changes on at least two ranks.

**6.8 Typed handlers need a typed contract.** whirlcoms delivers raw dicts to
handlers registered without one; decorate with `@<Message>.decorate_serdes()`
(see `generate_external` and `pop_ready_batch` in `whirlwind/rollout/rollout_proxy.py`).

**6.9 SHM payloads leak if the receiver dies before reading.** Payloads over
64 KB go through `/dev/shm/whirlwind-shm-*` and the receiver unlinks them;
sweep the directory after a crash, and check `df -h /dev/shm` when a run
stalls. Crashed runs can also leave process trees (standby ranks, engine
detokenizers); nodes now exit after missed heartbeats, but check `nvidia-smi`
after a crash.

**6.10 If NCCL segfaults inside communicator initialization, suspect the
host's InfiniBand stack before the code.** On a machine whose IB/verbs
libraries are broken, `ncclCommInitRank` dies with SIGSEGV for every process
group, even at world size 1 and inside SGLang's tensor-parallel init, and the
run looks like "the rollout node never registered" or "the model node died
right after the first weight export". `NCCL_IB_DISABLE=1` fixes it; verify
with a two-line `torchrun` all-reduce before launching anything bigger.

**6.11 Keep a tensor-parallel or FSDP group on GPUs that share a fast link.**
On a two-socket host, an SGLang `tp_size: 2` engine on two GPUs attached to
different CPU sockets (`SYS` in `nvidia-smi topo -m`) hung silently during
NCCL setup, so the rollout node never registered; the same configs on an
NVLink pair (`NV#`) ran normally. Check the topology matrix before choosing
`--rollout-gpus` / `--model-gpus`. The async mode's engine and updater talk
over IPC, not NCCL, so they can sit on any two GPUs.

**6.12 Diagnostic harnesses never write into a captures directory.** A replay
once overwrote capture files by reusing the orchestrator's filenames; captures
are read-only inputs.

**6.13 Hundreds of rootless podman containers exhaust the kernel key quota.**
Each container gets its own session keyring, and the per-user kernel quota
is about 200 keys (`/proc/sys/kernel/keys/maxkeys`), shared with everything
else the user runs. Past it, `podman run` fails with `crun: join keyctl ...:
Disk quota exceeded`. Share containers where the work allows it
(`sim_os_agent.py` runs every task in one container, each with its own
working directory over a read-only bind mount), or create containers that
need no keyring with podman's `keyring = false` (a `containers.conf` passed
through `CONTAINERS_CONF_OVERRIDE`). Count keys in `/proc/key-users` before
and after.

## 7. Methodology

**7.1 One-variable A/Bs decide; multi-variable "fixes" prove nothing.** The
decisive experiments each changed one thing (loss reduction; model module;
sleep/wake implementation). An IS A/B was invalid because its two reference
runs used different FSDP versions.

**7.2 An incomplete code trace is not a root cause.** Twice a root cause was
declared from a trace that missed a mutating step (a hidden
`set_mixed_precision` call; reasoning from config defaults the run overrode).
Any "X does Y differently" claim needs direct inspection of the live object
(dtype probe, OOM-or-not, bit-exact diff). A correlation across a handful of
node variants is not a mechanism either: the three-way comparison exonerated
the FSDP wrap and optimizer wrapping correctly but blamed hook order and
activation checkpointing, and the real cause was the sleep/wake alias.

**7.3 Runtime counter-evidence should stop a hypothesis before the test.**
"The HF node runs pure fp32" was already falsified by that run not OOMing.

**7.4 Bit-exact tests beat smoke tests.** The weight round-trip (0 differing
elements), the manual-vs-HF replay (loss difference 0.0 every round) and the
padding alignment against a no-pad reference closed questions; smoke runs
never did. Pin numerics with exact-reference tests.

**7.5 Capture-based replay separates reporter noise from training dynamics;
do it before launching 200-step A/Bs.** Persist start-of-step weights, the
full `TrainStepRequest` and the node's reported loss and `grad_norm`
(`WW_CAPTURE_EVERY`, `WW_CAPTURE_DIR`), replay through another node or
config, compare. Hold `micro_batch_size` constant in replays: a "manual loss
2.5× HF" finding was entirely a micro-batch mismatch plus the mean/sum bug.

**7.6 Trainer fidelity is proven at step 1 of a replay** (identical weights
and data ⇒ identical gradient), never by end-of-run weight distance (§5.2).

**7.7 Distinguish stability from learning.** A checkpoint-kill test (the
manual updater loading a healthy HF checkpoint and improving it) showed the
updater *maintains* a good policy and fails to *learn* from base; that moved
the search from "the updater breaks something" to "a per-step perturbation
the IS amplifier turns into divergence".

**7.8 Evaluation.** Seed noise is 2–3 points; single-sample accuracy on one
checkpoint bounces ±3–8 points between evaluations. Compare on avg@k with
k = 8, never on first-sample accuracy. The trainer's validation reward and an
external evaluation pack are different harnesses (+7–11 points absolute at
matched k, same prompts; orderings agree) — never compare levels across
them. Evaluate every checkpoint with the sampling settings of training, a
fixed budget, and a held-out shift set (MATH-500) next to the in-domain set;
pass@k is the cold-start signal, avg@k the comparison metric. Engine
nondeterminism at a fixed seed flips the sign of small gaps.

**7.9 A result counts only if the run is clean**: optimizer-only sleep/wake
(or a path proven resilient), per-rank denominators, no known-broken engine
configuration (renormalized logprobs, zero-logprob sentinel,
temperature-mismatched logprobs, sequence-mask IS over very long
completions). Record model, rollout, reward and training configs, seed and
commit with every result, in the repository, not on node-local disks.

**7.10 Keep a do-not-rewalk list.** Retracted or dead hypotheses for the
training-collapse investigation: bf16 forward gradient bias, IS-cap clipping,
weight desync between rollout and trainer, microbatch aggregation,
activation-checkpointing presence or mechanism, the clip path, optimizer
wrapping, early-basin stochastic selection, top_p 1.0, the 0.0 → NaN
sentinel as a fix, and truncating the trainer's logprobs to the top-p set.
The bf16-master hypothesis was retracted once and then re-confirmed for FSDP2
specifically (§1.1).

## 8. Debugging toolbox

Everything is off unless the variable is set.

| switch | what it does |
|---|---|
| `WW_PRINT_STEP1_DIAG=1` (+ `WW_PRINT_STEP1_DIAG_N`, default 3) | prints the `[STEP-DECOMP]` line for the first N optimizer flushes on rank 0: PPO `coef_1`, engine ratio mean/min/max/uncapped, `seq_sum`, `signed_mean`, pre- and post-IS loss, gradient norm. The early-fork canary (§5.1). |
| `WW_IS_DIAG_DIR` (+ `WW_IS_DIAG_NODE`, `WW_IS_DIAG_STEP`) | dumps shape, dtype, statistics and a content hash of every intermediate in `_compute_loss` per call; diff across nodes or configs (§5.3). |
| `WW_LOGITS_FP32=1` | fp32 logits before the log-softmax (§1.4). `WW_IS_FP32`, `WW_LOSS_FP32` are documented no-ops kept for A/Bs. |
| `WW_DETECT_ANOMALY=1`, `WW_SDP_DISABLE_CUDNN=1`, `WW_ATTN_IMPL=eager` | NaN localization and attention-backend fallbacks (§1.6). |
| `WW_CAPTURE_EVERY=N`, `WW_CAPTURE_DIR`, `WW_CAPTURE_EXIT=1` | the capture orchestrator (`training_grpo_capture.py`): every N steps saves start-of-step weights, the full `TrainStepRequest` and the reported metrics; `EXIT` stops after capturing (§7.5). |
| `WW_REALDATA_CAPTURE_OPTSTEP`, `WW_REALDATA_CAPTURE_DIR` | one-shot dump of the per-forward input tuples and weights at one optimizer step, for offline accumulation replay. |
| `WHIRLWIND_RPC_TIMEOUT`, `WHIRLWIND_HEARTBEAT_TIMEOUT`, `WHIRLWIND_HEARTBEAT_FATAL_MISSES`, `WHIRLWIND_NCCL_TIMEOUT` | the timeouts of §6. |
| `NODE_COMMS_ENSURE_REGISTER=1` | block node startup until registration succeeds (bringup sets it). |
| `WW_RUN_ID` | namespaces IPC endpoints, checkpoints and the torchrun port. |
| `HF_HUB_OFFLINE=1`, `WANDB_MODE=offline` | run without network once caches exist. |
| `WW_UPDATE_DIAG=1` (+ `WW_UPDATE_DIAG_DIR`, `WW_UPDATE_DIAG_STEP`, `WW_UPDATE_DIAG_SAVE=1`) | `_update_diag.py`: statistics of gradients, parameters and optimizer state at each update stage, hashable across nodes; `SAVE` dumps the full tensors. |
| `WW_STEP1_BITDIFF=1` (+ `WW_STEP1_BITDIFF_OUT`, `WW_STEP1_CAPTURE=1`, `WW_STEP1_CAPTURE_OUT`) | `_step1_bitdiff.py`: bit-level fingerprint of the first optimizer step for manual-vs-HF divergence hunts. |
| `WW_IS_DECOMP_DUMP=1` (+ `WW_IS_DECOMP_DIR`) | one-shot dump of the loss decomposition before and after the engine IS multiplier. |
| `WW_TRL_COEF_LOG=1` (+ `WW_TRL_COEF_LOG_N`, `WW_TRL_COEF_DUMP_DIR`) | HF-Trainer node: log or dump the PPO coefficients per step. |
| `WW_NO_ENTROPY=1` | skip the per-token entropy diagnostic (saves a full-vocabulary pass). |
| `WHIRLWIND_SEQ_DUMP=<path>` | reward node: append the scored sequences to a file for inspection. |
| `NODE_COMMS_DBG_STDOUT=1`, `NODE_COMMS_TIMING=1`, `NODE_COMMS_RPC_FAIL_TIMEOUT=1` | whirlcoms: RPC debug prints, call timing, and strict mode (every node raises on an RPC timeout). |
| `WW_BRINGUP_GRACE_S`, `WW_BRINGUP_TERM_WAIT_S` | bringup teardown: grace period before nodes are considered wedged (default 120 s) and the wait after SIGTERM (default 20 s). |

Metrics worth watching: `sampling/importance_sampling_ratio/{mean,min,max,uncapped_mean}`
and `sampling/sampling_logp_difference/{mean,max,signed_mean,seq_sum_mean}`
(a stable non-zero `signed_mean` is a systematic engine/trainer bias; a
negative `is_ratio/mean` means cross-wired collectives), `model_is/coef_1_mean`
and `clip_ratio/region_mean` (PPO side), `completions/clipped_ratio` and
`completions/mean_length` (rollout; zero clipped ratio at the cap is the
`finish_reason` bug), `grad_norm` (max across ranks) and `val/reward_mean`.

Config knobs that double as diagnostics: `micro_batch_size: -1` (§2.7) and
`engine_is_correction: false` (§4.9).

## 9. Known limitations

- Optimizer state is not checkpointed on the CPU-offload path; a resume
  restarts Adam moments.
- `steps_per_generation > 1` runs without cached old logprobs (the PPO ratio
  is 1 inside the window) and is refused under colocate.
- The async proxy-rollout and multi-turn agentic modes are experimental:
  LoRA updates only, no restart safety for the replay buffer, and terminal
  episode rewards only (every turn of an episode gets the same advantage;
  docs/async_proxy_rollout.md, docs/agentic_multiturn.md).
- SHM payload files leak if the receiver dies before reading (§6.9).
- Relative to TRL's `GRPOTrainer`: no DAPO, CISPO, SAPO, LUSPO or VESPO loss
  variants, no off-policy sequence masking, no entropy-quantile filtering, no
  frozen reference model or reference-model synchronization (the KL term uses
  the policy's own logprobs), no learned reward models. Loss, PPO clipping,
  advantage normalization and the engine IS correction follow TRL.
- Same machine only (IPC and `/dev/shm`).

## 10. Directions we did not finish

- Per-turn credit assignment for multi-turn episodes, and training an
  episode as one packed sequence instead of one sequence per turn.
  (Colocated generation/training overlap cannot speed up training on shared
  GPUs, ceiling ≈ 1.6%; only the decoupling the async mode provides
  justifies asynchrony.)
- PPO-style clipping specific to stale samples in the async mode, if the
  version-lag metrics show the bounded engine IS ratio is not enough.
- A sequence-mask vs token-mask IS A/B at 8k context.
