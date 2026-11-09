# Message Types and Dimensionality

## Base Classes

### `Message`

Abstract base for all inter-component messages. Provides serialization and RPC decoration.

**Tracing fields** (scalar, one per message):
- `dataset_id: str` — unique source dataset identifier
- `entry_id: int` — index within dataset
- `rollout_id: int` — rollout index
- `subrollout_id: int` — sub-rollout index (generation stage)

**Methods**:
- `to_dict()` → plain dict (via `dataclasses.asdict`)
- `from_dict(data)` → self (classmethod, ignores extra keys)
- `decorate_serdes()` → RPC endpoint decorator (classmethod)

### `ArgsKwargs(Message)`

Generic message for trivial payloads. Stores positional arguments in ``_msg_args``
and keyword arguments in ``_msg_kwargs``. Supports attribute access (``msg.path`` →
``msg._msg_kwargs["path"]``) and positional indexing (``msg[0]`` →
``msg._msg_args[0]``).

Subclasses declare payload constraints via class variables:

```python
class WeightSyncRequest(ArgsKwargs):
    _must_have_kwargs_types = {"checkpoint_path": str, "step": int}
```

Used for simple 1--3 field messages where a full ``@dataclass`` would be overkill.
Messages using ``ArgsKwargs`` are noted with "(keyword)" in their field tables below.

### `BatchedMessage(Message, ABC)`

For messages carrying per-sample lists. Subclasses implement:

- `batch_size` → `int` — number of samples
- `split_into(n)` → `List[Self]` — split across n ranks (duplicates if batch < n)
- `join(shards)` → `Self` — concatenate shards back together

**Per-sample tracing fields** (one per sample):
- `dataset_ids: List[str]`
- `entry_ids: List[int]`
- `rollout_ids: List[int]`
- `subrollout_ids: List[int]`

## Concrete Messages

### `PromptBatch` (BatchedMessage)

Sent: **Orchestrator → Rollout**

| Field | Type | Dim |
|---|---|---|
| prompts | List[str] | [batch] |
| responses | List[str] | [batch] (ground-truth answers, passed downstream) |
| system_prompt | Optional[str] | scalar (system message for chat template) |
| step | int | scalar |
| temperature | Optional[float] | scalar (override rollout config) |
| top_p | Optional[float] | scalar (override rollout config) |
| return_logprobs | bool | scalar (request per-token logprobs) |
| extra | Dict[str, Any] | scalar |
| dataset_ids | List[str] | [batch] (tracing) |
| entry_ids | List[int] | [batch] (tracing) |
| rollout_ids | List[int] | [batch] (tracing) |
| subrollout_ids | List[int] | [batch] (tracing) |
| batch_size | len(prompts) | — |

### `RolloutResult` (BatchedMessage)

Sent: **Rollout → Orchestrator**

| Field | Type | Dim |
|---|---|---|
| completions | List[str] | [batch] |
| token_ids | List[List[int]] | [batch, variable] |
| prompt_token_ids | List[List[int]] | [batch, variable] |
| logprobs | List[List[float]] | [batch, variable] (optional, per-token) |
| prompt_logprobs | List[List[float]] | [batch, variable] (optional) |
| finish_reasons | List[str] | [batch] ("length", "stop", "abort", …) |
| step | int | scalar |
| dataset_ids | List[str] | [batch] (tracing) |
| entry_ids | List[int] | [batch] (tracing) |
| rollout_ids | List[int] | [batch] (tracing) |
| subrollout_ids | List[int] | [batch] (tracing) |
| batch_size | len(completions) | — |

### `RewardRequest` (BatchedMessage)

Sent: **Orchestrator → Reward**

| Field | Type | Dim |
|---|---|---|
| prompts | List[str] | [batch] (optional, text-based rewards) |
| completions | List[str] | [batch] |
| responses | List[str] | [batch] (ground-truth answers for accuracy reward) |
| input_ids | List[List[int]] | [batch, variable] (optional) |
| logprobs | List[List[float]] | [batch, variable] (optional, per-token from rollout) |
| ref_logprobs | List[List[float]] | [batch, variable] (optional, per-token from ref model) |
| finish_reasons | List[str] | [batch] |
| step | int | scalar |
| extra | Dict[str, Any] | scalar |
| dataset_ids | List[str] | [batch] (tracing) |
| entry_ids | List[int] | [batch] (tracing) |
| rollout_ids | List[int] | [batch] (tracing) |
| subrollout_ids | List[int] | [batch] (tracing) |
| batch_size | len(completions) | — |

### `RewardResult` (BatchedMessage)

Sent: **Reward → Orchestrator**

| Field | Type | Dim |
|---|---|---|
| rewards | List[float] | [batch] (raw, pre-normalization) |
| advantages | List[float] | [batch] (group z-score normalized, for GRPO loss) |
| step | int | scalar |
| dataset_ids | List[str] | [batch] (tracing) |
| entry_ids | List[int] | [batch] (tracing) |
| rollout_ids | List[int] | [batch] (tracing) |
| subrollout_ids | List[int] | [batch] (tracing) |
| batch_size | len(rewards) | — |

### `TrainStepRequest` (BatchedMessage)

Sent: **Orchestrator → Model**

| Field | Type | Dim |
|---|---|---|
| input_ids | List[List[int]] | [batch, variable] |
| label_ids | List[List[int]] | [batch, variable] |
| rewards | List[float] | [batch] |
| advantages | List[float] | [batch] (GRPO z-score normalized) |
| old_logprobs | List[List[float]] | [batch, variable] (per-token, at generation time) |
| ref_logprobs | List[List[float]] | [batch, variable] (per-token, from ref model) |
| prompts | List[str] | [batch] (optional, debugging) |
| completions | List[str] | [batch] (optional, debugging) |
| step | int | scalar |
| global_batch_size_override | int | scalar (full batch size for DR-GRPO, 0 = use local) |
| global_max_completion_len_override | int | scalar (global max completion len for DR-GRPO, 0 = use local) |
| extra | Dict[str, Any] | scalar |
| dataset_ids | List[str] | [batch] (tracing) |
| entry_ids | List[int] | [batch] (tracing) |
| rollout_ids | List[int] | [batch] (tracing) |
| subrollout_ids | List[int] | [batch] (tracing) |
| batch_size | len(input_ids) | — |

**Note**: `label_ids` uses -100 for the prompt portion (masked from loss) and
completion tokens for the generation portion.

### `TrainStepResult` (BatchedMessage)

Sent: **Model → Orchestrator**

| Field | Type | Dim |
|---|---|---|
| step | int | scalar |
| loss | float | scalar |
| grad_norm | Optional[float] | scalar |
| lr | Optional[float] | scalar |
| total_completions | Optional[int] | scalar (total samples processed) |
| checkpoint_path | Optional[str] | scalar |
| batch_size | 1 (always) | — |

Each FSDP rank produces one result. `join()` delegates to `merge()`:
- Averages `loss` and `lr` across ranks
- Takes `max(grad_norm)` for safety reporting
- Sums `total_completions` across ranks
- Uses rank 0's `checkpoint_path`

### `TopKLogitsRequest` (BatchedMessage)

Sent: **Orchestrator → Model**

| Field | Type | Dim |
|---|---|---|
| input_ids | List[List[int]] | [batch, variable] |
| label_ids | List[List[int]] | [batch, variable] |
| rewards | List[float] | [batch] |
| topk | int | scalar (default: 5) |
| temperature | float | scalar (default: 1.0) |
| indices | List[int] | [batch] |
| step | int | scalar |
| batch_size | len(input_ids) | — |

### `TopKLogitsResult` (BatchedMessage)

Sent: **Model → Orchestrator**

| Field | Type | Dim |
|---|---|---|
| topk_ids | List[List[List[int]]] | [batch, seq_len, topk] |
| topk_logits | List[List[List[float]]] | [batch, seq_len, topk] |
| selected_logits | List[List[float]] | [batch, seq_len] |
| indices | List[int] | [batch] |
| step | int | scalar |
| batch_size | len(topk_ids) | — |

### `WeightSyncRequest` (ArgsKwargs — not batched)

Sent: **Orchestrator → Rollout**

| Keyword | Type |
|---|---|
| checkpoint_path | str |
| step | int |

### `MetricsBatch` (Message — not batched)

Sent: **Any node → Tracker**

| Field | Type |
|---|---|
| metrics | Dict[str, float] |
| node_id | str |

### `ModelUpdateRequest` (ArgsKwargs — not batched)

Sent: **Orchestrator → Model**

| Keyword | Type |
|---|---|
| lr | Optional[float] | per-step learning rate override (orchestrator computes this via warmup+cosine scheduler)

### `FlushMetrics` (ArgsKwargs — not batched)

Sent: **Orchestrator → Tracker** (flush accumulated metrics to W&B)

| Keyword | Type |
|---|---|
| step | int | Global training step to associate with this flush |

### `CheckpointRequest` (ArgsKwargs — not batched)

Sent: **Orchestrator → Model** (save/load DCP model + optimizer shards). The
endpoint name (``save_hot_checkpoint``, ``save_cold_checkpoint``,
``load_hot_checkpoint``) disambiguates semantics.

| Keyword | Type |
|---|---|
| path | str | checkpoint directory path |
| step | int | current training step (for logging) |

### `CheckpointResult` (ArgsKwargs — not batched)

Sent: **Model → Orchestrator**

| Keyword | Type |
|---|---|
| ok | bool | success flag |
| path | str | checkpoint directory |
| step | int | step number |
| error | str | error message (if not ok) |

### `ExportWeightsRequest` (ArgsKwargs — not batched)

Sent: **Orchestrator → Model** (consolidate FSDP shards → HF weights only)

| Keyword | Type |
|---|---|
| path | str | export directory path |

### `NoInputRequest` (ArgsKwargs — not batched)

Sent: **Orchestrator → Model / Rollout** for side-effect-only endpoints
(wake, sleep). Empty message — no args or kwargs.

### `FlushMetrics` (ArgsKwargs — not batched)

Sent: **Orchestrator → Tracker** (flush accumulated metrics to W&B)

| Keyword | Type |
|---|---|
| step | int | Global training step to associate with this flush |

### `StateGetRequest` (Message — not batched)

Sent: **Orchestrator → Any node** (read node state)

| Field | Type |
|---|---|
| keys | List[str] | top-level keys to return (empty = all) |

### `StateSetRequest` (Message — not batched)

Sent: **Orchestrator → Any node** (write node state)

| Field | Type |
|---|---|
| keys | List[str] | top-level keys to apply |
| data | Dict[str, Any] | key → value mapping |

### `StateResult` (Message — not batched)

Sent: **Any node → Orchestrator** (response to get_state / set_state)

| Field | Type |
|---|---|
| data | Dict[str, Any] | requested or applied key-value pairs |
| ok | bool | success flag |
| error | str | error message (if not ok) |

### `BatchRequest` (Message — not batched)

Sent: **Orchestrator → Dataloader** (request next batch)

| Field | Type |
|---|---|
| dataset_name | str | which dataset (e.g. "train", "val") |
| batch_size | int | override (0 = use dataset default) |
| step | int | current training step |

### `BatchResult` (Message — not batched)

Sent: **Dataloader → Orchestrator** (batch response)

| Field | Type |
|---|---|
| ok | bool | success flag |
| error | str | error message (if not ok) |
| prompts | List[str] | [batch] |
| responses | List[str] | [batch] (ground-truth) |
| system_prompt | Optional[str] | system message for chat template |
| step | int | scalar |
| dataset_ids | List[str] | [batch] (tracing) |
| entry_ids | List[int] | [batch] (tracing) |
| rollout_ids | List[int] | [batch] (tracing) |
| subrollout_ids | List[int] | [batch] (tracing) |

### `TableBatch` (Message — not batched)

Sent: **Any node → Tracker** (tabular data for W&B tables)

| Field | Type |
|---|---|
| table_name | str | table key in W&B (e.g. "rollouts") |
| columns | List[str] | column headers |
| data | List[List[str]] | rows |
| node_id | str | source node identifier |
| step | int | training step |

### `OperationResult` (Message — not batched)

Sent: **Model → Orchestrator** (for sleep, wake, save_checkpoint)

| Field | Type |
|---|---|
| ok | bool |
| message | str |

## Legacy Helpers

Module-level convenience functions (backward compatible):

```python
_to_dict(obj) -> Dict[str, Any]     # calls obj.to_dict() for Message, else asdict()
_from_dict(cls, data) -> Self       # calls cls.from_dict(data) for Message, else filtered constructor
```

These are used throughout `training_orch.py` for RPC serialization.

## Dimensionality Table

| Message | Field | Type | Dim |
|---|---|---|---|
| PromptBatch | prompts | List[str] | [batch] |
| PromptBatch | responses | List[str] | [batch] |
| PromptBatch | system_prompt | Optional[str] | scalar |
| PromptBatch | temperature | Optional[float] | scalar |
| PromptBatch | top_p | Optional[float] | scalar |
| PromptBatch | return_logprobs | bool | scalar |
| PromptBatch | entry_ids | List[int] | [batch] |
| RolloutResult | completions | List[str] | [batch] |
| RolloutResult | token_ids | List[List[int]] | [batch, variable] |
| RolloutResult | prompt_token_ids | List[List[int]] | [batch, variable] |
| RolloutResult | logprobs | List[List[float]] | [batch, variable] |
| RolloutResult | finish_reasons | List[str] | [batch] |
| RolloutResult | entry_ids | List[int] | [batch] |
| RewardRequest | completions | List[str] | [batch] |
| RewardRequest | responses | List[str] | [batch] |
| RewardRequest | input_ids | List[List[int]] | [batch, variable] |
| RewardRequest | logprobs | List[List[float]] | [batch, variable] |
| RewardRequest | ref_logprobs | List[List[float]] | [batch, variable] |
| RewardRequest | finish_reasons | List[str] | [batch] |
| RewardRequest | entry_ids | List[int] | [batch] |
| RewardResult | rewards | List[float] | [batch] |
| RewardResult | advantages | List[float] | [batch] |
| RewardResult | entry_ids | List[int] | [batch] |
| TrainStepRequest | input_ids | List[List[int]] | [batch, variable] |
| TrainStepRequest | label_ids | List[List[int]] | [batch, variable] |
| TrainStepRequest | rewards | List[float] | [batch] |
| TrainStepRequest | advantages | List[float] | [batch] |
| TrainStepRequest | old_logprobs | List[List[float]] | [batch, variable] |
| TrainStepRequest | ref_logprobs | List[List[float]] | [batch, variable] |
| TrainStepRequest | entry_ids | List[int] | [batch] |
| TrainStepRequest | global_batch_size_override | int | scalar |
| TrainStepRequest | global_max_completion_len_override | int | scalar |
| TrainStepResult | loss | float | scalar |
| TrainStepResult | grad_norm | Optional[float] | scalar |
| TrainStepResult | lr | Optional[float] | scalar |
| TrainStepResult | total_completions | Optional[int] | scalar |
| TopKLogitsRequest | rewards | List[float] | [batch] |
| TopKLogitsResult | topk_ids | List[List[List[int]]] | [batch, seq, topk] |
| TopKLogitsResult | topk_logits | List[List[List[float]]] | [batch, seq, topk] |
| WeightSyncRequest | (keyword) checkpoint_path | str | scalar |
| ExportWeightsRequest | (keyword) path | str | scalar |
| CheckpointRequest | (keyword) path | str | scalar |
| CheckpointResult | (keyword) ok | bool | scalar |
| NoInputRequest | — | — | empty |
| FlushMetrics | (keyword) step | int | scalar |
| OperationResult | ok | bool | scalar |
| StateGetRequest | keys | List[str] | variable |
| StateSetRequest | keys | List[str] | variable |
| StateSetRequest | data | Dict[str, Any] | scalar |
| StateResult | data | Dict[str, Any] | scalar |
| BatchRequest | dataset_name | str | scalar |
| BatchRequest | batch_size | int | scalar |
| BatchResult | prompts | List[str] | [batch] |
| BatchResult | responses | List[str] | [batch] |
| TableBatch | columns | List[str] | variable |
| TableBatch | data | List[List[str]] | [rows, cols] |
