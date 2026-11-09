# LoRA (PEFT) support for Whirlwind — native adapter reload design

> **Status: implemented and tested.** Native LoRA adapter
> reloading: trainer exports adapter-only, rollout loads it as an adapter via
> the engine's native LoRA API. NOT the TRL merge-and-push-full pattern.
>
> **Validated end-to-end** (SGLang colocate, 5 opt-steps, 0 errors, loss
> 0.808→0.099). See "Test run results" below.
>
> **Known limitation**: vLLM `enable_lora` + `enable_sleep_mode` (cumem
> allocator) are INCOMPATIBLE in vLLM 0.24.0 — the LoRA memory pool is not in
> cumem's tag set, so `sleep()` after `add_lora()` crashes with `CUDART
> illegal memory access` (vllm #39934/#48297). `remove_lora` before sleep does
> NOT help (the pool allocation persists). SGLang is unaffected (uses
> `torch_memory_saver` — pool-level `tms_pause`/`tms_resume`, not cumem's
> allocation-iteration). **Use SGLang for colocate LoRA; vLLM LoRA needs
> non-colocate (no sleep mode).**

## Goal

Add PEFT/LoRA training to the manual FSDP2 updater node, with **native adapter
reloading** at the rollout: the rollout loads the LoRA adapter directly (not a
merged full model). This keeps weight-sync tiny (adapter ≈ MB vs full model ≈
GB) and lets the rollout serve the base + adapter without a full reload.

## Why native adapter reload (not TRL's merge-and-push-full)

TRL's `vllm_generation.sync_weights` does `merge_adapter()` → push full params
→ `unmerge_adapter()`. We instead export adapter-only and load it natively:
- **Smaller sync**: adapter (MB) vs full merged model (GB) written + loaded.
- **No merge under FSDP2** (TRL's own `TODO: does this work with FSDP?` — the
  one open risk in their approach). We sidestep it entirely.
- **Engine-native**: vLLM/SGLang both support dynamic `/load_lora_adapter`.

## Engine APIs (verified in our env: vllm 0.24.0, sglang 0.5.14, peft 0.19.1)

- **vLLM**: `LLMEngine.add_lora(LoRARequest(lora_name, lora_int_id, lora_path))`
  loads; use via `generate(..., lora_request=LoRARequest(...))`. Needs
  `enable_lora=True` (or `max_loras>=1`, `max_lora_rank`) at `LLM()` init.
  Reads standard PEFT dir: `adapter_config.json` + `adapter_model.safetensors`.
- **SGLang**: `engine.load_lora_adapter(LoadLoRAAdapterReqInput(lora_name,
  lora_path, ...))` / `unload_lora_adapter(...)`. Same PEFT dir format.

## Layout — 6 changes, no merge, no second model

### 1. Config (`whirlwind/config.py`) — new fields
```python
peft_lora: bool = False            # enable LoRA training
lora_r: int = 16                   # rank
lora_alpha: int = 32
lora_dropout: float = 0.0
lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj","v_proj"])
lora_bias: str = "none"            # none | all | lora_only
```

### 2. Model node (`model_fsdp2.py:intitialize_model`) — attach adapter
After `from_pretrained`, before `fully_shard`:
```python
self._is_peft = False
if self.config.peft_lora:
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=self.config.lora_r, lora_alpha=self.config.lora_alpha,
                          target_modules=self.config.lora_target_modules,
                          lora_dropout=self.config.lora_dropout, bias=self.config.lora_bias)
    self._model = get_peft_model(self._model, lora_cfg)
    self._is_peft = True
```
Order matches TRL `grpo_trainer.py:366` (get_peft_model before FSDP wrap).

### 3. Model node (`_make_optimizer`) — train adapter only
```python
params = [p for p in self._fsdp.parameters() if p.requires_grad]
return torch.optim.AdamW(params, ...)
```
PEFT sets `requires_grad=False` on the frozen base; the filter is all that's
needed. Adapter params are small → fp32 master is cheap.

### 4. Model node (`_compute_per_token_logps` ref path) — disable_adapter for ref
GRPO needs ref (KL) logps. With LoRA the base model IS the reference (TRL
`grpo_trainer.py:665`: "the reference model is not needed since the adapter can
be disabled"). No second model in memory:
```python
if self._is_peft and ref_needed:
    with self._model.disable_adapter():
        ref_logps = self._compute_per_token_logps(...)  # base only
```
(For an already-trained adapter, TRL uses `add_adapter("ref", ...)` to snapshot
it as the ref; we start from base so `disable_adapter` = base = ref.)

### 5. Model node (`_export_weights`) — adapter-only export
Branch on `self._is_peft`:
```python
if self._is_peft:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    # gather adapter state dict ONLY (ignore_frozen_params=True — pytorch #169448 fix,
    # else FSDP2 hangs on frozen base params)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
    adapter_sd = get_model_state_dict(self._fsdp, options=options)
    if self.rank == 0:
        # write standard PEFT adapter dir the rollout engines read natively
        from peft import PeftModel  # for set_peft_model_state_dict
        os.makedirs(ckpt_dir, exist_ok=True)
        # save adapter_config.json + adapter_model.safetensors
        self._model.save_pretrained(ckpt_dir, state_dict=adapter_sd)  # base_model is the PeftModel
else:
    <existing full-state-dict export>
```
**No merge.** The export is adapter-only (MB). The orchestrator's
`_call_export_weights` + `_call_sync` are unchanged — they already pass the dir
path through. (open question: does `save_pretrained` with a gathered state_dict
work under FSDP2 — validate in smoke test.)

### 6. Rollout nodes — load adapter natively (NOT load_weights)
Both rollout nodes' `_update_weights` branch on whether the dir is an adapter
(check for `adapter_config.json`):
- **vLLM** (`rollout_vllm.py`): init with `enable_lora=True` (new config flag);
  `_update_weights` → `engine.add_lora(LoRARequest(lora_name=f"step{step}",
  lora_int_id=step, lora_path=ckpt_path))`; generate with
  `lora_request=LoRARequest(...)`. Unload prior adapter before adding the new one
  (`engine.remove_lora(prev_id)`) to keep one active adapter.
- **SGLang** (`rollout_sglang.py`): init with `--lora-target-modules` /
  `--max-lora-rank`; `_update_weights` →
  `engine.load_lora_adapter(LoadLoRAAdapterReqInput(lora_name=f"step{step}",
  lora_path=ckpt_path))`; unload prior via `unload_lora_adapter`.

### RPC — unchanged
`ExportWeightsRequest` (path) + `WeightSyncRequest` (checkpoint_path, step)
already carry the dir. The dir now contains an adapter instead of a full model;
the rollout detects which and loads accordingly.

## Open risks to validate in a smoke test (before full build)

1. **FSDP2 sharding of LoRA params**: does `fully_shard` pick up injected
   `lora_A`/`lora_B` and all-gather/reduce-scatter them correctly? (The
   `mixed Tensor/DTensor` error class — TRL #4550, transformers #42417.)
2. **`save_pretrained(state_dict=gathered)` under FSDP2**: gather with
   `ignore_frozen_params=True` then PEFT save — does it produce a loadable
   adapter dir? (pytorch #169448 says gather hangs without the flag.)
3. **`disable_adapter()` under FSDP2**: does the ref-forward (base only) work
   on the sharded model?
4. **vLLM/SGLang adapter swap in colocate sleep/wake**: adapter loaded after
   `wake_up` — does the sleep/wake cycle preserve/evict the adapter correctly?

The smoke test: build model → `get_peft_model` → `fully_shard` → one
forward/backward (train adapter) → `save_pretrained` adapter-only → (separately)
load adapter in a vLLM engine → generate with it. If all four pass, the build is
mechanical against this spec.

## Smoke test results (two GPUs, Qwen3-0.6B-Base)

All 5 risks PASSED:
1. `get_peft_model` + `fully_shard`: OK (no mixed Tensor/DTensor; 0.385%
   trainable = 2.3M/596M params)
2. `save_pretrained` adapter-only under FSDP2 (`ignore_frozen_params=True`):
   OK (valid `adapter_config.json` + `adapter_model.safetensors`, 9.2MB, no
   pytorch #169448 hang)
3. `disable_adapter()` ref forward under FSDP2: OK (GRPO ref-model trick works)
4. vLLM native adapter load + generate: OK (`add_lora` + `lora_request`)
5. SGLang native adapter load: OK (`load_lora_adapter` + `async_generate`)

(standalone smoke scripts, not shipped)

## Test run results (SGLang colocate, 5 opt-steps)

Full production run through the colocate sleep/wake cycle
(`configs/model/fsdp2_base_colocate_lora.yaml` + `configs/rollout/sglang_colocate_tp2_lora.yaml`):
- PEFT attached: 0.385% trainable (2.3M/596M), r=16 alpha=32
- val@0: 0.1260 (base model baseline)
- 5 adapter swaps (opt-steps 1–5), 0 crashes, 0 errors
- SGLang sleep/wake + LoRA: stable (pool-level pause, NOT the vLLM cumem crash)
- Loss: 0.808 → 0.795 → 0.578 → 0.652 → **0.099** (adapter learning + applied
  in generation — the policy improves, rewards rise, loss collapses)
- ~11 min/opt-step (SGLang eager + LoRA overhead; full-FT vLLM was ~2.4 min)

### vLLM vs SGLang for colocate LoRA

| engine | sleep/wake mechanism | LoRA + sleep/wake | verdict |
|--------|---------------------|-------------------|---------|
| vLLM 0.24 | cumem allocator (iterate + copy allocations) | **CRASH** (LoRA pool not in cumem tags, #39934) | non-colocate only |
| SGLang 0.5.14 | torch_memory_saver (pool-level tms_pause/resume) | **WORKS** (pool paused whole, LoRA included) | colocate OK |

The vLLM `_sleep`/`_wake` unload-before-sleep / reload-after-wake workaround
(removing the adapter before sleep) does NOT help — the LoRA memory POOL
(allocated at init) persists after `remove_lora` and still crashes cumem. The
fix is engine-level (vLLM #48297 RFC: "LoRA Adapter Lifecycle for RL Training",
not yet landed). SGLang sidesteps it entirely.
