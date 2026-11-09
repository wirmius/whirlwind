"""Model update node — FSDP2 training with gradient computation and checkpointing.

Exposes (on rank 0 only):
    forward_no_grad     — inference pass, return top-K logits per position
    forward_accum_grad  — forward + backward (accumulates gradients)
    model_update_step   — clip grads, optimizer step, scheduler step, zero_grad
    save_hot_checkpoint — DCP shards (model + optimizer) with wake/sleep
    sleep               — move FSDP shards to CPU (pinned memory)
    wake                — move FSDP shards to GPU (non-blocking)
    ping                — health check

Distributed execution:
    The ``@_fsdp_using_function`` decorator handles splitting batched messages
    across FSDP ranks, scattering shards via ``scatter_object_list``, executing
    the inner core function on each rank, and gathering/joining results.

    Non-lead ranks block in ``_standby()``, receiving work via dist collectives
    and dispatching to registered handlers.

Launch via torchrun:
    torchrun --nproc_per_node=8 -m whirlwind.model_update --config model.yaml

The whirlcoms RPC node is registered on rank 0 so the orchestrator
has a single stable target for the ``model`` service.
"""

import copy
import logging
import os
from datetime import timedelta
import pathlib
import shutil
import threading
from functools import partial
from typing import Any, Callable

import torch
import torch.distributed as dist
from zmq import asyncio

from whirlwind.config import ModelConfig
from whirlwind.model import updater_utils
from whirlwind.messages import (
    BatchedMessage,
    CheckpointRequest,
    CheckpointResult,
    ExportWeightsRequest,  # noqa: F401
    Message,
    ModelUpdateRequest,
    NoInputRequest,
    OperationResult,
    StateGetRequest,
    StateResult,
    StateSetRequest,
    TopKLogitsRequest,
    TopKLogitsResult,
    TrainStepRequest,
    TrainStepResult,
    _to_dict,
    map_topk,
)
from whirlwind.whirlcoms import Node

logger = logging.getLogger(__name__)

# fn_name rank 0 scatters from stop() to end the other ranks' _standby() loop.
_STANDBY_STOP = "__standby_stop__"


def _discover_decoder_layer_cls(model: torch.nn.Module, granularity: str) -> set[type]:
    """The model's transformer-block classes, for per-layer FSDP2 wrapping.

    Structural instance scan: the repeated decoder layer's class name contains
    "DecoderLayer" (LlamaDecoderLayer, Qwen3DecoderLayer, Qwen3_5MoeDecoderLayer,
    Gemma3DecoderLayer, ...). An earlier candidate-list/MRO scan matched
    nothing on transformers 5.x, so models silently ran as a single ROOT-ONLY
    FSDP2 unit: the whole model all-gathered on every forward, and activation
    checkpointing did nothing (it only wraps child FSDP units, never the root).
    Numerics are unaffected either way; memory is not.

    Granularity "layer"/"transformer_block" with no match logs a WARNING (the
    caller keeps the root wrap).
    """
    layer_cls_set: set[type] = set()
    if granularity in ("transformer_block", "layer"):
        for child in model.modules():
            if "DecoderLayer" in type(child).__name__:
                layer_cls_set.add(type(child))
        if not layer_cls_set:
            logger.warning(
                "FSDP2 wrap found NO decoder-layer classes on %s (granularity %r) — "
                "falling back to a ROOT-ONLY unit: the whole model all-gathers every "
                "forward and activation checkpointing no-ops. Check the model's "
                "decoder-layer class name.",
                type(model).__name__,
                granularity,
            )
    return layer_cls_set


def hybrid_shard_mesh_shape(world_size: int, local_world_size: int) -> tuple[int, int]:
    """``(replicate, shard)`` mesh shape for HYBRID_SHARD: one row per node.

    torchrun numbers ranks node by node, so row ``i`` of the row-major mesh
    holds node ``i``'s ranks: shards stay on a node, replicas span nodes.
    """
    if local_world_size < 1 or world_size % local_world_size:
        raise ValueError(
            "HYBRID_SHARD needs the same number of ranks on every node: "
            f"WORLD_SIZE={world_size} is not a multiple of "
            f"LOCAL_WORLD_SIZE={local_world_size}"
        )
    return world_size // local_world_size, local_world_size


def _fsdp_using_function(fn_name: str, *, require_message: bool = True):
    """Decorator factory that distributes a handler across FSDP ranks.

    Usage (outer decorator is the message serdes, inner is FSDP dispatch)::

        @TrainStepRequest.decorate_serdes()
        @_fsdp_using_function("train_step")
        def _train_step(self, req: TrainStepRequest) -> TrainStepResult:
            ...

    On ``world_size == 1``: calls the inner function directly.
    On ``world_size > 1``, rank 0:
        1. Splits the input message via ``split_into(world_size)``
        2. ``scatter_object_list`` the shards to each rank
        3. Calls the inner function on rank 0's shard
        4. ``all_gather_object`` results from all ranks
        5. Joins/gathers results into a single message

    Non-lead ranks receive their shards in ``_standby()`` via the same
    collectives, call the handler from ``_distributed_handlers``, and
    contribute results to ``all_gather_object``.

    Args:
        fn_name: Name used for scatter dispatch on non-lead ranks.
        require_message: If True (default), asserts the first arg is a
            ``Message`` or ``BatchedMessage``. Set to False for handlers
            that operate on plain values (e.g. checkpoint with a path str).
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            assert self._initialized_model

            # Single-rank: just call directly
            if self.world_size == 1:
                return func(self, *args, **kwargs)

            msg = args[0] if args else kwargs.get("data")

            if require_message:
                assert isinstance(msg, BatchedMessage) or isinstance(msg, Message), (
                    f"_fsdp_using_function({fn_name!r}) requires a Message input, "
                    f"got {type(msg).__name__}"
                )

            # ---- Rank 0: distribute work ----
            if self.rank == 0:
                self._fsdp_lock.acquire()
                # Split into shards (one per rank)
                if isinstance(msg, BatchedMessage):
                    shards = msg.split_into(self.world_size)
                else:
                    # Plain value (str, Message, etc.) — replicate to all ranks
                    shards = [msg for _ in range(self.world_size)]

                # Scatter shard tuples: (fn_name, shard) to each rank
                scatter_list = [(fn_name, sh) for sh in shards]
                rank0_recv = [None]
                dist.scatter_object_list(rank0_recv, scatter_list, src=0)

                # Call handler on rank 0's shard. If it raises, still take part in
                # the gather (with an error sentinel) so the standby ranks stay in
                # lockstep, release the lock, THEN re-raise. Without
                # this, peers hung in all_gather_object until the NCCL watchdog
                # and every later RPC blocked on the lock forever.
                _, rank0_shard = rank0_recv[0]  # type: ignore[misc]
                _handler_exc: BaseException | None = None
                try:
                    rank0_result = func(self, rank0_shard)
                except BaseException as exc:
                    _handler_exc = exc
                    rank0_result = OperationResult(
                        ok=False, message=f"{type(exc).__name__}: {exc}"
                    )

                # Gather results from all ranks
                gathered = [None] * self.world_size
                dist.all_gather_object(gathered, rank0_result)  # type: ignore[call-arg]
                self._fsdp_lock.release()
                if _handler_exc is not None:
                    raise _handler_exc

                # Join gathered results
                assert all([isinstance(ga, type(gathered[0])) for ga in gathered]), (
                    f"Must have returns of the same types: {[type(ga) for ga in gathered]}"
                )

                output_msg_type = type(gathered[0])
                if issubclass(output_msg_type, BatchedMessage):
                    return output_msg_type.join(gathered)
                elif issubclass(output_msg_type, Message):
                    ## TODO: assert all the messages are equal
                    return gathered[0]
                else:
                    raise ValueError(f"Unclear nreturn type ({output_msg_type})")
            else:
                # ---- Non-lead rank: receive work ----
                # Just call the function of interest
                # The scattering and gathering is handled in the _standby
                rank0_result = func(self, msg)
                return rank0_result

        return wrapper

    return decorator


def _to_cpu_pinned(val: torch.Tensor) -> torch.Tensor:
    """Move tensor to CPU with pin_memory() if possible.

    DTensors (distributed tensors) don't support pin_memory,
    so we fall back to plain .cpu() for those.
    """
    from torch.distributed._tensor import DTensor  # type: ignore[attr-defined, no-redef]

    if isinstance(val, DTensor):
        return val.cpu()
    return val.cpu().pin_memory()


class ModelUpdateNode:
    """FSDP2 training node.

    Usage:
        node = ModelUpdateNode(ModelConfig(model_path="..."))
        # start() runs on ALL ranks
        await node.start()
        # rank 0 → asyncio event loop; ranks 1+ → _standby()
        await node.stop()
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._node: Node | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._optimizer: Any | None = None
        self._fsdp: Any | None = None

        # Precision flags, set at construction (not in the model-load path) so
        # unit tests that call _compute_loss/_compute_per_token_logps directly
        # on a bare node don't hit AttributeError. All env-gated no-ops unless
        # set; _use_autocast mirrors the FSDP autocast mode (TRL-faithful
        # forward). See the model-load path for the full precision rationale.
        self._use_autocast = bool(getattr(self.config, "fsdp_autocast", False))
        self._logits_fp32 = bool(os.environ.get("WW_LOGITS_FP32"))
        self._is_fp32 = bool(os.environ.get("WW_IS_FP32"))
        self._loss_fp32 = bool(os.environ.get("WW_LOSS_FP32"))
        self._compute_entropy = not bool(os.environ.get("WW_NO_ENTROPY"))

        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Ranks on one node, set by torchrun; sizes the HYBRID_SHARD mesh.
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        # Process group over which each parameter's shards are disjoint: the
        # world for FULL_SHARD (None), one node's ranks for HYBRID_SHARD.
        self._shard_group: Any = None

        # Thread-safe lock protecting all FSDP operations on rank 0.
        # Prevents concurrent RPC handlers from stepping on each other
        # if the node processes multiple requests simultaneously.
        self._fsdp_lock: threading.Lock = threading.Lock()
        if self._fsdp_lock.locked():
            self._fsdp_lock.release()

        # Buffer to accumulate per-step metrics across gradient accumulation
        # cycles.  Populated by _forward_accum_grad, flushed by
        # _model_update_step.  Keyed by step so multi-step accumulation works.
        self._accum_buf: dict = {"losses": [], "samples": [], "diag": []}
        self._last_diag: dict[
            str, float
        ] = {}  # aggregated diagnostics, merged into tracker metrics
        self._initialized_model = False
        self._asleep: bool = False  # tracks sleep/wake state for checkpoint ops
        self._is_peft: bool = False  # True when LoRA adapters are attached (see config.peft_lora)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the underlying RPC node is still running (polled by the ``__main__`` keep-alive loop)."""
        return self._node.is_running if self._node is not None else False
    async def start(self) -> None:
        # ----------------------------------------------------------------
        # 1. Register with whirlcoms (rank 0 only)
        # ----------------------------------------------------------------
        if self.rank == 0:
            self._node = Node(self.config.node_id, auto_process=True)
            self._node.expose("forward_no_grad")(self.forward_no_grad)
            self._node.expose("forward_accum_grad")(self.forward_accum_grad)
            self._node.expose("model_update_step")(self.model_update_step)
            self._node.expose("export_weights")(self.export_weights)
            self._node.expose("sleep")(self.sleep)
            self._node.expose("wake")(self.wake)
            self._node.expose("ping")(self.ping)
            self._node.expose("initialize_model")(self.intitialize_model)
            self._node.expose("get_state")(self._get_state)
            self._node.expose("set_state")(self._set_state)
            self._node.expose("save_hot_checkpoint")(self.save_hot_checkpoint)
            self._node.expose("save_cold_checkpoint")(self.save_cold_checkpoint)
            self._node.expose("load_hot_checkpoint")(self.load_hot_checkpoint)
            await self._node.start()

        if self.world_size > 1:
            # PG watchdog budget: at dp2+8k the lead rank awaits the val
            # rollout RPC (15-20 min single-engine) while peers sit in
            # _standby's scatter_object_list — the NCCL default 600s watchdog
            # kills the peer mid-broadcast. Scale via env for long-context dp2
            # runs.
            _nccl_to = timedelta(
                seconds=int(os.environ.get("WHIRLWIND_NCCL_TIMEOUT", "600"))
            )
            dist.init_process_group(
                "nccl",
                rank=self.rank,
                world_size=self.world_size,
                timeout=_nccl_to,
            )
            torch.cuda.set_device(self.local_rank)

        if self.rank == 0:
            logger.info(
                "Model update node '%s' started (model=%s, rank=0, world_size=%d)",
                self.config.node_id,
                self.config.model_path,
                self.world_size,
            )
        else:
            logger.info(
                "Model update rank %d started (model=%s)",
                self.rank,
                self.config.model_path,
            )

    def intitialize_model(self, data) -> None:
        """Initialize distributed group, model, tokenizer, optimizer, FSDP wrapper.

        Called on ALL ranks. Only rank 0 registers the whirlcoms RPC node.
        """
        import transformers  # type: ignore[import-not-found]
        from torch.distributed.device_mesh import init_device_mesh  # type: ignore[import-not-found]
        from torch.distributed.fsdp import (  # type: ignore[import-not-found]
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            OffloadPolicy,
            fully_shard,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        assert not self._initialized_model
        # ----------------------------------------------------------------
        # 1. Distributed init (all ranks)
        # ----------------------------------------------------------------
        logger.debug(f"Waiting for initilzation permission: {self.local_rank}.")
        if self.world_size > 1:
            # make sure everyone is at the table
            dist.barrier()

        # ----------------------------------------------------------------
        # 2. Load model & tokenizer (all ranks)
        # ----------------------------------------------------------------
        # Optional NaN localization: WW_DETECT_ANOMALY=1 wraps every op in
        # an anomaly-checking mode that pinpoints the exact backward
        # function producing the first NaN/inf gradient (slow: 2-3x).
        # Diagnostic only — used to chase an H100 SXM (sm_90, driver
        # 610.57.04) step-2 backward NaN.
        if os.environ.get("WW_DETECT_ANOMALY"):
            torch.autograd.set_detect_anomaly(True)
            logger.warning("detect_anomaly ENABLED (WW_DETECT_ANOMALY set)")
        # H100 SXM (sm_90, driver 610.57.04 + cuDNN 9.19): the cuDNN
        # fused-attention BACKWARD returns NaN (forward is fine) — localized
        # via detect_anomaly: "Function 'ScaledDotProductCudnnAttentionBackward0'
        # returned nan values in its 0th output" at the first post-update
        # step, while a different GPU class never hit it.
        # WW_SDP_DISABLE_CUDNN=1 drops that backend → SDPA falls back to the
        # native flash kernel. WW_ATTN_IMPL=eager bypasses SDPA entirely.
        if os.environ.get("WW_SDP_DISABLE_CUDNN"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            logger.warning("cuDNN SDPA backend disabled (WW_SDP_DISABLE_CUDNN)")
        dtype_map = {
            "bfloat16": "bfloat16",
            "float32": "float32",
            "float16": "float16",
        }
        torch_dtype_str = dtype_map.get(self.config.torch_dtype, "bfloat16")
        torch_dtype = getattr(torch, torch_dtype_str)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            **self.config.model_init_kwargs,
        )
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore[attr-defined]

        # Resolve attention implementation: flash > sdpa > eager
        # (WW_ATTN_IMPL overrides — e.g. 'eager' to dodge broken SDPA
        # backends on a given machine.)
        attn_impl: str = os.environ.get("WW_ATTN_IMPL", "")
        if not attn_impl:
            attn_impl = "eager"
            try:
                if transformers.is_flash_attn_2_available():  # type: ignore[attr-defined]
                    attn_impl = "flash_attention_2"
                else:
                    attn_impl = "sdpa"
            except AttributeError:
                attn_impl = "sdpa"

        # One loading seam: FP8 checkpoints are dequantized at load, composite
        # VL-MoE checkpoints load their text backbone, everything else goes
        # through AutoModelForCausalLM (whirlwind/model/model_loading.py).
        from whirlwind.model.model_loading import load_causal_lm

        self._model = load_causal_lm(
            self.config.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            model_init_kwargs=self.config.model_init_kwargs,
        )

        # Disable dropout (recommended for RLHF — prevents noisy logprobs)
        if self.config.disable_dropout:
            for attr in (
                "dropout",
                "attention_dropout",
                "hidden_dropout",
                "emb_dropout",
            ):
                if hasattr(self._model.config, attr):
                    setattr(self._model.config, attr, 0.0)

        # Cache the HF config once (in memory) for reuse in _export_weights.
        # _export_weights previously called AutoConfig.from_pretrained() on
        # EVERY opt-step to write config.json into the checkpoint dir — that
        # hit the HF Hub (a HEAD request per step, ~1000 over a run) even
        # though the result never changes. The config is identical to the one
        # the model was initialized with (dropout disabled above), so we save
        # this cached copy instead. Captured before FSDP wrap for a stable
        # reference. (vLLM's load_weights doesn't need config.json at all;
        # SGLang's update_weights_from_disk does, so we still write it.)
        self._hf_config = copy.deepcopy(self._model.config)

        # The trainer never generates: run every forward without a KV cache.
        # With activation checkpointing a cache is actively harmful — the
        # backward recompute of each decoder layer appends its keys to the
        # forward's cache a second time and attention fails on the doubled
        # key length. (The export copy above keeps the original setting.)
        if hasattr(self._model.config, "use_cache"):
            self._model.config.use_cache = False

        # Print model info before FSDP wrap (after wrap, params are DTensors)
        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Model loaded: %s (%.1fB params, attn=%s, dtype=%s)",
            type(self._model).__name__,
            total_params / 1e9,
            getattr(self._model.config, "_attn_implementation", "unknown"),
            self._model.dtype,
        )

        # ----------------------------------------------------------------
        # 2b. (optional) PEFT / LoRA — attach adapters over a frozen base.
        # Gated import: peft is only needed when peft_lora=True. The node must
        # NOT import peft at module load (so non-LoRA runs work without peft
        # installed). Errors are raised here (clear, at init) rather than at
        # import time. See docs/lora_design.md.
        # ----------------------------------------------------------------
        if self.config.peft_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "peft_lora=True but the 'peft' package is not installed. "
                    "Install it (pip install peft) or set peft_lora=false."
                ) from exc

            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.lora_bias,
            )
            self._model = get_peft_model(self._model, lora_cfg)
            self._is_peft = True
            n_train = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            logger.info(
                "PEFT/LoRA attached: %d trainable params (%.3f%% of %d), "
                "r=%d alpha=%d targets=%s",
                n_train, 100.0 * n_train / max(total_params, 1), total_params,
                self.config.lora_r, self.config.lora_alpha,
                self.config.lora_target_modules,
            )

        # ----------------------------------------------------------------
        # 3. FSDP2 wrap (all ranks)
        # ----------------------------------------------------------------

        # 3a. Device mesh. FULL_SHARD: one dimension, every rank holds a shard
        # of every parameter. HYBRID_SHARD: one row per node — shard within a
        # node, replicate across nodes — so parameter all-gathers and gradient
        # reduce-scatters stay on the node and only one gradient all-reduce per
        # step crosses the inter-node link.
        strategy = str(self.config.fsdp_sharding_strategy).upper()
        if strategy == "FULL_SHARD":
            mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(self.world_size,),
                mesh_dim_names=("fsdp",),
            )
        elif strategy == "HYBRID_SHARD":
            mesh_shape = hybrid_shard_mesh_shape(self.world_size, self.local_world_size)
            mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=mesh_shape,
                mesh_dim_names=("replicate", "shard"),
            )
            self._shard_group = mesh.get_group("shard")
            if self.rank == 0:
                logger.info(
                    "HYBRID_SHARD mesh: %d node(s) x %d ranks (replicate x shard)%s",
                    *mesh_shape,
                    " — one node, same as FULL_SHARD" if mesh_shape[0] == 1 else "",
                )
        else:
            raise ValueError(
                f"Unsupported fsdp_sharding_strategy: {self.config.fsdp_sharding_strategy!r}. "
                "Use FULL_SHARD or HYBRID_SHARD."
            )
        self._mesh = mesh  # stored for chunked tied-embedding DTensor conversion

        # 3b. Mixed precision policy (FSDP2 feature).
        # CRITICAL precision semantics: FSDP2 keeps the sharded parameter in
        # its ORIGINAL (load-time) dtype and uses that for the optimizer step;
        # `param_dtype` only casts the UNSHARDed (all-gathered) param for
        # forward/backward COMPUTE (see _fsdp_api.py MixedPrecisionPolicy:
        # "The optimizer step uses the sharded parameter in the original dtype").
        # So the master/optimizer dtype == torch_dtype (the load dtype), and the
        # compute dtype == param_dtype. To match TRL (bf16=True -> bf16 compute
        # with an FP32 master), load in float32 (torch_dtype="float32") and set
        # param_dtype/output_dtype=bf16 here. Loading in bfloat16 makes the
        # MASTER bf16, and per-step AdamW updates (~1e-5) fall 12x below the BF16
        # ULP (~1.2e-4) -> round to 0 -> weights freeze -> flat learning. This
        # was the flat-reward root cause. See docs/bf16_master_weight.md.
        #
        # fsdp_autocast (TRL-faithful mode): instead of the FSDP param_dtype
        # cast (whole forward bf16, bf16 logits), keep params fp32 and run the
        # forward under torch.autocast(bf16) with logits cast to fp32 -- exactly
        # what HF Trainer's bf16=True does (accelerator.py:1807-1818 wraps
        # forward in autocast + convert_outputs_to_fp32; FSDP keeps fp32 params).
        # Under autocast, matmuls/linear run bf16 but softmax/lm_head stay fp32
        # (autocast op-allowlist), so the log-softmax gradient is fp32. The two
        # forward call sites (_compute_per_token_logps, forward_no_grad) apply
        # the autocast context + fp32 logit cast directly.
        # NOTE: _use_autocast and the env-gated precision-remedy flags
        # (_logits_fp32/_is_fp32/_loss_fp32/_compute_entropy) are initialized
        # in __init__ (kept there so bare-node unit tests work); see __init__.
        #
        # WW_NO_ENTROPY: skip the per-token entropy diagnostic. Entropy is only
        # a LOGGED metric (it enters the loss only via kl_coef>0; WW runs
        # kl_coef=0). Its computation materializes a full fp32 softmax over the
        # 151k vocab (~80 GB at a 128-completion/rank full-batch forward), which
        # is the dominant memory cost and the OOM trigger at micro_batch_size=-1
        # (no microbatching). Disabling it is loss-neutral when kl_coef=0 and
        # frees the memory mb=-1 needs. No-op in production unless set.
        self._compute_entropy = not bool(os.environ.get("WW_NO_ENTROPY"))
        # Gradient reduce dtype (config-driven for the manual-vs-HF A/B).
        # HF/Accelerate sets reduce_dtype=bf16 via set_mixed_precision('bf16');
        # the manual node historically hard-coded fp32. Default fp32 preserves
        # the manual behavior; set fsdp_grad_reduce_dtype="bfloat16" to align.
        grad_reduce_dtype = getattr(
            torch, getattr(self.config, "fsdp_grad_reduce_dtype", "float32"), torch.float32
        )
        if self.config.fsdp_mixed_precision and not self._use_autocast:
            compute_dtype = torch.bfloat16
            if torch_dtype == torch.bfloat16:
                if self._is_peft:
                    # peft creates the adapters in float32 whatever the base
                    # dtype, and FSDP2 keeps each sharded master in its load
                    # dtype: a frozen bf16 base with fp32 trainable adapters
                    # keeps the fp32-master rule for everything that trains.
                    logger.info(
                        "fsdp_mixed_precision=true with a bf16 base under LoRA: "
                        "frozen base bf16, trainable adapters fp32."
                    )
                else:
                    logger.warning(
                        "fsdp_mixed_precision=true with torch_dtype=bfloat16: the "
                        "optimizer MASTER is bf16 (sub-ULP updates round to 0 -> "
                        "frozen weights). Set torch_dtype='float32' for an FP32 "
                        "master + bf16 compute (TRL-matching)."
                    )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=compute_dtype,
                reduce_dtype=grad_reduce_dtype,
                output_dtype=compute_dtype,
            )
        elif self._use_autocast:
            # TRL-faithful: NO FSDP param/output cast (params stay fp32);
            # reduce_dtype=fp32 keeps all-reduce in fp32. bf16 compute happens
            # via torch.autocast at the forward call sites, not via param_dtype.
            mp_policy = MixedPrecisionPolicy(
                param_dtype=None,
                reduce_dtype=grad_reduce_dtype,
                output_dtype=None,
            )
            logger.info(
                "fsdp_autocast=true: FSDP params kept in torch_dtype (%s); "
                "forward runs under torch.autocast(bf16) with fp32 logits "
                "(TRL bf16=True semantics).",
                torch_dtype_str,
            )
        else:
            mp_policy = MixedPrecisionPolicy()

        # 3c. Offload policy (CPUOffloadPolicy for FSDP2)
        #     Use OffloadPolicy() (base class) for no offloading
        offload_policy = (
            CPUOffloadPolicy(pin_memory=True) if self.config.fsdp_cpu_offload else OffloadPolicy()
        )

        # 3d. Discover transformer block classes for per-layer sharding
        #     (structural instance scan; fail-loud on no match — see
        #     _discover_decoder_layer_cls).
        layer_cls_set = _discover_decoder_layer_cls(
            self._model, self.config.fsdp_wrap_granularity
        )
        ### NOTE: The embedding / lm_head layers are NOT sharded so far! (might be handy in some situations)

        # 3e. Fully shard each transformer block, then the root module
        #     FSDP2 requires bottom-up wrapping (children before parent)
        n_layer_units = 0
        for child in self._model.modules():
            if layer_cls_set and isinstance(child, tuple(layer_cls_set)):
                fully_shard(
                    child,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=self.config.fsdp_reshard_after_forward,
                    offload_policy=offload_policy,
                )
                n_layer_units += 1
                logger.debug("FSDP2 wrapped: %s", type(child).__name__)
        logger.info(
            "FSDP2 wrap: %d decoder-layer unit(s) + root (%s)",
            n_layer_units,
            "per-layer sharding, activation checkpointing active"
            if n_layer_units
            else "ROOT-ONLY: whole-model all-gather, activation checkpointing no-op",
        )

        # Root wrap reshard: manual default False (unsharded for backward);
        # HF/Accelerate + TRL use True. Config-driven for the A/B.
        root_reshard = bool(getattr(self.config, "fsdp_root_reshard_after_forward", False))

        # Chunked lm_head forward (logits_chunk_size > 0): avoid materializing
        # the full [B, C, V] logits by running the lm_head in token chunks.
        #
        # UNTIED embeddings: wrap body + lm_head as separate FSDP2 units (each
        # all-gathers on its own forward) + root (state API). The body forward
        # → hidden → chunked lm_head, each through its own FSDP forward.
        #
        # TIED embeddings (tie_word_embeddings=True, e.g. Qwen3): lm_head.weight
        # IS embed_tokens.weight — CANNOT be in separate FSDP2 units. Keep the
        # standard root wrap (no split). The chunked forward monkey-patches the
        # lm_head to chunk + selective_log_softmax INSIDE the root's FSDP
        # forward (the only context where the DTensor weight works), so the
        # per-token logps are computed without materializing [B, C, V] logits.
        self._chunked = bool(getattr(self.config, "logits_chunk_size", 0)) > 0
        self._body_module = getattr(self._model, "model", None)
        self._lm_head_module = getattr(self._model, "lm_head", None)
        self._chunked_tied = False
        if self._chunked and self._body_module is not None and self._lm_head_module is not None:
            tied = bool(getattr(self._model.config, "tie_word_embeddings", False))
            self._chunked_tied = tied
            if not tied:
                # Untied: body + lm_head as separate FSDP2 units + root
                fully_shard(
                    self._body_module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=self.config.fsdp_reshard_after_forward,
                    offload_policy=offload_policy,
                )
                fully_shard(
                    self._lm_head_module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=self.config.fsdp_reshard_after_forward,
                    offload_policy=offload_policy,
                )
            # Root wrap (always): for untied it's an empty unit (state API);
            # for tied it's the STANDARD wrap owning all params (no split).
            fully_shard(
                self._model,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=root_reshard,
                offload_policy=offload_policy,
            )
            logger.info(
                "FSDP2 chunked lm_head mode: %s (logits_chunk_size=%d, tied=%s)",
                "root wrap (tied embeddings, monkey-patch chunked forward)" if tied
                else "body + lm_head + root wrap (untied)",
                self.config.logits_chunk_size, tied,
            )
        else:
            self._chunked = False
            fully_shard(
                self._model,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=root_reshard,  # root module
                offload_policy=offload_policy,
            )
        self._fsdp = self._model

        # 3f. Activation checkpointing
        if self.config.fsdp_activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # type: ignore[import-not-found]
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
            from torch.distributed.fsdp import FSDPModule  # type: ignore[import-not-found]

            non_reentrant_wrapper = partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )

            def _check_fn(submodule):
                return isinstance(submodule, FSDPModule)

            apply_activation_checkpointing(
                self._fsdp,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=_check_fn,
            )

        # 3g. torch.compile (after FSDP wrap + AC, before optimizer)
        if self.config.torch_compile:
            logger.info(
                "Compiling model (backend=%s, mode=%s)",
                self.config.torch_compile_backend,
                self.config.torch_compile_mode,
            )
            self._fsdp = torch.compile(
                self._fsdp,
                backend=self.config.torch_compile_backend,
                mode=self.config.torch_compile_mode,
            )

        # ----------------------------------------------------------------
        # 4. Optimizer (all ranks)
        # ----------------------------------------------------------------
        self._optimizer = self._make_optimizer()
        self._cpu_adamw_state: dict = {}  # fp32 state for _cpu_adamw_fp32_step

        # ----------------------------------------------------------------
        # 5. Checkpoint dir (all ranks)
        # ----------------------------------------------------------------
        pathlib.Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.debug("Initializing the model complete.")
        self._initialized_model = True

        return {}

    async def stop(self) -> None:
        if self._node:
            await self._node.stop()
        if self.world_size > 1:
            if self.rank == 0:
                self._release_standby_ranks()
            dist.destroy_process_group()
        logger.info("Model update node stopped")

    def _release_standby_ranks(self, lock_timeout_s: float = 30.0) -> None:
        """Send the stop message the other ranks wait for in ``_standby()``.

        Without it they stay blocked in ``scatter_object_list`` after rank 0
        destroys the process group: torchrun outlives the finished run on the
        head node, and ranks on other nodes die with an NCCL remote error
        (exit 1) once the head's processes are gone.
        """
        if not self._initialized_model:  # the other ranks never reached standby
            return
        if not self._fsdp_lock.acquire(timeout=lock_timeout_s):
            logger.warning("An operation still holds the FSDP lock; standby ranks not released")
            return
        try:
            stop = [(_STANDBY_STOP, None) for _ in range(self.world_size)]
            dist.scatter_object_list([None], stop, src=0)
        except Exception:
            logger.exception("Releasing the standby ranks failed")
        finally:
            self._fsdp_lock.release()

    # ------------------------------------------------------------------
    # Non-lead rank standby loop
    # ------------------------------------------------------------------

    def _standby(self) -> None:
        """Blocking work loop for non-lead FSDP ranks.

        Receives work signals and data shards from rank 0 via dist collectives,
        dispatches to registered handlers, and contributes results back via
        ``all_gather_object``. Loops until rank 0 sends the stop message from
        ``stop()``, or until a collective raises ``RuntimeError`` (the process
        group is gone).
        """
        while True:
            try:
                # Try to receive a broadcast message first (non-batched ops)
                # We use scatter_object_list for batched ops and
                # broadcast_object_list for non-batched ops.
                # The fn_name tells us which mode was used.

                # Receive the scattered shard tuples: (fn_name, shard) to each rank
                scatter_list = [None for _ in range(self.world_size)]
                recvd_item = [None]
                dist.scatter_object_list(recvd_item, scatter_list, src=0)

                # Call handler on rank 0's shard
                fn_name, current_shard = recvd_item[0]  # type: ignore[misc]
                if fn_name == _STANDBY_STOP:
                    logger.info("Model update rank %d: stop from rank 0, leaving standby", self.rank)
                    dist.destroy_process_group()
                    break

                # Dispatch to registered handler
                handler = getattr(self, fn_name, None)
                if handler is None:
                    raise KeyError(f"Unknown distributed handler: {fn_name}")

                try:
                    current_result = handler(current_shard)
                except Exception as exc:
                    # Stay in lockstep with rank 0's all_gather instead of dying.
                    # (CUDA OOM is a RuntimeError the outer handler would mistake
                    # for a destroyed process group and exit quietly.) Rank 0 sees
                    # the sentinel and reports the error.
                    logger.exception("Rank %d: handler %s failed", self.rank, fn_name)
                    current_result = OperationResult(
                        ok=False, message=f"{type(exc).__name__}: {exc}"
                    )

                # Gather results from all ranks
                gathered = [None] * self.world_size
                dist.all_gather_object(gathered, current_result)  # type: ignore[call-arg]

                # done, rank 0 will now get the result, we wait for the next operation
            except RuntimeError:
                # dist.destroy_process_group() was called — exit cleanly
                logger.info("Model update rank %d exiting standby", self.rank)
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_and_stack(
        self,
        input_ids: list[list[int]],
        label_ids: list[list[int]],
        advantages: list[float] | list[list[float]] | None,
        sampling_logprobs: list[list[float]] | None,
        ref_logprobs: list[list[float]] | None,
        old_logprobs: list[list[float]] | None = None,
        mask_ids: int = -100,
    ) -> dict[str, torch.Tensor | None]:
        """TRL-aligned tensor assembly (grpo_trainer.py:1873-1931).

        Delegates to :func:`whirlwind.model.updater_utils.pad_and_stack`. Splits
        pre-joined input_ids/label_ids at the -100 boundary, LEFT-pads prompts,
        RIGHT-pads completions, concats -> prompt_completion_ids, and right-pads
        engine logprobs/advantages to C_max (completion-aligned by construction).
        """
        device = torch.device("cuda", self.local_rank)
        pad_token_id = getattr(self._tokenizer, "pad_token_id", 0)  # type: ignore[union-attr]
        return updater_utils.pad_and_stack(
            input_ids,
            label_ids,
            advantages,
            sampling_logprobs,
            ref_logprobs,
            old_logprobs,
            device=device,
            pad_token_id=pad_token_id,
            warn_key=id(self),
            mask_ids=mask_ids,
        )


    def _pad_full_sequence(
        self,
        input_ids: list[list[int]],
        label_ids: list[list[int]],
        mask_ids: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad pre-joined prompt+completion sequences (eval / full-seq path).

        Delegates to :func:`whirlwind.model.updater_utils.pad_full_sequence`.
        Used by ``_forward_no_grad`` (full-seq top-K forward, no attention_mask).
        """
        device = torch.device("cuda", self.local_rank)
        pad_token_id = getattr(self._tokenizer, "pad_token_id", 0)  # type: ignore[union-attr]
        return updater_utils.pad_full_sequence(
            input_ids, label_ids, device=device, pad_token_id=pad_token_id, mask_ids=mask_ids
        )


    def _compute_loss(
        self,
        per_token_logps: torch.Tensor,
        old_per_token_logps: torch.Tensor | None,
        ref_per_token_logps: torch.Tensor | None,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        global_batch_size_override: int = 0,
        global_max_completion_len_override: int = 0,
        completion_len: torch.Tensor | None = None,
        prompt_len: torch.Tensor | None = None,
        total_tokens: int = 0,
        sampling_per_token_logps: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the GRPO training loss (TRL-style PPO clipped surrogate).

        Delegates the pure loss math (PPO surrogate + engine IS + KL + reduction)
        to :func:`whirlwind.model.updater_utils.compute_grpo_loss_core` and keeps
        only the diagnostics plumbing (``_accum_buf`` buffering, the
        ``WW_IS_DIAG`` intermediate-tensor dump, and the ``WW_IS_DECOMP`` capture)
        here -- those are infrastructure-coupled and not part of the shared math.
        See ``updater_utils.compute_grpo_loss_core`` for the formula and the
        ``loss_reduction`` variants (grpo / dr_grpo / bnpo / ce).
        """
        loss_reduction = self.config.loss_reduction
        loss, inter = updater_utils.compute_grpo_loss_core(
            per_token_logps,
            old_per_token_logps,
            ref_per_token_logps,
            advantages,
            mask,
            kl_coef=self.config.kl_coef,
            clip_low=self.config.clip_range_low,
            clip_high=self.config.clip_range_high,
            loss_reduction=loss_reduction,
            importance_sampling_level=self.config.importance_sampling_level,
            engine_is_correction=getattr(self.config, "engine_is_correction", False),
            engine_is_mode=self.config.engine_is_mode,
            engine_is_cap=self.config.engine_is_cap,
            max_allowed_len=self.config.max_completion_length,
            global_batch_size_override=global_batch_size_override,
            completion_len=completion_len,
            prompt_len=prompt_len,
            total_tokens=total_tokens,
            sampling_per_token_logps=sampling_per_token_logps,
            entropy=entropy,
            loss_fp32=self._loss_fp32,
            is_fp32=self._is_fp32,
        )
        # Unpack the intermediates the diagnostics/dump/capture below read.
        per_token_logps = inter["per_token_logps"]
        old_per_token_logps = inter["old_per_token_logps"]
        sampling_per_token_logps = inter["sampling_per_token_logps"]
        ref_per_token_logps = inter["ref_per_token_logps"]
        advantages = inter["advantages"]
        mask = inter["mask"]
        entropy = inter["entropy"]
        log_ratio = inter["log_ratio"]
        coef_1 = inter["coef_1"]
        coef_2 = inter["coef_2"]
        per_token_loss1 = inter["per_token_loss1"]
        per_token_loss2 = inter["per_token_loss2"]
        _loss_pre_is = inter["per_token_loss_pre_is"]
        _is_logps_diff = inter["is_logps_diff"]
        _is_ratio_uncapped = inter["is_ratio_uncapped"]
        importance_sampling_ratio = inter["importance_sampling_ratio"]
        _loss_post_is = inter["per_token_loss_post_is"]
        per_token_kl = inter["per_token_kl"]
        per_token_loss = inter["per_token_loss_final"]
        masked_loss = inter["masked_loss"]
        per_seq_loss = inter["per_seq_loss"]
        batch_size = inter["batch_size"]
        clip_low = self.config.clip_range_low
        clip_high = self.config.clip_range_high

        # ------------------------------------------------------------------
        # Diagnostics (mirror TRL ``grpo_trainer.py`` ``self._metrics`` keys).
        # Computed per-microbatch here, buffered in ``_accum_buf["diag"]``,
        # then averaged across microbatches + ranks in ``_model_update_step``
        # and sent to the tracker alongside train_loss/grad_norm/lr. Skipped for
        # the "ce" SFT path (no PPO surrogate / IS ratio).
        # TRL keys covered: clip_ratio/{low_mean,high_mean,region_mean},
        # sampling/importance_sampling_ratio/{min,mean,max},
        # sampling/sampling_logp_difference/{mean,max}, kl, entropy.
        # (clip_ratio/low_min + high_max need min/max rank aggregation — follow-up.)
        # ------------------------------------------------------------------
        if self.config.loss_reduction != "ce" and hasattr(self, "_accum_buf"):
            with torch.no_grad():

                def _masked_mean(x: torch.Tensor) -> torch.Tensor:
                    # TRL masked_batch_mean: sequence-level (x.shape[1]==1) →
                    # mean over batch; token-level → token-weighted mean.
                    if x.shape[1] == 1:
                        return x.mean()
                    return (x * mask).sum() / mask.sum().clamp(min=1.0)

                diag: dict[str, float] = {}
                is_low = (coef_1 < 1.0 - clip_low) & (advantages < 0)
                is_high = (coef_1 > 1.0 + clip_high) & (advantages > 0)
                diag["clip_ratio/low_mean"] = _masked_mean(is_low.float()).item()
                diag["clip_ratio/high_mean"] = _masked_mean(is_high.float()).item()
                diag["clip_ratio/region_mean"] = _masked_mean((is_low | is_high).float()).item()
                # Model-IS (PPO ratio coef_1 = exp(current-old)) masked mean + how
                # far it sits from 1.0. On-policy (GAS=1/num_iter=1) this is
                # ~1.0 exactly (old==current.detach()); only moves under policy
                # drift (num_iter>1). Lets the step-decomposition print show the
                # model-IS side alongside the engine-IS side.
                diag["model_is/coef_1_mean"] = _masked_mean(coef_1).item()
                diag["model_is/coef_1_abs_dev_from_1"] = _masked_mean((coef_1 - 1.0).abs()).item()
                # Core PPO loss BEFORE the engine-IS multiplier (the part that
                # actually carries gradient). Contrasts with the post-IS loss
                # (scaled by the detached engine ratio) to expose the multiplier.
                if _loss_pre_is is not None:
                    diag["loss/pre_is_mean"] = _masked_mean(_loss_pre_is).item()

                if importance_sampling_ratio is not None:
                    seq_level = self.config.engine_is_mode in ("sequence_mask", "sequence_truncate")
                    flat = (
                        importance_sampling_ratio.flatten()
                        if seq_level
                        else importance_sampling_ratio[mask.bool()]
                    )
                    # Always emit min/mean/max (sentinels for empty shards) so the
                    # key set is identical on every rank — keeps the cross-rank
                    # all_reduce aligned (no desync/hang). Sentinels aggregate
                    # correctly: min=+inf is dropped by min-of-mins, max=-inf by
                    # max-of-maxes, and collapsed to 0.0 at the end if all-empty.
                    if flat.numel() > 0:
                        diag["sampling/importance_sampling_ratio/min"] = flat.min().item()
                        diag["sampling/importance_sampling_ratio/mean"] = flat.mean().item()
                        diag["sampling/importance_sampling_ratio/max"] = flat.max().item()
                    else:
                        diag["sampling/importance_sampling_ratio/min"] = float("inf")
                        diag["sampling/importance_sampling_ratio/mean"] = 0.0
                        diag["sampling/importance_sampling_ratio/max"] = float("-inf")

                    # ---- Over-cap diagnostics (IS-ratio clipping) ----
                    # Capture the UNcapped ratio distribution (before the cap
                    # masks/clamps it to 0/cap) and the worst-offending sequences
                    # so we can see exactly what the cap is discarding. Runs in
                    # both the clipped and unclipped configs (when unclipped,
                    # uncapped == capped so the over-cap frac is 0).
                    _is_cap = self.config.engine_is_cap
                    if (
                        _is_cap is not None
                        and _is_cap > 0
                        and sampling_per_token_logps is not None
                        and old_per_token_logps is not None
                    ):
                        _pt_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                        if seq_level:
                            _seq_sums = _pt_diff.sum(dim=-1)  # [batch]
                            _uncapped = torch.exp(_seq_sums)
                        else:
                            _uncapped = torch.exp(_pt_diff[mask.bool()])
                        if _uncapped.numel() > 0:
                            diag["sampling/importance_sampling_ratio/uncapped_mean"] = _uncapped.mean().item()
                            diag["sampling/importance_sampling_ratio/uncapped_min"] = _uncapped.min().item()
                            diag["sampling/importance_sampling_ratio/uncapped_max"] = _uncapped.max().item()
                            diag["sampling/importance_sampling_ratio/over_cap_frac"] = (
                                (_uncapped > _is_cap).float().mean().item()
                            )
                            # Worst offenders (seq mode only): log the entry index,
                            # seq_sum, uncapped ratio, and a per-token logprob sample
                            # for the sequences with the largest |seq_sum|. These
                            # are the entries the cap is killing (or, with clipping
                            # disabled, the entries that now pass through unbounded).
                            if seq_level:
                                _seq_lens = mask.sum(dim=-1)  # [batch]
                                _valid = _seq_lens > 0
                                if _valid.any():
                                    _vs = _seq_sums[_valid]
                                    _vl = _seq_lens[_valid]
                                    _vu = _uncapped[_valid]
                                    # top-3 by absolute seq_sum (the drivers of the
                                    # collapsed/inflated ratio)
                                    _order = _vs.abs().argsort(descending=True)[:3]
                                    _entries = []
                                    for _i in _order.tolist():
                                        _entries.append(
                                            f"seq_sum={_vs[_i].item():+.3f} "
                                            f"len={int(_vl[_i].item())} "
                                            f"uncapped_ratio={_vu[_i].item():.4g}"
                                        )
                                    diag["sampling/importance_sampling_ratio/worst_offenders"] = " | ".join(_entries)

                if sampling_per_token_logps is not None and old_per_token_logps is not None:
                    delta = (old_per_token_logps - sampling_per_token_logps).abs()
                    delta_masked = delta[mask.bool()]
                    if delta_masked.numel() > 0:
                        diag["sampling/sampling_logp_difference/mean"] = delta_masked.mean().item()
                        diag["sampling/sampling_logp_difference/max"] = delta_masked.max().item()
                    else:
                        diag["sampling/sampling_logp_difference/mean"] = 0.0
                        diag["sampling/sampling_logp_difference/max"] = float("-inf")

                    # Signed-diff diagnostic (read-only — does not change the loss).
                    # Reveals whether FSDP systematically over- or under-estimates
                    # logprobs vs the engine, and how the bias aggregates at the
                    # sequence level (which is what sequence_mask exponentiates).
                    # A stable non-zero signed_mean => systematic bias (the IS-cap
                    # disease); a signed_mean near 0 with abs_mean large => noise.
                    _signed = (old_per_token_logps - sampling_per_token_logps)[mask.bool()]
                    if _signed.numel() > 0:
                        diag["sampling/sampling_logp_difference/signed_mean"] = _signed.mean().item()
                        diag["sampling/sampling_logp_difference/positive_frac"] = (
                            (_signed > 0).float().mean().item()
                        )
                        # Per-sequence sum (what exp() is applied to in sequence_mask)
                        _seq = (
                            (old_per_token_logps - sampling_per_token_logps) * mask
                        ).sum(dim=-1)
                        _seq = _seq[mask.sum(dim=-1) > 0]
                        if _seq.numel() > 0:
                            diag["sampling/sampling_logp_difference/seq_sum_mean"] = _seq.mean().item()
                            diag["sampling/sampling_logp_difference/seq_sum_max"] = _seq.max().item()

                if per_token_kl is not None:
                    diag["kl"] = _masked_mean(per_token_kl).item()

                if entropy is not None:
                    # TRL ``entropy`` = masked_batch_mean(entropy_from_logits).
                    diag["entropy"] = _masked_mean(entropy).item()

                self._accum_buf["diag"].append(diag)
        # --- Intermediate-tensor divergence diagnostic (env-gated) -------------
        # When WW_IS_DIAG_DIR is set, dump norms/scalars of EVERY intermediate
        # that feeds the loss for THIS _compute_loss call (one microbatch shard),
        # so manual-vs-HF replays on the SAME captured round can be diffed
        # stage-by-stage to locate where the math diverges. All reads are under
        # torch.no_grad() (no graph retention, no overhead to the backward).
        # No-op + zero overhead otherwise (env lookup cached on self). Writes to
        # a SEPARATE dir (never the captures dir) — captures stay read-only.
        _diag_dir = getattr(self, "_is_diag_dir", None)
        if _diag_dir is None:
            _diag_dir = os.environ.get("WW_IS_DIAG_DIR") or ""
            self._is_diag_dir = _diag_dir
        if _diag_dir:
            try:
                import json as _json
                _node_tag = os.environ.get("WW_IS_DIAG_NODE", "node")
                _step_tag = os.environ.get("WW_IS_DIAG_STEP", "")
                _idx = self._is_diag_idx = getattr(self, "_is_diag_idx", -1) + 1
                with torch.no_grad():
                    import hashlib as _hl
                    def _sha(t):
                        # content hash of the raw bytes (bit-diff, not norm-diff).
                        # bf16 has no numpy dtype -> view as int16 for the raw bits.
                        _tc = t.detach().cpu().contiguous()
                        if t.dtype == torch.bfloat16:
                            _b = _tc.view(torch.int16).numpy().tobytes()
                        else:
                            _b = _tc.numpy().tobytes()
                        return _hl.sha256(_b).hexdigest()[:16]
                    def _s(t):
                        if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
                            return None
                        tf = t.float()
                        return {
                            "shape": list(t.shape),
                            "dtype": str(t.dtype).replace("torch.", ""),
                            "sha": _sha(t),
                            "min": float(tf.min().item()),
                            "mean": float(tf.mean().item()),
                            "max": float(tf.max().item()),
                            "abs_mean": float(tf.abs().mean().item()),
                            "l2_norm": float(tf.norm().item()),
                        }
                    _rec = {
                        "node": _node_tag, "call_idx": _idx,
                        "batch_size": int(batch_size),
                        "mask_tokens": float(mask.sum().item()),
                        "global_batch_size_override": int(global_batch_size_override),
                        "effective_max_len": int(self.config.max_completion_length),
                        "loss_reduction": loss_reduction,
                        "engine_is_correction": bool(getattr(self.config, "engine_is_correction", False)),
                        "engine_is_mode": getattr(self.config, "engine_is_mode", None),
                        "engine_is_cap": getattr(self.config, "engine_is_cap", None),
                        # --- inputs (post any WW_LOSS_FP32 upcast) ---
                        "per_token_logps": _s(per_token_logps),
                        "old_per_token_logps": _s(old_per_token_logps),
                        "sampling_per_token_logps": _s(sampling_per_token_logps),
                        "ref_per_token_logps": _s(ref_per_token_logps),
                        "advantages": _s(advantages),
                        "mask": _s(mask),
                        "entropy": _s(entropy),
                        # --- PPO surrogate stage ---
                        "log_ratio": _s(log_ratio),
                        "coef_1": _s(coef_1),
                        "coef_2": _s(coef_2),
                        "per_token_loss1": _s(per_token_loss1),
                        "per_token_loss2": _s(per_token_loss2),
                        "per_token_loss_pre_is": _s(_loss_pre_is),
                        # --- engine IS-correction stage ---
                        "is_logps_diff": _s(_is_logps_diff),
                        "is_ratio_uncapped": _s(_is_ratio_uncapped),
                        "importance_sampling_ratio": _s(importance_sampling_ratio),
                        "per_token_loss_post_is": _s(_loss_post_is),
                        # --- KL stage ---
                        "per_token_kl": _s(per_token_kl),
                        "per_token_loss_final": _s(per_token_loss),
                        # --- aggregation stage ---
                        "masked_loss": _s(masked_loss),
                        "per_seq_loss": _s(per_seq_loss),
                        "loss": float(loss.item()),
                    }
                os.makedirs(_diag_dir, exist_ok=True)
                _pfx = f"{_node_tag}_rank{self.rank}"
                if _step_tag:
                    _pfx = f"s{_step_tag}_{_pfx}"
                _path = os.path.join(_diag_dir, f"{_pfx}_call{_idx:03d}.json")
                with open(_path, "w") as _f:
                    _json.dump(_rec, _f, indent=2)
            except Exception:
                logger.exception("WW_IS_DIAG: failed to dump")
        # --- end intermediate-tensor divergence diagnostic --------------------

        # [IS-DECOMP] Stash the per-token/microbatch tensors needed for the
        # identical-batch IS-vs-noIS gradient decomposition (dual backward) and
        # the per-sequence/per-token IS-coefficient dump. First-call-only
        # (one microbatch); cleared after _forward_accum_grad consumes it.
        # Gated by WW_IS_DECOMP_DUMP. _loss_pre_is is the PPO surrogate BEFORE
        # the detached engine-IS multiply => differentiable, == the noIS loss.
        if os.environ.get("WW_IS_DECOMP_DUMP"):
            try:
                _cnt = getattr(self, "_is_decomp_calls", 0)
                self._is_decomp_calls = _cnt + 1
                if _cnt == 0:  # first _compute_loss call this run
                    self._is_decomp_capture = {
                        "pre_is_pt": _loss_pre_is,          # [B,C] pre-IS per-token loss (diff'able)
                        "post_is_pt": per_token_loss,       # [B,C] post-IS per-token loss
                        "ratio": importance_sampling_ratio, # [B,1] detached (sequence_mask mode)
                        "old_minus_sampling": (
                            (old_per_token_logps - sampling_per_token_logps) * mask
                        ) if (sampling_per_token_logps is not None) else None,
                        "old_lp": old_per_token_logps,
                        "sampling_lp": sampling_per_token_logps,
                        "mask": mask,
                        "advantages": advantages,
                        "coef_1": coef_1,
                        "seq_lens": mask.sum(-1).detach(),
                        "loss_reduction": loss_reduction,
                        "eff_batch": (
                            global_batch_size_override
                            if (loss_reduction == "dr_grpo" and global_batch_size_override > 0)
                            else batch_size
                        ),
                        "eff_maxlen": (
                            self.config.max_completion_length
                            if loss_reduction == "dr_grpo"
                            else None
                        ),
                    }
            except Exception:
                self._is_decomp_capture = None

        return loss

    @staticmethod
    def _selective_log_softmax(
        logits: torch.Tensor,
        label_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-probability of chosen tokens (per-token, memory-efficient).

        Delegates to :func:`whirlwind.model.updater_utils.selective_log_softmax`
        (kept here as a staticmethod for backward-compatible call sites and the
        ``tests/test_trl_padding_alignment.py`` regression that accesses
        ``ModelUpdateNode._selective_log_softmax``).
        """
        return updater_utils.selective_log_softmax(logits, label_ids)

    def _compute_per_token_logps(
        self,
        model: Any,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        logits_to_keep: int,
        temperature: float,
        compute_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute per-token log-probs for the completion span only (TRL-aligned).

        Mirrors TRL ``_get_per_token_logps_and_entropies`` (grpo_trainer.py:
        1108-1124): forward over ``prompt_completion_ids`` (left-pad prompts +
        right-pad completions), shift, slice to the last ``logits_to_keep``
        positions (= the completion span), gather the chosen-token logprob.

        Returns ``[batch, logits_to_keep]`` -- completion-shaped, already
        aligned to ``completion_mask`` and the rollout's
        ``sampling_per_token_logps``. No prepend/append hack: because every
        completion is left-aligned at column P_max, completion logprobs are
        completion-column-aligned by construction.

        Args:
            model:                 FSDP-wrapped model
            prompt_completion_ids: [batch, P+C] (left-pad prompts + right-pad completions)
            attention_mask:        [batch, P+C]
            completion_ids:        [batch, C] chosen completion token ids
                                   (pad_token_id at right-pad; masked out downstream)
            logits_to_keep:        C = completion_ids.size(1); slice logits to the
                                   completion span
            temperature:           temperature scaling for logits
            compute_entropy:       if True, also return per-token categorical entropy
                (under no_grad) aligned to the same completion span.

        Returns:
            (per_token_logps, entropy) where entropy is None unless
            ``compute_entropy=True``. Both are [batch, C].
        """
        # ------------------------------------------------------------------
        # Chunked lm_head forward (logits_chunk_size > 0): the body produces
        # hidden states, then lm_head + selective_log_softmax run in token
        # chunks so the full [B, C, V] logits are never materialized at once
        # (peak = one chunk). Per-token logps are assembled to [B, C] and the
        # STANDARD loss is called once on them (unmodified) with global
        # normalizers. Works for all reductions incl. grpo (whose per-sequence
        # normalization is non-additive across token chunks, so the loss must
        # see the full [B, C] -- chunking is forward-only, never loss-level).
        if (
            getattr(self, "_chunked", False)
            and getattr(self, "_body_module", None) is not None
            and getattr(self, "_lm_head_module", None) is not None
        ):
            if getattr(self, "_chunked_tied", False):
                return self._compute_per_token_logps_chunked_tied(
                    model=model,
                    prompt_completion_ids=prompt_completion_ids,
                    attention_mask=attention_mask,
                    completion_ids=completion_ids,
                    logits_to_keep=logits_to_keep,
                    temperature=temperature,
                    compute_entropy=compute_entropy,
                )
            return self._compute_per_token_logps_chunked(
                prompt_completion_ids=prompt_completion_ids,
                attention_mask=attention_mask,
                completion_ids=completion_ids,
                logits_to_keep=logits_to_keep,
                temperature=temperature,
                compute_entropy=compute_entropy,
            )
        # ------------------------------------------------------------------
        # Forward pass. Under fsdp_autocast, run under torch.autocast(bf16)
        # and cast logits to fp32 -- TRL bf16=True semantics (accelerator.py:
        # 1807-1818). matmuls/linear cast bf16 on-the-fly; softmax/lm_head stay
        # fp32 (autocast op-allowlist), so the log-softmax gradient is fp32.
        if self._use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(prompt_completion_ids, attention_mask=attention_mask)  # type: ignore[operator]
            logits = outputs.logits.float()  # [batch, P+C, vocab_size]
        else:
            outputs = model(prompt_completion_ids, attention_mask=attention_mask)  # type: ignore[operator]
            logits = outputs.logits  # [batch, P+C, vocab_size]
            # [precision remedy B] WW_LOGITS_FP32: upcast logits to fp32 before
            # selective_log_softmax so per_token_logps are fp32. The bf16
            # forward noise in the logits remains (params are bf16), but the
            # log-softmax + downstream seq-sums/exps are fp32-precise.
            # Equivalent to TRL's cast_lm_head_to_fp32. Env-gated, no-op off.
            if self._logits_fp32:
                logits = logits.float()

        # Shift + slice to the completion span (TRL grpo_trainer.py:1109-1114).
        # logits[:, :-1] drops the last position (nothing to predict); then keep
        # the last C positions, which predict completion_ids[0..C-1]:
        #   shifted position (P-1)+j predicts input_ids[P+j] = completion_ids[j].
        logits = logits[:, :-1, :]  # [batch, P+C-1, vocab_size]
        if logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:, :]  # [batch, C, vocab_size]
        else:
            # Degenerate: no completion tokens.
            zeros = torch.zeros(
                prompt_completion_ids.size(0), 0, dtype=logits.dtype, device=logits.device
            )
            return zeros, (zeros.clone() if compute_entropy else None)

        # Temperature scaling (applied after slicing, matching TRL).
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-9)

        # log-prob of the chosen completion tokens. completion_ids carries
        # pad_token_id (NOT -100) at right-pad slots; those positions gather
        # meaningless logprobs that completion_mask zeros out downstream -- exactly
        # as TRL (which also passes completion_ids with the pad token there).
        per_token_logps = self._selective_log_softmax(logits, completion_ids)  # [batch, C]

        # Per-token categorical entropy (TRL ``entropy_from_logits``), under
        # no_grad so it never builds a graph. Computed on the same completion-
        # sliced logits -> [batch, C], already aligned (no prepend needed).
        entropy: torch.Tensor | None = None
        if compute_entropy:
            with torch.no_grad():
                # entropy = logsumexp(logits) - sum(softmax(logits) * logits)
                entropy = torch.logsumexp(logits, dim=-1) - (
                    torch.softmax(logits, dim=-1).mul(logits).sum(dim=-1)
                )

        return per_token_logps, entropy

    def _compute_per_token_logps_chunked(
        self,
        *,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        logits_to_keep: int,
        temperature: float,
        compute_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Chunked-lm_head variant of :meth:`_compute_per_token_logps`.

        Body (transformer) and lm_head are wrapped as SEPARATE FSDP2 units
        (see ``__init__`` when ``logits_chunk_size > 0``), so this calls them
        individually: one body forward -> hidden [B, P+C, H] (the grad
        boundary), then lm_head + selective_log_softmax over token chunks of
        ``logits_chunk_size`` along the completion span. The full [B, C, V]
        logits tensor is never materialized (peak = one chunk's [B, cs, V]).

        The per-token logps are concatenated back to [B, C] (cheap: B*C floats,
        not B*C*V) and returned in the exact same shape/alignment as the
        non-chunked path, so the downstream standard loss is called once and
        UNMODIFIED. Chunking is forward-only; the loss never sees chunks.
        """
        # --- body forward -> hidden states (the grad boundary) ---
        if self._use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self._body_module(
                    prompt_completion_ids, attention_mask=attention_mask
                )
            hidden = outputs.last_hidden_state  # [B, P+C, H]
        else:
            outputs = self._body_module(
                prompt_completion_ids, attention_mask=attention_mask
            )
            hidden = outputs.last_hidden_state  # [B, P+C, H]

        # Shift + slice to the completion span (mirror the non-chunked path:
        # hidden[:, :-1] drops the last position; keep the last C positions,
        # which predict completion_ids[0..C-1]).
        hidden = hidden[:, :-1, :]  # [B, P+C-1, H]
        if logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]  # [B, C, H]
        else:
            zeros = torch.zeros(
                prompt_completion_ids.size(0),
                0,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            return zeros, (zeros.clone() if compute_entropy else None)

        C = hidden.size(1)
        chunk_size = max(int(getattr(self.config, "logits_chunk_size", 0)), 1)
        per_token_logps_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []

        # Tied embeddings: lm_head is NOT FSDP-wrapped; its weight IS
        # embed_tokens.weight (a DTensor from body's FSDP unit, kept gathered
        # by reshard_after_forward=False). F.linear(regular, DTensor) fails, so
        # we extract the weight's local tensor (full after body all-gather) and
        # do a regular F.linear. The backward flows through .to_local() (a view
        # of the DTensor's _local_tensor) so FSDP's reduce-scatter handles it.
        tied = getattr(self, "_chunked_tied", False)
        _tied_weight = None
        _tied_bias = None
        if tied:
            from torch.distributed.tensor import DTensor  # type: ignore
            if isinstance(self._lm_head_module.weight, DTensor):
                _tied_weight = self._lm_head_module.weight.to_local()
                if getattr(self._lm_head_module, "bias", None) is not None:
                    _tied_bias = self._lm_head_module.bias.to_local()

        # --- lm_head in token chunks: peak logits = [B, cs, V], not [B, C, V] ---
        for i in range(0, C, chunk_size):
            h_chunk = hidden[:, i : i + chunk_size, :]  # [B, cs', H]
            ids_chunk = completion_ids[:, i : i + chunk_size]  # [B, cs']

            if _tied_weight is not None:
                # Tied: regular matmul with the gathered weight's local tensor
                logits_chunk = torch.nn.functional.linear(
                    h_chunk, _tied_weight, _tied_bias
                )
            elif self._use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits_chunk = self._lm_head_module(h_chunk)  # [B, cs', V]
                logits_chunk = logits_chunk.float()  # TRL bf16=True semantics
            else:
                logits_chunk = self._lm_head_module(h_chunk)  # [B, cs', V]
                # [precision remedy B] WW_LOGITS_FP32 upcast, same as non-chunked.
                if self._logits_fp32:
                    logits_chunk = logits_chunk.float()

            # Tied path upcast (no autocast wrapper was used)
            if _tied_weight is not None:
                if self._use_autocast:
                    logits_chunk = logits_chunk.float()
                elif self._logits_fp32:
                    logits_chunk = logits_chunk.float()

            if temperature != 1.0:
                logits_chunk = logits_chunk / max(temperature, 1e-9)

            per_token_logps_chunks.append(
                self._selective_log_softmax(logits_chunk, ids_chunk)
            )  # [B, cs']

            if compute_entropy:
                with torch.no_grad():
                    entropy_chunks.append(
                        torch.logsumexp(logits_chunk, dim=-1)
                        - torch.softmax(logits_chunk, dim=-1)
                        .mul(logits_chunk)
                        .sum(dim=-1)
                    )  # [B, cs']

        per_token_logps = torch.cat(per_token_logps_chunks, dim=1)  # [B, C]
        entropy = torch.cat(entropy_chunks, dim=1) if compute_entropy else None
        return per_token_logps, entropy

    def _compute_per_token_logps_chunked_tied(
        self,
        *,
        model: Any,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        logits_to_keep: int,
        temperature: float,
        compute_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Chunked lm_head for TIED embeddings (root FSDP wrap, no split).

        With tied embeddings, lm_head.weight IS embed_tokens.weight — they
        can't be in separate FSDP2 units, and the DTensor weight only works
        INSIDE the FSDP forward context. So we monkey-patch the lm_head's
        forward to chunk + selective_log_softmax, then call the ROOT forward
        (which all-gathers everything via FSDP). The chunked lm_head calls run
        inside the root's FSDP forward where the DTensor weight is usable.

        The monkey-patched lm_head returns per_token_logps [B, C] as the
        model's "logits" output — this flows to the loss for backward, so
        FSDP's reduce-scatter triggers correctly on the root's backward hook.
        """
        original_forward = self._lm_head_module.forward
        chunk_size = max(int(getattr(self.config, "logits_chunk_size", 0)), 1)
        result: dict[str, Any] = {"ptl": None, "ent": None}
        _self = self
        _selective = self._selective_log_softmax

        def chunked_forward(hidden: torch.Tensor) -> torch.Tensor:
            # hidden is [B, P+C, H] (model forward doesn't slice when no
            # logits_to_keep kwarg is passed). Shift + slice to completion span.
            h = hidden[:, :-1, :]  # [B, P+C-1, H]
            if logits_to_keep > 0:
                h = h[:, -logits_to_keep:, :]  # [B, C, H]
            C = h.size(1)
            ptl_chunks: list[torch.Tensor] = []
            ent_chunks: list[torch.Tensor] = []
            for i in range(0, C, chunk_size):
                h_chunk = h[:, i : i + chunk_size, :]
                ids_chunk = completion_ids[:, i : i + chunk_size]
                if _self._use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits_chunk = original_forward(h_chunk)
                    logits_chunk = logits_chunk.float()
                else:
                    logits_chunk = original_forward(h_chunk)
                    if _self._logits_fp32:
                        logits_chunk = logits_chunk.float()
                if temperature != 1.0:
                    logits_chunk = logits_chunk / max(temperature, 1e-9)
                ptl_chunks.append(_selective(logits_chunk, ids_chunk))
                if compute_entropy:
                    with torch.no_grad():
                        ent_chunks.append(
                            torch.logsumexp(logits_chunk, dim=-1)
                            - torch.softmax(logits_chunk, dim=-1)
                            .mul(logits_chunk)
                            .sum(dim=-1)
                        )
            result["ptl"] = torch.cat(ptl_chunks, dim=1)
            if compute_entropy:
                result["ent"] = torch.cat(ent_chunks, dim=1)
            # Return per_token_logps as the model's "logits" — flows to loss
            # for backward so FSDP reduce-scatter triggers on root backward.
            return result["ptl"]

        self._lm_head_module.forward = chunked_forward  # type: ignore[method-assign]
        try:
            if self._use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        prompt_completion_ids, attention_mask=attention_mask
                    )
            else:
                outputs = model(
                    prompt_completion_ids, attention_mask=attention_mask
                )
        finally:
            self._lm_head_module.forward = original_forward  # type: ignore[method-assign]

        # outputs.logits IS the per_token_logps [B, C] from chunked_forward.
        per_token_logps = outputs.logits
        entropy = result["ent"]
        return per_token_logps, entropy

    def _use_grad_accum_microbatch_size(self, apply_fn: Callable, data: BatchedMessage):
        # Sentinel micro_batch_size <= 0 (use -1): FULLY BYPASS microbatching.
        # One forward + one backward over the full (rank-sharded) batch — no
        # split_into, no per-microbatch loop, no per-microbatch loss aggregation.
        # This is the "is microbatching itself the problem?" A/B: it removes the
        # per-microbatch bf16-forward tile-shape noise that the IS exp(seq_sum)
        # amplifies (see TIPS_AND_TRICKS.md), and the per-microbatch loss grouping. The
        # dr_grpo denominator stays correct: _compute_loss falls back to the
        # rank-shard batch_size (== data.batch_size // world_size) when the
        # override is 0, which equals the per-rank share the split branch sets.
        # CAUTION: at the real batch (256 -> 128/rank) one forward materializes
        # ~50 GB of bf16 logits (151k vocab) — ~100 GB with WW_LOGITS_FP32. HF
        # OOMs even at mb=8, so mb=-1 is manual-node + bf16-logits only unless
        # the global batch is shrunk.
        if self.config.micro_batch_size <= 0:
            return apply_fn(data)
        assert self.config.micro_batch_size >= 1, "Must have microbatch of at least 1"
        if self.config.micro_batch_size * self.world_size < data.batch_size:
            # use the built in stuff to execute the operation one by one
            assert data.batch_size % (self.config.micro_batch_size * self.world_size) == 0, (
                f"Must be aligned, no padding available: {data.batch_size} {self.config.micro_batch_size} {self.world_size}"
            )
            n_parts = data.batch_size // (self.config.micro_batch_size * self.world_size)

            # Compute full-batch overrides for loss normalization.
            # These are injected into every microbatch so that DR-GRPO's
            # denominator uses global batch_size and max completion length
            # rather than the local microbatch shard dimensions.
            from whirlwind.messages import TrainStepRequest

            if isinstance(data, TrainStepRequest):
                # Per-rank batch size for the dr_grpo/grpo denominator.
                # FSDP2 averages gradients across ranks (ReduceOp.AVG), so each
                # rank must normalize by its OWN share. This split runs BEFORE
                # the _fsdp_using_function rank scatter (it decorates the inner
                # _forward_accum_grad), so data is the GLOBAL batch here: divide
                # by world_size, or the gradient comes out world_size× smaller
                # than the mb=-1 path (measured 0.5001× at world_size 2). No-op
                # at world_size 1.
                global_batch_size = data.batch_size // self.world_size
                global_max_completion_len = max(
                    sum(1 for t in lab if t != -100) for lab in data.label_ids
                )
                # Global total real-token count (sum of completion lengths over
                # the full batch) for the bnpo/ce denominator. Injected so the
                # loss uses the GLOBAL total, not the per-microbatch shard sum
                # (which would be wrong when the mask is sliced).
                global_total_tokens = max(1, sum(
                    sum(1 for t in lab if t != -100) for lab in data.label_ids
                ) // self.world_size)  # per-rank share, same reason as above
            else:
                global_batch_size = 0
                global_max_completion_len = 0
                global_total_tokens = 0

            microbatch_breakdown = data.split_into(n_parts)
            accum = []
            for mb_data in microbatch_breakdown:
                if isinstance(mb_data, TrainStepRequest):
                    if mb_data.global_batch_size_override == 0:
                        mb_data.global_batch_size_override = global_batch_size
                    if mb_data.global_max_completion_len_override == 0:
                        mb_data.global_max_completion_len_override = global_max_completion_len
                    if mb_data.global_total_tokens_override == 0:
                        mb_data.global_total_tokens_override = global_total_tokens
                accum.append(apply_fn(mb_data))

            if isinstance(accum[0], BatchedMessage):
                # Microbatch aggregation: SUM per-microbatch TrainStepResult
                # losses (NOT average). dr_grpo/bnpo/ce per-microbatch losses
                # already divide by the full-batch denominator
                # (global_batch_size_override × max_completion_length), so
                # SUMMING recovers the true batch loss. The prior join()
                # AVERAGE spuriously divided by n_microbatches, making the
                # reported loss (a) ~n_microbatchesx smaller than the true
                # loss and (b) micro_batch_size-dependent (mb8 vs mb4 reported
                # ~2x different losses for the SAME batch). The gradient is
                # unaffected -- loss.backward() runs per-microbatch inside
                # _forward_accum_grad (grads accumulate/sum), so training was
                # always correct; only the reported scalar was wrong.
                #
                # Rank combination (in _fsdp_using_function) still AVERAGES
                # via join(), which matches _model_update_step's
                # all_reduce(SUM)/world_size -- so path-1 (this return) now
                # equals path-2 (the _model_update_step buffer aggregation).
                #
                # grad_norm/lr are None at this stage (set later in
                # model_update_step); only loss is combined here.
                if isinstance(accum[0], TrainStepResult):
                    return TrainStepResult(
                        loss=sum(r.loss for r in accum if r.loss is not None),
                        grad_norm=None,
                        lr=None,
                        total_completions=next(
                            (r.total_completions for r in accum
                             if r.total_completions is not None), None),
                        checkpoint_path=accum[0].checkpoint_path,
                    )
                return type(accum[0]).join(accum)
            elif isinstance(accum[0], Message):
                return accum[0]
            else:
                raise AssertionError(f"wrong message type {type(accum[0])}")
        else:
            # if withing limits, do no extra tricks
            return apply_fn(data)

    # ------------------------------------------------------------------
    # Exposed endpoints (rank 0, sync)
    # ------------------------------------------------------------------

    @TopKLogitsRequest.decorate_serdes()
    def forward_no_grad(self, data: TopKLogitsRequest) -> Any:
        """Inference pass: return top-K logits per position (no gradients).

        Applies temperature before ranking.
        """
        return self._forward_no_grad(data)

    @_fsdp_using_function("_forward_no_grad")  # type: ignore[misc]
    def _forward_no_grad(self, req: TopKLogitsRequest) -> TopKLogitsResult:
        """Run forward pass with no gradients, extract top-K + selected logits.

        Called on each FSDP rank with a shard of the original batch.
        """
        model = self._fsdp
        # Eval path: full-sequence right-padded forward (top-K at every position).
        # Uses _pad_full_sequence, NOT the TRL training structure -- eval is not
        # part of the IS / loss path and keeps its original layout (val numbers
        # stay stable). See _pad_and_stack for the TRL-aligned training tensors.
        input_ids, label_ids = self._pad_full_sequence(req.input_ids, req.label_ids)

        with torch.no_grad():
            if self._use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids)  # type: ignore[operator]
                logits = outputs.logits.float()
            else:
                outputs = model(input_ids)  # type: ignore[operator]
                logits = outputs.logits  # [batch, seq_len, vocab_size]

        topk_ids, topk_logits, selected_logits = map_topk(
            logits, label_ids, req.topk, req.temperature
        )

        return TopKLogitsResult(
            topk_ids=topk_ids.tolist(),
            topk_logits=topk_logits.tolist(),
            selected_logits=selected_logits.tolist(),
            # Propagate the per-sample tracing fields so the result stays
            # correlated with its source prompts/dataset entries across
            # split_into/join/slice_by/iter_by_*. (Previously these were
            # dropped, leaving the result positionally correct but
            # untraceable back to its inputs.)
            dataset_ids=req.dataset_ids,
            entry_ids=req.entry_ids,
            rollout_ids=req.rollout_ids,
            subrollout_ids=req.subrollout_ids,
            step=req.step,
        )

    @TrainStepRequest.decorate_serdes()
    def forward_accum_grad(self, data: TrainStepRequest) -> Any:
        """Forward pass + backward (accumulates gradients).

        Does NOT call optimizer.step() — that's model_update_step's job.
        """
        # Re-enabled microbatching (was disabled for troubleshooting). With GAS
        # (gradient_accumulation_steps) now orchestrator-owned via req.accum_steps,
        # each micro-shard inherits accum_steps from split_into and applies loss/accum_steps
        # inside _forward_accum_grad; the sum over micro-shards is the correct
        # (global_loss / accum_steps). micro_batch_size controls per-forward memory;
        # at micro_batch_size=1 this forwards one completion at a time (peak logits
        # ~0.3 GB) instead of the whole batch at once (the prior OOM trigger when
        # this line was commented out).
        try:
            from whirlwind.model import _step1_bitdiff
            _step1_bitdiff.record_input_checksum(data.input_ids)
            _step1_bitdiff.maybe_capture_request(data)
        except Exception:
            pass
        return self._use_grad_accum_microbatch_size(self._forward_accum_grad, data)

    @_fsdp_using_function("_forward_accum_grad")  # type: ignore[misc]
    def _forward_accum_grad(self, req: TrainStepRequest) -> TrainStepResult:
        """Run GRPO forward + backward on a shard. Gradients accumulate across calls.

        Called on each FSDP rank with a shard of the original batch.

        Computes per-token log-probs via a forward pass, then applies
        the PPO-clipped surrogate loss weighted by GRPO advantages.
        Optionally adds KL regularization from a reference model.
        """
        model = self._fsdp

        # Pad variable-length sequences and stack into tensors
        tensors = self._pad_and_stack(
            req.input_ids,
            req.label_ids,
            req.advantages,
            req.sampling_logprobs,
            req.ref_logprobs,
            req.old_logprobs,  # PPO old-policy logprobs (optional; None = recompute)
        )
        prompt_completion_ids = tensors["prompt_completion_ids"]
        attention_mask = tensors["attention_mask"]
        completion_ids = tensors["completion_ids"]
        completion_mask = tensors["completion_mask"]
        logits_to_keep = tensors["logits_to_keep"]
        advantages = tensors["advantages"]
        # ``sampling_logprobs`` carries the engine generation-time logprobs
        # (the rollout's vLLM/SGLang logprobs). In TRL terms this is
        # ``sampling_per_token_logps`` — the generation-engine logprobs used
        # ONLY for the bounded engine↔trainer IS ratio, NOT for ``coef_1``.
        sampling_per_token_logps = tensors["sampling_logprobs"]
        ref_logprobs = tensors["ref_logprobs"]
        # ``old_logprobs`` (if provided by the caller) are PPO old-policy
        # logprobs — used for ``coef_1`` and the engine-IS ``old`` term. None
        # means the caller did not supply them; see the resolution below.
        provided_old_logps = tensors["old_logprobs"]
        assert prompt_completion_ids is not None
        assert attention_mask is not None
        assert completion_ids is not None
        assert completion_mask is not None
        assert advantages is not None

        # Compute per-token log-probs for the current policy (differentiable).
        # TRL-aligned: completion-shaped [B, C] logps, aligned to completion_mask
        # and sampling_per_token_logps by construction (no prepend/offset hack).
        per_token_logps, entropy = self._compute_per_token_logps(
            model=model,
            prompt_completion_ids=prompt_completion_ids,
            attention_mask=attention_mask,
            completion_ids=completion_ids,
            logits_to_keep=logits_to_keep,
            temperature=self.config.temperature,
            compute_entropy=self._compute_entropy,
        )

        # ``old_per_token_logps`` feeds ``coef_1`` (the PPO ratio) and the
        # ``old`` term of the engine IS ratio. Resolution order:
        #   1. If the caller provided PPO old-policy logprobs (req.old_logprobs),
        #      use those directly (e.g. a replay/diagnostic with captured old
        #      logprobs, or a future num_iterations>1 path that caches them).
        #   2. Else use ``per_token_logps.detach()``. At GAS=1 /
        #      num_iterations=1 (steps_per_generation=1) the old policy == the
        #      current policy (weights synced to the engine, no update in
        #      between), so old_per_token_logps == per_token_logps EXACTLY — a
        #      separate no_grad forward would return the same values (same
        #      weights, same inputs, same kernel) at ~32 extra forwards/step on
        #      the model GPU. Using .detach() is free, bit-identical, and gives
        #      coef_1 = exp(0) = 1.0 exactly (PPO clipping fully dormant, no
        #      bf16 noise) — strictly better than a fresh forward. Matches
        #      TRL's own no-IS fallback (grpo_trainer.py:2478-2479).
        #      For num_iterations>1 the caller MUST pass req.old_logprobs: a
        #      fresh forward on CURRENT weights would be wrong there (it
        #      yields current, not old), so the old fresh-forward branch is gone.
        do_engine_is = (
            getattr(self.config, "engine_is_correction", False)
            and self.config.loss_reduction != "ce"
            and sampling_per_token_logps is not None
        )
        if provided_old_logps is not None:
            old_per_token_logps = provided_old_logps
        else:
            # old == current at GAS=1; .detach() is exact and free (was: a
            # redundant no_grad forward when do_engine_is, None otherwise).
            old_per_token_logps = per_token_logps.detach()

        # Compute GRPO loss (PPO-clipped surrogate + engine IS correction + optional KL)
        loss = self._compute_loss(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_per_token_logps=ref_logprobs,
            advantages=advantages,
            mask=completion_mask,
            global_batch_size_override=req.global_batch_size_override,
            global_max_completion_len_override=req.global_max_completion_len_override,
            completion_len=tensors["completion_len"],
            prompt_len=tensors["prompt_len"],
            total_tokens=req.global_total_tokens_override,
            sampling_per_token_logps=sampling_per_token_logps,
            entropy=entropy,
        )

        # Gradient-accumulation scaling (orchestrator-owned via the request).
        # Divide the per-call loss by accum_steps so that accumulating
        # `accum_steps` forward_accum_grad calls yields the correct MEAN
        # gradient over the full accumulated batch (TRL-aligned). 1 = no-op
        # (byte-identical to legacy behaviour). This is the only place GAS
        # touches the loss/gradient; _compute_loss is now scale-agnostic.
        accum_steps = max(getattr(req, "accum_steps", 1), 1)
        if accum_steps > 1:
            loss = loss / accum_steps

        # Backward pass (gradients accumulate)
        #
        # [IS-DECOMP] Identical-batch IS-vs-noIS gradient comparison + per-seq/
        # per-token IS-coefficient dump. On the first microbatch of the run,
        # backward the PRE-IS loss (== noIS loss, the detached engine ratio not
        # yet applied) to measure the core gradient, zero it, then let the
        # normal post-IS backward proceed — both from the SAME forward, same
        # weights, same batch. Gated by WW_IS_DECOMP_DUMP; one-shot; guarded.
        _decomp_core_norm = None
        _cap = getattr(self, "_is_decomp_capture", None)
        if _cap is not None:
            try:
                _pre = _cap["pre_is_pt"]
                _m = _cap["mask"]
                _red = _cap["loss_reduction"]
                _ml = _pre * _m
                if _red == "dr_grpo":
                    _pre_scalar = _ml.sum() / (_cap["eff_batch"] * _cap["eff_maxlen"])
                elif _red == "bnpo":
                    _pre_scalar = _ml.sum() / _m.sum().clamp(min=1.0)
                elif _red == "ce":
                    _pre_scalar = -(_pre * _m).sum() / _m.sum().clamp(min=1.0)
                else:  # grpo
                    _pre_scalar = (_ml.sum(-1) / _m.sum(-1).clamp(min=1.0)).mean()
                _pre_scalar = _pre_scalar / accum_steps  # match post-IS scaling
                _pre_scalar.backward(retain_graph=True)
                _decomp_core_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                )
                # Clear the pre-IS grad so it does not contaminate the real
                # post-IS accumulation that follows.
                for _p in model.parameters():
                    if _p.grad is not None:
                        _p.grad = None
            except Exception:
                logger.exception("IS-DECOMP: pre-IS backward failed")
                _decomp_core_norm = None

        loss.backward()

        # [IS-DECOMP] record post-IS (actual) grad norm + dump per-seq/per-token
        # tensors, then release the capture. Same forward as the core measurement.
        if _cap is not None:
            try:
                _decomp_actual_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                )
                _dump_dir = os.environ.get("WW_IS_DECOMP_DIR") or "/tmp/ww_is_decomp"
                os.makedirs(_dump_dir, exist_ok=True)
                _rank = self.rank
                _ratio = _cap["ratio"]
                _seqmask = (_cap["mask"].sum(-1) > 0)
                # per-sequence summary (move to cpu for saving)
                if _ratio is not None and _ratio.dim() >= 1:
                    _r_per = _ratio.flatten()
                    if _r_per.numel() == _cap["seq_lens"].numel():
                        _seq_ratio = _r_per.detach().cpu()
                    else:  # token-level ratio -> take per-seq mean as approx
                        _seq_ratio = (_ratio * _cap["mask"]).sum(-1) / _cap["mask"].sum(-1).clamp(min=1.0)
                        _seq_ratio = _seq_ratio.detach().cpu()
                else:
                    _seq_ratio = None
                _oms = _cap["old_minus_sampling"]
                _seq_sum = (_oms.sum(-1) if _oms is not None else None)
                _seq_sum = _seq_sum.detach().cpu() if _seq_sum is not None else None
                torch.save(
                    {
                        "rank": _rank,
                        "core_grad_norm": _decomp_core_norm,
                        "actual_grad_norm": _decomp_actual_norm,
                        "grad_attenuation": (
                            _decomp_actual_norm / _decomp_core_norm
                            if _decomp_core_norm else None
                        ),
                        "seq_ratio": _seq_ratio,            # [B] engine IS ratio per completion
                        "seq_sum": _seq_sum,                # [B] sum(old-sampling) per completion
                        "seq_lens": _cap["seq_lens"].cpu(),
                        "old_minus_sampling": _oms.detach().cpu() if _oms is not None else None,  # [B,C]
                        "old_lp": _cap["old_lp"].detach().cpu(),
                        "sampling_lp": (_cap["sampling_lp"].detach().cpu()
                                         if _cap["sampling_lp"] is not None else None),
                        "mask": _cap["mask"].detach().cpu(),
                        "advantages": _cap["advantages"].detach().cpu(),
                        "coef_1": _cap["coef_1"].detach().cpu(),
                        "pre_is_pt": _cap["pre_is_pt"].detach().cpu(),   # [B,C] pre-IS per-token loss
                        "post_is_pt": _cap["post_is_pt"].detach().cpu(), # [B,C] post-IS per-token loss
                    },
                    os.path.join(_dump_dir, f"decomp_rank{_rank}.pt"),
                )
                if _rank == 0:
                    print(
                        f"[IS-DECOMP] IDENTICAL-BATCH A/B "
                        f"core(noIS)_grad={_decomp_core_norm} "
                        f"actual(IS)_grad={_decomp_actual_norm} "
                        f"attenuation=actual/core="
                        f"{(_decomp_actual_norm/_decomp_core_norm) if _decomp_core_norm else 'NA'} "
                        f"(engine ratio mean drives this)",
                        flush=True,
                    )
            except Exception:
                logger.exception("IS-DECOMP: post-IS record/dump failed")
            finally:
                self._is_decomp_capture = None  # release graph refs

        # ---- [REALDATA-CAPTURE] one-shot capture of the 16 per-forward input
        # tuples (+ weights once) at a target opt-step, for the offline
        # manual-vs-HF accumulation replay.
        # Env-gated, fires exactly once per rank at the target opt-step; no-op
        # otherwise. Captures the raw req tensors BEFORE pad_and_stack so the
        # replay can re-run _pad_and_stack faithfully. Weights dumped on the
        # first capture call of the target step (same weights across all 16
        # forwards; updated only at the flush).
        _rd_step = os.environ.get("WW_REALDATA_CAPTURE_OPTSTEP")
        if _rd_step:
            try:
                _target = int(_rd_step)
                _acc = max(getattr(req, "accum_steps", 1), 1)
                # persistent forward counter (TrainStepRequest has no .step attr)
                _c = getattr(self, "_realdata_fwd_count", 0) + 1
                self._realdata_fwd_count = _c
                _opt = ((_c - 1) // _acc) + 1            # opt-step (1-based)
                _fwd_in_opt = ((_c - 1) % _acc) + 1       # 1..accum_steps within the opt-step
                if _opt == _target:
                    _rd_dir = os.environ.get("WW_REALDATA_CAPTURE_DIR") or "/tmp/ww_realdata"
                    os.makedirs(_rd_dir, exist_ok=True)
                    _rec = {
                        "opt_step": _opt, "fwd_in_opt": _fwd_in_opt, "rank": self.rank,
                        "accum_steps": _acc, "fwd_count": _c,
                        "input_ids": req.input_ids, "label_ids": req.label_ids,
                        "advantages": req.advantages, "sampling_logprobs": req.sampling_logprobs,
                        "ref_logprobs": req.ref_logprobs, "old_logprobs": req.old_logprobs,
                    }
                    torch.save(_rec, os.path.join(_rd_dir, f"fwd_opt{_opt}_fwd{_fwd_in_opt}_rank{self.rank}.pt"))
                    # dump weights once per rank (first fwd of the target opt-step)
                    if _fwd_in_opt == 1 and not os.path.exists(os.path.join(_rd_dir, f"weights_opt{_opt}_rank{self.rank}.pt")):
                        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
                        _sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
                        torch.save(_sd, os.path.join(_rd_dir, f"weights_opt{_opt}_rank{self.rank}.pt"))
                        logger.info("[REALDATA-CAPTURE] dumped weights + fwd %d at opt-step %d rank %d -> %s", _fwd_in_opt, _opt, self.rank, _rd_dir)
                    else:
                        logger.info("[REALDATA-CAPTURE] dumped fwd %d at opt-step %d rank %d", _fwd_in_opt, _opt, self.rank)
            except Exception:
                logger.exception("[REALDATA-CAPTURE] failed")

        # Buffer loss and sample count for accumulation tracking.
        # Each rank buffers its own shard metrics; _model_update_step
        # aggregates across ranks and accumulation steps.
        self._accum_buf["losses"].append(loss.item())
        self._accum_buf["samples"].append(len(req.input_ids))

        return TrainStepResult(
            loss=loss.item(),
            grad_norm=None,
            lr=None,
        )

    @ModelUpdateRequest.decorate_serdes()
    def model_update_step(self, data: ModelUpdateRequest) -> Any:
        """Execute optimizer step: clip grads, step, zero_grad, log metrics.

        Flushes accumulated gradients. Should be called after one or more
        forward_accum_grad calls.
        """
        result = self._model_update_step(data)
        # Log training metrics asynchronously (fire-and-forget, doesn't block).
        # All model-node metrics are training-only (validation does no training),
        # so group them under the ``train/`` W&B supergroup. The tracker preserves
        # the ``train/`` prefix instead of re-prefixing with the node id.
        if self._node is not None:
            metrics: dict[str, float] = {
                "train/loss": result.loss,
            }
            if result.grad_norm is not None:
                metrics["train/grad_norm"] = float(result.grad_norm)
            if result.lr is not None:
                metrics["train/lr"] = float(result.lr)
            if result.total_completions is not None:
                metrics["train/total_completions"] = float(result.total_completions)
            # Merge per-step diagnostics (IS ratio, clip ratios, sampling logp
            # difference, kl, entropy) — mirrors TRL grpo_trainer self._metrics
            # keys. Prefix each with ``train/`` so they land in the train group.
            for k, v in self._last_diag.items():
                # Numeric diags -> float for wandb; string diags (e.g. the
                # worst_offenders log) pass through as-is (wandb accepts strings).
                metrics[f"train/{k}"] = float(v) if isinstance(v, (int, float)) else v
            self._node.call_bg_no_return(
                "tracker",
                "log_metrics",
                {
                    "metrics": metrics,
                    "node_id": self.config.node_id,
                },
            )
        return result

    @_fsdp_using_function("_model_update_step")  # type: ignore[misc]
    def _model_update_step(self, req: ModelUpdateRequest) -> TrainStepResult:
        """Clip gradients, run optimizer step, step scheduler, zero gradients.

        Called on ALL ranks (broadcast).
        """
        model = self._fsdp

        # [step1-bitdiff] grad stats BEFORE clip (manual path) — clip mutates grads
        _grad_pre = None
        try:
            from whirlwind.model import _step1_bitdiff
            if _step1_bitdiff.enabled():
                _grad_pre = _step1_bitdiff.grad_stats_pre_clip(model.parameters())
        except Exception:
            pass

        # [update-diag] grad stats BEFORE clip (manual path) — the true backward
        # + all-reduce output. No-op unless WW_UPDATE_DIAG is set.
        try:
            from whirlwind.model import _update_diag
            _update_diag.capture_grads(model, "grad_pre_clip", self.rank, "manual")
        except Exception:
            pass

        # Gradient clipping. With fsdp_cpu_offload=True the sharded grads are
        # CPU DTensors and stock clip_grad_norm_ fails ("No backend type
        # associated with device type cpu" — its DTensor reduce needs a mesh
        # backend, absent on CPU). Use the CPU-offload-aware path instead;
        # otherwise the proven stock clip_grad_norm_ (GPU DTensors).
        if getattr(self.config, "fsdp_cpu_offload", False):
            grad_norm = self._clip_grad_norm_cpu_offload(
                model.parameters(), max_norm=self.config.max_grad_norm
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),  # type: ignore[arg-type]
                max_norm=self.config.max_grad_norm,
            )

        # [update-diag] grad stats AFTER clip (manual path) — what the optimizer sees.
        try:
            from whirlwind.model import _update_diag
            _update_diag.capture_grads(model, "grad_post_clip", self.rank, "manual")
        except Exception:
            pass

        # Optional per-step LR override
        if req.lr is not None:
            for param_group in self._optimizer.param_groups:  # type: ignore[union-attr]
                param_group["lr"] = req.lr

        # [step1-bitdiff] param norm BEFORE optimizer step (manual path)
        _param_norm_pre = None
        try:
            from whirlwind.model import _step1_bitdiff
            if _step1_bitdiff.enabled():
                _param_norm_pre = _step1_bitdiff.param_l2_norm(model.parameters())
        except Exception:
            pass

        # [update-diag] param stats BEFORE optimizer.step (manual path) — == W_s
        try:
            from whirlwind.model import _update_diag
            _update_diag.capture_params(model, "param_pre_step", self.rank, "manual")
            _update_diag.capture_optim(self._optimizer, "optim_pre_step", model)
        except Exception:
            pass

        # With fsdp_cpu_offload=True, params + grads are CPU DTensors, but
        # raw torch.optim.AdamW creates bf16 optimizer state (matching the bf16
        # param) which is numerically unstable on CPU (NaNs ~step 19). fp32
        # state doesn't work either (AdamW's _single_tensor_adam requires
        # state dtype == grad dtype on CPU). So for CPU offload we use a custom
        # fp32-internal Adam step (upcast param→fp32, Adam math in fp32, write
        # back bf16) — keeps bf16 params (no colocate OOM) with fp32 numerics.
        if getattr(self.config, "fsdp_cpu_offload", False):
            self._cpu_adamw_fp32_step()
        else:
            self._optimizer.step()  # type: ignore[union-attr]
        self._optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]

        # [update-diag] param stats AFTER optimizer.step (manual path) — == W'
        # (the update result; the key manual-vs-HF comparison stage).
        try:
            from whirlwind.model import _update_diag
            _update_diag.capture_params(model, "param_post_step", self.rank, "manual")
            _update_diag.capture_optim(self._optimizer, "optim_post_step", model)
            _update_diag.write(self.rank, "manual")
        except Exception:
            pass

        # [step1-bitdiff] capture fingerprints + exit after step 1 (manual path)
        try:
            from whirlwind.model import _step1_bitdiff
            _step1_bitdiff.dump_and_exit(model.parameters(), float(grad_norm), self.rank, "manual", _param_norm_pre, _grad_pre)
        except Exception:
            pass

        current_lr = req.lr if req.lr is not None else self.config.learning_rate

        # Aggregate buffered losses across accumulation steps.
        # Each rank has its own buffer (one entry per shard per fwd call).
        buf_losses = self._accum_buf["losses"]
        buf_samples = self._accum_buf["samples"]
        acc_steps = max(self.config.gradient_accumulation_steps, 1)
        num_shards = len(buf_losses)

        if self.config.loss_reduction == "dr_grpo":
            # After the denominator fix, each rank's dr_grpo loss uses the
            # per-rank batch share (B_global / world_size) as the denominator,
            # so sum(buf_losses) on a rank = (rank's local numerator) /
            # (B_local × cap). all_reduce SUM collects every rank's numerator
            # over the SAME per-rank denominator, and the existing /world_size
            # then yields Σ_global / (world_size × B_local × cap) =
            # Σ_global / (B_global × cap) = the true global loss. Do NOT
            # multiply by world_size here — that was correct only under the
            # old global-denominator override and would over-report by
            # world_size×. No-op at world_size==1 (÷1), so single-GPU is
            # unchanged.
            rank_loss = sum(buf_losses)
        elif self.config.loss_reduction in ("bnpo", "ce"):
            # Same as dr_grpo: each rank's token denominator is its share
            # (total_tokens / world_size), so all_reduce SUM / world_size of
            # sum(buf_losses) is Σ_numerators / total_tokens, the true loss.
            rank_loss = sum(buf_losses)
        else:
            # "grpo" — mean-of-means is correct (per-seq mean then global mean).
            rank_loss = sum(buf_losses) / num_shards if num_shards else 0.0

        # All-reduce loss (mean) and samples (sum) across ranks.
        # NCCL requires CUDA tensors — use .cuda() when multi-GPU.
        device = torch.device("cuda") if self.world_size > 1 else torch.device("cpu")
        loss_tensor = torch.tensor(rank_loss, dtype=torch.float32, device=device)
        samples_tensor = torch.tensor(sum(buf_samples), dtype=torch.float64, device=device)
        if self.world_size > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor = loss_tensor / self.world_size
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

        # Final correction for BNPO/DR-GRPO/CE/GRPO: multiply by acc_steps
        # to undo the per-shard division in _compute_loss. Every loss mode
        # divides by acc_steps for gradient scaling (see _compute_loss), so
        # every mode's reported metric needs this undo -- including GRPO, whose
        # mean-of-means aggregation above is otherwise left /acc_steps too low.
        if self.config.loss_reduction in ("bnpo", "dr_grpo", "ce", "grpo"):
            loss_tensor = loss_tensor * acc_steps

        # Flush the buffer now that we've extracted metrics
        self._accum_buf["losses"] = []
        self._accum_buf["samples"] = []

        # Aggregate diagnostics across microbatches and ranks. Aggregation op
        # depends on the key semantics (mirror TRL: nanmin/nanmax for the
        # extremes, mean for the rest) — averaging min/max would report
        # "mean of per-shard minimums" etc., NOT the global min/max.
        #   keys ending in "/min" → min-of-mins (cross-rank all_reduce MIN)
        #   keys ending in "/max" → max-of-maxes (cross-rank all_reduce MAX)
        #   everything else        → mean      (cross-rank all_reduce SUM / world)
        # The per-microbatch diag always emits the same key set on every rank
        # (sentinels for empty shards), so the collective is desync-safe.
        diag_buf = self._accum_buf.get("diag", [])
        agg_diag: dict[str, float] = {}
        if diag_buf:
            keys: set[str] = set()
            for d in diag_buf:
                keys.update(d.keys())
            # IMPORTANT: iterate keys in a DETERMINISTIC order (sorted). Set /
            # dict iteration order is hash-based and differs across torchrun
            # ranks (PYTHONHASHSEED is randomized per process). The cross-rank
            # all_reduce loop below pairs tensors by CALL ORDER — if ranks
            # iterate keys in different orders, the collectives get cross-wired
            # (rank0's all_reduce(uncapped_mean) pairs with rank1's
            # all_reduce(signed_mean)), producing impossible logged values like
            # is_ratio/mean < 0 and uncapped_min > uncapped_max. Sorting makes
            # every rank issue the all_reduce calls in the same key sequence.
            for k in sorted(keys):
                vals = [d[k] for d in diag_buf if k in d]
                if not vals:
                    continue
                # Non-numeric diag values (e.g. the worst_offenders string
                # log) are passed through last-write-wins and excluded from
                # the cross-rank all_reduce (which requires floats).
                if not all(isinstance(v, (int, float)) for v in vals):
                    agg_diag[k] = vals[-1]
                    continue
                if k.endswith("/min"):
                    agg_diag[k] = min(vals)  # min-of-mins
                elif k.endswith("/max"):
                    agg_diag[k] = max(vals)  # max-of-maxes
                else:
                    agg_diag[k] = sum(vals) / len(vals)  # mean-of-means
            if self.world_size > 1 and agg_diag:
                # Sorted for the same cross-rank ordering reason as above —
                # all_reduce calls must be issued in identical key order on
                # every rank or the tensors get cross-wired.
                for k in sorted(agg_diag.keys()):
                    v = agg_diag[k]
                    # Skip non-numeric values (passed through above) —
                    # all_reduce requires a float tensor.
                    if not isinstance(v, (int, float)):
                        continue
                    t = torch.tensor(v, dtype=torch.float32, device=device)
                    if k.endswith("/min"):
                        dist.all_reduce(t, op=dist.ReduceOp.MIN)  # type: ignore[arg-type]
                    elif k.endswith("/max"):
                        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # type: ignore[arg-type]
                    else:
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)  # type: ignore[arg-type]
                        t = t / self.world_size
                    agg_diag[k] = t.item()
            # Collapse sentinels (+inf/-inf from all-empty shards) to 0.0.
            for k, v in list(agg_diag.items()):
                if v == float("inf") or v == float("-inf") or v != v:
                    agg_diag[k] = 0.0
        self._last_diag = agg_diag
        self._accum_buf["diag"] = []

        # [step1-decomp] One-shot intermediate-value trace: print the EXACT
        # scalars that determine the gradient at the first few optimizer
        # steps, so the annihilation mechanism is visible numerically (not
        # via SHA/hand-wave). Gated by WW_PRINT_STEP1_DIAG; prints on rank 0
        # for the first WW_PRINT_STEP1_DIAG_N flushes (default 3).
        try:
            import os as _os
            if _os.environ.get("WW_PRINT_STEP1_DIAG"):
                _n = int(_os.environ.get("WW_PRINT_STEP1_DIAG_N", "3"))
                cnt = getattr(self, "_step1_decomp_cnt", 0)
                if cnt < _n and self.rank == 0:
                    self._step1_decomp_cnt = cnt + 1
                    _g = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                    _rmean = agg_diag.get("sampling/importance_sampling_ratio/mean", float("nan"))
                    _rmin = agg_diag.get("sampling/importance_sampling_ratio/min", float("nan"))
                    _rmax = agg_diag.get("sampling/importance_sampling_ratio/max", float("nan"))
                    _rumean = agg_diag.get("sampling/importance_sampling_ratio/uncapped_mean", float("nan"))
                    _seqsum = agg_diag.get("sampling/sampling_logp_difference/seq_sum_mean", float("nan"))
                    _signed = agg_diag.get("sampling/sampling_logp_difference/signed_mean", float("nan"))
                    _c1 = agg_diag.get("model_is/coef_1_mean", float("nan"))
                    _c1dev = agg_diag.get("model_is/coef_1_abs_dev_from_1", float("nan"))
                    _clip = agg_diag.get("clip_ratio/region_mean", float("nan"))
                    _lpre = agg_diag.get("loss/pre_is_mean", float("nan"))
                    _implied_core = (_g / _rmean) if _rmean and _rmean == _rmean and _rmean != 0 else float("nan")
                    print(
                        f"[STEP-DECOMP] opt-step={cnt+1} "
                        f"MODEL-IS coef_1={_c1:.5f} (|.-1|={_c1dev:.2e} clip_region={_clip:.4f}) | "
                        f"ENGINE-IS ratio[mean={_rmean:.5f} min={_rmin:.5f} max={_rmax:.5f} uncapped={_rumean:.5f}] "
                        f"logp_diff[seq_sum={_seqsum:.4f} signed={_signed:.5f}] | "
                        f"LOSS pre_is={_lpre:.5f} post_is={loss_tensor.item():.5f} | "
                        f"GRAD actual={_g:.5f} implied_core=actual/ratio={_implied_core:.5f}",
                        flush=True,
                    )
        except Exception:
            pass

        return TrainStepResult(
            loss=loss_tensor.item(),
            grad_norm=grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            lr=current_lr,
            total_completions=int(samples_tensor.item()),
        )

    @_fsdp_using_function("_save_hot_checkpoint_dcp")  # type: ignore[misc]
    def _save_hot_checkpoint_dcp(self, req: CheckpointRequest) -> OperationResult:
        """Save DCP sharded model + optimizer state dicts.

        Called on ALL ranks (broadcast). Each rank saves its own shard.

        Any failure is caught and returned as an ``OperationResult(ok=False)``
        rather than allowed to propagate: this handler runs inside
        ``_fsdp_using_function``, whose outer ``all_gather_object`` would be
        skipped on a raised exception and desync the standby ranks (deadlock).
        ``CheckpointException`` is a ``BaseException`` subclass, so we catch
        ``BaseException`` (re-raising real interrupts).
        """
        path = req.path
        import torch.distributed.checkpoint as dcp  # type: ignore[import-not-found]
        from torch.distributed.checkpoint.state_dict import get_state_dict

        try:
            # Write into a sibling temp dir and swap it in only after EVERY rank
            # has finished, so a crash mid-save can never corrupt the only hot
            # checkpoint (the in-place overwrite could).
            tmp = path.rstrip("/") + ".tmp"
            os.makedirs(tmp, exist_ok=True)

            assert self._fsdp is not None
            assert self._optimizer is not None

            state_dict_model, state_dict_optim = get_state_dict(
                self._fsdp,
                self._optimizer,
            )
            dcp.save(  # type: ignore[union-attr]
                state_dict={"model": state_dict_model, "optim": state_dict_optim},
                checkpoint_id=tmp,
            )
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path, ignore_errors=True)
                os.replace(tmp, path)
            logger.info("DCP checkpoint saved: %s", path)
            return OperationResult(ok=True, message=f"DCP checkpoint saved to {path}")
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            logger.exception("DCP checkpoint save failed")
            return OperationResult(ok=False, message=f"DCP save failed: {exc}")

    @ExportWeightsRequest.decorate_serdes()
    def export_weights(self, data: ExportWeightsRequest) -> Any:
        """Export consolidated HF weights (no DCP shards, no optimizer).

        All ranks participate (all-reduce to consolidate), but only rank 0
        writes to disk.  Used for fast weight-sharing with the rollout node.
        """
        return self._export_weights(data)

    @_fsdp_using_function("_export_weights")  # type: ignore[misc]
    def _export_weights(self, req: ExportWeightsRequest) -> OperationResult:
        """Consolidate FSDP shards into HF format on rank 0.

        All ranks contribute via all-reduce.  Rank 0 writes:
        - ``pytorch_model.bin``  (full consolidated state dict)
        - ``config.json``        (model config for SGLang reload)

        No DCP shards, no optimizer state — lightweight weight export only.

        Under PEFT/LoRA (``self._is_peft``) this instead exports the adapter
        ONLY (``adapter_config.json`` + ``adapter_model.safetensors``, the
        standard PEFT format the rollout engines load natively). The frozen
        base is NOT exported — the rollout already has the base loaded and
        applies the adapter on top. This makes the sync ~MB vs ~GB. Uses
        ``ignore_frozen_params=True`` (pytorch #169448: FSDP2 state-dict gather
        hangs on frozen base params without it). See docs/lora_design.md.
        """
        ckpt_dir = req.path
        assert self._fsdp is not None

        # --- PEFT / LoRA: adapter-only export ---
        if self._is_peft:
            from torch.distributed.checkpoint.state_dict import (  # type: ignore[import-not-found]
                StateDictOptions,
                get_model_state_dict,
            )

            logger.info("All ranks beginning ADAPTER-only export: %s", ckpt_dir)
            # ignore_frozen_params=True is REQUIRED: FSDP2's state-dict gather
            # hangs on frozen (base) params otherwise (pytorch #169448).
            adapter_state_dict = get_model_state_dict(
                self._fsdp,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                    ignore_frozen_params=True,
                ),
            )
            if self.rank == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                # PEFT save_pretrained writes adapter_config.json +
                # adapter_model.safetensors (the standard adapter dir both
                # vLLM and SGLang read natively via add_lora / load_lora_adapter).
                self._model.save_pretrained(ckpt_dir, state_dict=adapter_state_dict)
                logger.info("Exported LoRA adapter: %s", ckpt_dir)
            dist.barrier()
            return OperationResult(ok=True, message=f"LoRA adapter exported to {ckpt_dir}")

        # --- full fine-tuning: full state-dict export (existing path) ---
        from torch.distributed.checkpoint.state_dict import (  # type: ignore[import-not-found]
            StateDictOptions,
            get_model_state_dict,
        )

        logger.info("All ranks beginning weight export: %s", ckpt_dir)
        full_state_dict = get_model_state_dict(
            self._fsdp,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

        # All ranks participate in all-reduce to get full state dict
        if self.rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
            logger.info("Writing the gathered weights to: %s", model_path)
            torch.save(full_state_dict, model_path)
            logger.info("Exported consolidated weights: %s", model_path)

            # Save config for SGLang weight reload. Reuse the config cached
            # at init (self._hf_config) instead of re-fetching from the HF Hub
            # on every export — AutoConfig.from_pretrained() issues a HEAD
            # request per call even when the file is cached, which over a
            # 1000-step run means ~1000 pointless network round-trips. The
            # config never changes during training, so the cached in-memory
            # copy is identical. (vLLM's load_weights ignores config.json
            # entirely; only SGLang's update_weights_from_disk reads it.)
            self._hf_config.save_pretrained(ckpt_dir)
            logger.info("Exported config: %s", ckpt_dir)

        dist.barrier()
        return OperationResult(ok=True, message=f"Weights exported to {ckpt_dir}")

    # ------------------------------------------------------------------
    # Sleep / Wake — free GPU for the colocate rollout phase
    # ------------------------------------------------------------------
    # Goal: during the rollout phase (vLLM owns the GPU), release as much
    # model-side GPU memory as possible; during the training phase, have
    # everything back on GPU.
    #
    # IMPORTANT (root cause of the IS-ratio collapse): an earlier implementation
    # reassigned ``fsdp_param._sharded_param_data`` to a fresh CPU/GPU tensor
    # each cycle. That BREAKS the storage alias between ``_sharded_param_data``
    # and ``sharded_param._local_tensor`` (both are views of the same
    # ``padded_sharded_param`` created at init — see FSDP2 ``_fsdp_param.py``
    # lines 399/408). After the break, the optimizer writes the update to the
    # DTensor's (old) storage while the next forward's all-gather reads
    # ``_sharded_param_data`` (the reassigned, stale storage). The resulting
    # per-step perturbation compounds through the IS ``exp(seq_sum)`` amplifier
    # over ~15 steps until the IS ratio collapses (0.14→0.02) and the
    # validation reward freezes.
    #
    # So sleep moves ONLY optimizer state to CPU and leaves the FSDP param
    # shards on GPU. The storage alias is never touched, so training is
    # correct (val 0.63, matching HF-Trainer/TRL).
    #
    # For LARGE models whose shards don't fit alongside vLLM's KV cache, do NOT
    # use the shard reassignment. Instead set ``fsdp_cpu_offload: true`` in the
    # config — FSDP2's ``CPUOffloadPolicy`` keeps params on CPU permanently and
    # copies them to GPU per-forward (then frees), which preserves the storage
    # alias (it never reassigns ``_sharded_param_data`` after init). The optim
    # state is still offloaded here. Trade-off: slower training (per-forward
    # CPU→GPU copy + optimizer update on the CPU-resident shard).

    @NoInputRequest.decorate_serdes()
    def sleep(self, data: Message):
        return self._sleep(data)

    @_fsdp_using_function("_sleep")
    def _sleep(self, data: Message) -> Any:
        """Release model-side GPU memory for the colocate rollout phase.

        Moves ONLY optimizer state to CPU; FSDP param shards stay on GPU
        (storage alias untouched, see above).
        """
        assert self._fsdp is not None
        assert self._optimizer is not None

        kept_grads = 0
        moved_optim = 0

        # Param shards stay on GPU. Grads are only counted for the log: they
        # stay on GPU at GAS>1 (assigning a CPU tensor to a cuda DTensor .grad
        # fails device validation, and the per-rank grad footprint is
        # negligible), so gradient accumulation across a sleep/wake cycle is
        # preserved bit-exactly.
        fsdp_state = self._fsdp._get_fsdp_state()
        if fsdp_state._fsdp_param_group is not None:
            for fsdp_param in fsdp_state._fsdp_param_group.fsdp_params:
                if (
                    fsdp_param.sharded_param.grad is not None
                    and fsdp_param.sharded_param.grad.device.type == "cuda"
                ):
                    kept_grads += 1

        # Optimizer states -> CPU (safe: pure PyTorch optimizer-state iteration;
        # no FSDP storage alias involved). This is the dominant GPU chunk freed
        # for the rollout phase (AdamW m+v ≈ 2×param size).
        for param_states in self._optimizer.state.values():
            for key, val in param_states.items():
                if hasattr(val, "device") and val.device.type == "cuda":
                    param_states[key] = _to_cpu_pinned(val)
                    moved_optim += 1

        torch.cuda.empty_cache()
        logger.info(
            "Sleep (rank %d): %d optim states to CPU; params and %d grads stay on GPU",
            self.rank, moved_optim, kept_grads,
        )
        return OperationResult(
            ok=True,
            message=(
                f"Moved {moved_optim} optim states to CPU; params and {kept_grads} "
                f"grads stay on GPU (rank {self.rank})"
            ),
        )

    @NoInputRequest.decorate_serdes()
    def wake(self, data: Message):
        return self._wake(data)

    @_fsdp_using_function("_wake")
    def _wake(self, data: Message) -> Any:
        """Restore model-side GPU memory for the training phase.

        Restores ONLY optimizer state to GPU; param shards never left.
        """
        assert self._fsdp is not None
        assert self._optimizer is not None

        device = torch.device("cuda", self.local_rank)
        restored_optim = 0

        # Optimizer states -> GPU.
        for param_states in self._optimizer.state.values():
            for key, val in param_states.items():
                if hasattr(val, "device") and val.device.type == "cpu":
                    param_states[key] = val.to(device, non_blocking=True)
                    restored_optim += 1

        logger.info("Wake (rank %d): %d optim states to GPU", self.rank, restored_optim)
        return OperationResult(
            ok=True,
            message=f"Restored {restored_optim} optim states to GPU (rank {self.rank})",
        )

    # ------------------------------------------------------------------
    # Internal wake/sleep helpers (used by checkpoint ops)
    # ------------------------------------------------------------------

    def _wake_model(self) -> None:
        """Wake the model from sleep (GPU tensors)."""
        if self._asleep:
            self._wake(Message())
            self._asleep = False

    def _sleep_model(self) -> None:
        """Sleep the model (CPU tensors)."""
        if not self._asleep:
            self._sleep(Message())
            self._asleep = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cpu_adamw_fp32_step(self) -> None:
        """fp32-internal AdamW step for FSDP2 CPU-offloaded bf16 params.

        Raw ``torch.optim.AdamW`` creates bf16 optimizer state (matching the
        bf16 param) which NaNs on CPU ~step 19 (bf16 precision loses the
        small update / denominator). fp32 state breaks AdamW's ``lerp_``
        (CPU requires state dtype == grad dtype). This custom step does the
        Adam math in fp32 internally: fp32 momentum buffers (created once),
        fp32 master copy of the param for the update, written back to bf16.
        Equivalent to ``torch.optim.AdamW`` with ``fused=False`` in fp32, but
        with bf16 param storage — the standard "bf16 params + fp32 optimizer
        state" pattern. Reads lr/betas/eps/wd from the configured optimizer's
        param group so the config stays the single source of truth.
        """
        pg = self._optimizer.param_groups[0]  # type: ignore[union-attr]
        lr = pg["lr"]
        beta1, beta2 = pg["betas"]
        eps = pg["eps"]
        wd = pg.get("weight_decay", 0.0)
        st = self._cpu_adamw_state
        st["step"] = st.get("step", 0) + 1
        step_idx = st["step"]
        bc1 = 1 - beta1 ** step_idx
        bc2 = 1 - beta2 ** step_idx
        for p in self._fsdp.parameters():  # type: ignore[union-attr]
            if p.grad is None:
                continue
            # Operate on the plain sharded local tensors (NOT the DTensor
            # wrapper) so all ops are plain-tensor arithmetic on CPU.
            pdata = p.data._local_tensor if hasattr(p.data, "_local_tensor") else p.data
            g = p.grad._local_tensor if hasattr(p.grad, "_local_tensor") else p.grad
            ps = st.setdefault(id(p), {})
            if "exp_avg" not in ps:
                ps["exp_avg"] = torch.zeros_like(pdata, dtype=torch.float32)
                ps["exp_avg_sq"] = torch.zeros_like(pdata, dtype=torch.float32)
            ea, easq = ps["exp_avg"], ps["exp_avg_sq"]
            g32 = g.float()
            ea.mul_(beta1).add_(g32, alpha=1 - beta1)
            easq.mul_(beta2).addcmul_(g32, g32, value=1 - beta2)
            denom = (easq.sqrt() / (bc2 ** 0.5)).add_(eps)
            step_size = lr / bc1
            p32 = pdata.float()
            if wd != 0:
                # Decoupled weight decay as torch AdamW does it: p *= 1 - lr*wd.
                # (Was p += wd*p: wrong sign and no lr factor -> x(1+wd) per step.)
                p32.mul_(1.0 - lr * wd)
            p32.addcdiv_(ea, denom, value=-step_size)
            pdata.copy_(p32.to(pdata.dtype))

    def _sync_optim_state_to_param_device(self) -> None:
        """Move all optimizer states onto each param's device, as FP32.

        torch.optim.AdamW's foreach state-init path allocates ``exp_avg`` /
        ``exp_avg_sq`` / ``step`` on the default CUDA device even when the param
        is a CPU DTensor (FSDP2 CPUOffloadPolicy). This moves every state tensor
        to the param's ``_local_tensor`` device (CPU under offload) BEFORE
        ``step()``, so the in-place Adam update sees matching devices.

        States are forced to FP32 regardless of param dtype: under CPUOffloadPolicy
        the params are bf16, and CPU bf16 AdamW state is numerically unstable
        (NaNs ~step 19 because the bf16 update magnitude falls below bf16
        resolution and the denominator loses precision). FP32 state on CPU is
        stable; AdamW's in-place ops auto-promote the bf16 grad into the fp32
        state, and the bf16 param update absorbs the (small) fp32 result. This
        is the standard "fp32 optimizer state + bf16 params" pattern. Idempotent
        and a no-op once states are already fp32 on the param device.
        """
        for p, st in self._optimizer.state.items():  # type: ignore[union-attr]
            if not st:
                continue
            tgt = p._local_tensor.device if hasattr(p, "_local_tensor") else p.device
            for k, v in list(st.items()):
                if not hasattr(v, "device"):
                    continue
                vd = v._local_tensor.device if hasattr(v, "_local_tensor") else v.device
                vdt = (v._local_tensor.dtype if hasattr(v, "_local_tensor")
                       else getattr(v, "dtype", None))
                need_dev = vd != tgt
                need_dtype = vdt is not None and vdt != torch.float32
                if need_dev or need_dtype:
                    st[k] = v.to(device=tgt, dtype=torch.float32, non_blocking=True)

    def _clip_grad_norm_cpu_offload(self, params, max_norm: float) -> torch.Tensor:
        """Grad-norm clip for FSDP2 CPU-offloaded sharded DTensor grads.

        Stock ``torch.nn.utils.clip_grad_norm_`` fails on CPU DTensors because
        its total-norm aggregation triggers a DeviceMesh reduce that has no CPU
        backend. This helper bypasses the mesh: it computes each shard's local
        sq-norm, sums locally, all-reduces the scalar across the shard group via
        NCCL (the scalar is moved to CUDA for the reduce), takes sqrt to get the
        global norm, and scales each shard in-place by the clip coefficient.
        Correct because the global L2 norm of a sharded grad is the sqrt of the
        sum of per-shard sq-norms, and scaling every shard by the same coef
        scales the global norm by that coef. The reduce spans one node's ranks
        under HYBRID_SHARD: the other nodes hold identical copies of the same
        shards, and summing over the world would inflate the norm by
        sqrt(nodes). Returns the global grad norm (as a CPU tensor) to match
        the stock function's return type shape.
        """
        import torch.distributed as dist

        local_sq = torch.zeros((), dtype=torch.float32, device="cpu")
        grads: list[torch.Tensor] = []
        for p in params:
            if p.grad is None:
                continue
            g = p.grad
            local_t = g._local_tensor if hasattr(g, "_local_tensor") else g
            local_sq = local_sq + local_t.float().pow(2).sum().cpu()
            grads.append(g)
        if self.world_size > 1:
            dev = torch.device("cuda", self.local_rank)
            t = local_sq.to(dev, non_blocking=True)
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self._shard_group)  # type: ignore[arg-type]
            # Blocking copy: a non_blocking one lands in pinned memory
            # asynchronously, and the sqrt below read it before it arrived.
            local_sq = t.cpu()
        total_norm = local_sq.sqrt()
        clip_coef = max_norm / (total_norm + 1e-6)
        if float(clip_coef) < 1.0:
            cf = float(clip_coef)
            for g in grads:
                local_t = g._local_tensor if hasattr(g, "_local_tensor") else g
                local_t.mul_(cf)
        return total_norm

    def _make_optimizer(self) -> Any:
        """Create optimizer from config.

        Under PEFT/LoRA only the adapter params have ``requires_grad=True``
        (the frozen base is requires_grad=False), so filtering to trainable
        params trains the adapter only and avoids wasting optimizer state on
        the frozen base. For full fine-tuning all params are trainable, so the
        filter is a no-op.
        """
        trainable_params = [p for p in self._fsdp.parameters() if p.requires_grad]
        if self.config.optim == "adamw":
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optim == "sgd":
            return torch.optim.SGD(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optim}")

    def ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model_loaded": self._model is not None,
        }

    # ------------------------------------------------------------------
    # State endpoints
    # ------------------------------------------------------------------

    def _full_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "lr": self.config.learning_rate,
        }
        return {
            "config": _to_dict(self.config),
            "state": state,
        }

    @StateGetRequest.decorate_serdes()
    def _get_state(self, req: StateGetRequest) -> StateResult:
        state = self._full_state()
        if req.keys:
            state = {k: v for k, v in state.items() if k in req.keys}
        return StateResult(data=state, ok=True)

    @StateSetRequest.decorate_serdes()
    def _set_state(self, req: StateSetRequest) -> StateResult:
        # TODO: possibly set the LR here
        applied = {k: v for k, v in req.data.items() if not req.keys or k in req.keys}
        return StateResult(data=applied, ok=True)

    # ------------------------------------------------------------------
    # Hot checkpoint — DCP shards (model + optimizer state dicts)
    # ------------------------------------------------------------------

    @CheckpointRequest.decorate_serdes()
    def save_hot_checkpoint(self, req: CheckpointRequest) -> CheckpointResult:
        """Save per-rank model + optimizer state dicts via DCP.

        Wrapped in a wake/sleep cycle so it works when the node is asleep.
        """
        was_asleep = self._asleep
        try:
            if was_asleep:
                self._wake_model()

            result = self._save_hot_checkpoint_dcp(req)
            return (
                CheckpointResult(ok=True, path=req.path, step=req.step, error="")
                if result.ok
                else CheckpointResult(ok=False, path=req.path, step=req.step, error=result.message)
            )
        except Exception as exc:
            logger.exception("Hot checkpoint failed")
            return CheckpointResult(ok=False, path=req.path, step=req.step, error=str(exc))
        finally:
            if was_asleep:
                self._sleep_model()

    # ------------------------------------------------------------------
    # Cold checkpoint — consolidated HF weights only
    # ------------------------------------------------------------------

    @CheckpointRequest.decorate_serdes()
    def save_cold_checkpoint(self, req: CheckpointRequest) -> Any:
        """Save consolidated HF-compatible weights.

        Delegates to ``export_weights`` for the actual FSDP work, wrapped in a
        wake/sleep cycle so it works when the node is asleep.
        """
        return self._save_cold_checkpoint(req)

    def _save_cold_checkpoint(self, req: CheckpointRequest) -> CheckpointResult:
        """Save consolidated HF weights.

        Reuses ``export_weights`` for the actual FSDP work.
        """
        was_asleep = self._asleep
        try:
            if was_asleep:
                self._wake_model()

            # Delegate to the existing export_weights endpoint. NOTE:
            # export_weights is @ExportWeightsRequest.decorate_serdes()-wrapped,
            # so even an IN-PROCESS call returns the SERIALIZED dict
            # ({'_msg_args': [], '_msg_kwargs': {'ok': True, ...}, ...}) —
            # 'ok' is nested, not top-level. Reconstruct via from_dict (and
            # accept a raw Message too). The old top-level .get("ok") check
            # always reported ok=False with an empty error even when the
            # export succeeded.
            result = self.export_weights(ExportWeightsRequest(path=req.path))
            if isinstance(result, dict):
                result = OperationResult.from_dict(result)
            ok = bool(getattr(result, "ok", False))
            msg = str(getattr(result, "message", ""))
            return (
                CheckpointResult(ok=ok, path=req.path, step=req.step, error="")
                if ok
                else CheckpointResult(ok=False, path=req.path, step=req.step, error=msg)
            )
        except Exception as exc:
            logger.exception("Cold checkpoint failed")
            return CheckpointResult(ok=False, path=req.path, step=req.step, error=str(exc))
        finally:
            if was_asleep:
                self._sleep_model()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Load hot checkpoint — DCP shards (model + optimizer)
    # ------------------------------------------------------------------

    @CheckpointRequest.decorate_serdes()
    def load_hot_checkpoint(self, req: CheckpointRequest) -> CheckpointResult:
        """Load per-rank model + optimizer state dicts via DCP.

        Wrapped in a wake/sleep cycle so it works when the node is asleep.
        """
        was_asleep = self._asleep
        try:
            if was_asleep:
                self._wake_model()

            return self._load_hot_checkpoint_dcp(req)
        except Exception as exc:
            logger.exception("Hot checkpoint load failed")
            return CheckpointResult(ok=False, path=req.path, step=req.step, error=str(exc))
        finally:
            if was_asleep:
                self._sleep_model()

    @_fsdp_using_function("_load_hot_checkpoint_dcp")  # type: ignore[misc]
    def _load_hot_checkpoint_dcp(self, req: CheckpointRequest) -> CheckpointResult:
        """Load per-rank state dicts from DCP checkpoint.

        Uses ``torch.distributed.checkpoint.load()`` to restore
        both model and optimizer state dicts.
        """
        path = req.path
        step = req.step
        import torch.distributed.checkpoint as dcp  # type: ignore[import-not-found]
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        try:
            assert self._fsdp is not None
            assert self._optimizer is not None

            # Build FSDP-aware stubs matching the layout save_hot_checkpoint
            # writes (keys like `optim.state.0.step`). Using the raw
            # self._optimizer.state_dict() here yields a different layout and
            # makes dcp.load raise "Missing key ... optim.state.0.step".
            state_dict_model, state_dict_optim = get_state_dict(self._fsdp, self._optimizer)
            state_dict = {"model": state_dict_model, "optim": state_dict_optim}

            # Load from checkpoint (all ranks participate)
            dcp.load(state_dict=state_dict, checkpoint_id=path)  # type: ignore[union-attr]

            # Apply loaded state to model + optimizer
            set_state_dict(
                self._fsdp,
                self._optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
            )

            logger.info("DCP checkpoint loaded: %s (step=%d)", path, step)
            return CheckpointResult(ok=True, path=path, step=step, error="")
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # ``dcp.load`` raises ``CheckpointException`` (a ``BaseException``
            # subclass in torch 2.11), which ``except Exception`` does NOT catch.
            # Letting it escape would skip the outer ``all_gather_object`` in
            # ``_fsdp_using_function`` and deadlock the standby ranks. Catch it
            # and return an error result so the collective stays in sync.
            logger.exception("DCP checkpoint load failed")
            return CheckpointResult(ok=False, path=path, step=step, error=str(exc))


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Whirlwind Model Update Node")
    parser.add_argument(
        "--config", default="configs/model/fsdp2.yaml", help="Path to model YAML config"
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = ModelConfig.from_yaml(args.config)
    print(f"Starting model update node: model={config.model_path}, fsdp={config.fsdp_size}")

    node = ModelUpdateNode(config)

    async def _main() -> None:
        await node.start()
        print("Model update node running. Press Ctrl+C to stop.")
        try:
            while node.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    # Lead = GLOBAL rank 0 (the only rank that registers for RPC). With the
    # local rank, local rank 0 on every other node took the lead path, found
    # no RPC node to serve and exited, hanging the whole process group.
    if node.rank > 0:
        # Non-lead rank: run start() for dist init + model loading, then block in standby
        asyncio.run(node.start())
        node.intitialize_model(
            {}
        )  # <-- this will block until the corresponding request comes to the rank 0
        # once the model is initialized, we can enter the standby mode
        node._standby()
    else:
        asyncio.run(_main())
