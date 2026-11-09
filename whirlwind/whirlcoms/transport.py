"""SHM-aware async ZeroMQ transport with tensor support.

Handles:
- Small payloads: inline ZMQ frames (msgpack)
- Large payloads: POSIX SHM regions + metadata frame
- torch.Tensor: zero-copy via SHM (torch.frombuffer)
"""

from __future__ import annotations

import contextlib
import mmap
import os
import uuid
from typing import Any

import msgpack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHM_PREFIX = "/whirlwind-shm-"
DEFAULT_SHM_THRESHOLD = 64 * 1024  # 64 KB -> above this, use SHM

# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _is_tensor(obj: Any) -> bool:
    """Check if obj is a torch.Tensor without importing torch at module level."""
    try:
        import torch

        return isinstance(obj, torch.Tensor)
    except ImportError:
        return False


def _tensor_to_shm(tensor: Any, shm_prefix: str = SHM_PREFIX) -> tuple[str, dict]:
    """Write a CPU tensor to a named SHM region in /dev/shm.

    Returns:
        (shm_name, metadata_dict) where metadata has shape, dtype, nbytes.
    """
    tensor = tensor.cpu().contiguous()
    nbytes = tensor.numel() * tensor.element_size()

    shm_name = f"{shm_prefix}{uuid.uuid4().hex}"
    shm_path = os.path.join("/dev/shm", shm_name.lstrip("/"))

    fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, nbytes)

    # Write tensor data directly via mmap. numpy has no bfloat16: view those
    # bits as int16 (same bytes; _tensor_from_shm rebuilds from meta["dtype"]).
    import torch

    buf = (tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor).numpy()
    mm = mmap.mmap(fd, nbytes)
    mm.write(buf.tobytes())
    mm.flush()
    mm.close()

    meta = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nbytes": nbytes,
        "shm_name": shm_name,
        "shm_path": shm_path,
    }
    return shm_name, meta


def _tensor_from_shm(meta: dict) -> Any:
    """Reconstruct a torch.Tensor from SHM metadata.

    Opens the SHM region, creates a tensor via frombuffer, then unlinks.
    """
    import torch

    shm_path = meta["shm_path"]
    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int8": torch.int8,
        "torch.int16": torch.int16,
    }
    dtype = dtype_map.get(meta["dtype"])
    if dtype is None:
        raise ValueError(f"Unsupported tensor dtype: {meta['dtype']}")

    with open(shm_path, "rb") as f:
        data = f.read(meta["nbytes"])

    tensor = torch.frombuffer(data, dtype=dtype).view(meta["shape"])
    # Unlink SHM (receiver-owned cleanup)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(shm_path)
    return tensor


# ---------------------------------------------------------------------------
# Serialization / Deserialization
# ---------------------------------------------------------------------------


def _serialize_payload(
    obj: Any,
    shm_threshold: int = DEFAULT_SHM_THRESHOLD,
) -> tuple[bytes, dict | None, str | None]:
    """Serialize an object for ZMQ transport.

    Returns:
        (frame_bytes, shm_meta_or_none, shm_path_or_none)

        For tensors > threshold: frame is msgpack metadata, shm_meta has
        tensor info, and shm_path is where data lives.
        For small payloads: frame is msgpack of the whole thing.
    """
    if _is_tensor(obj):
        tensor = obj.cpu().contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > shm_threshold:
            shm_name, meta = _tensor_to_shm(tensor)
            frame = msgpack.packb(
                {
                    "_type": "tensor_shm",
                    "meta": meta,
                },
                use_bin_type=True,
            )
            return frame, meta, meta["shm_path"]
        else:
            # Small tensor — inline pickle
            import pickle

            frame = msgpack.packb(
                {
                    "_type": "tensor_inline",
                    "data": pickle.dumps(tensor),  # noqa: S301 — trusted same-machine nodes
                },
                use_bin_type=True,
            )
            return frame, None, None

    # Regular object — msgpack (with torch tensor detection for lists/dicts)
    encoder = _TorchAwareEncoder()
    packed = msgpack.packb(obj, default=encoder.encode, use_bin_type=True)
    if len(packed) > shm_threshold and not encoder._has_tensor:
        # Large non-tensor payload — put in SHM as raw bytes
        shm_name = f"{SHM_PREFIX}{uuid.uuid4().hex}"
        shm_path = os.path.join("/dev/shm", shm_name.lstrip("/"))
        with open(shm_path, "wb") as f:
            f.write(packed)
        frame = msgpack.packb(
            {
                "_type": "blob_shm",
                "nbytes": len(packed),
                "shm_name": shm_name,
                "shm_path": shm_path,
            },
            use_bin_type=True,
        )
        return frame, {"shm_path": shm_path, "nbytes": len(packed)}, shm_path
    return packed, None, None


def _deserialize_payload(data: bytes) -> Any:
    """Deserialize a received frame back to a Python object."""
    obj = msgpack.unpackb(data, raw=False)
    return _recursive_deserialize(obj)


def _recursive_deserialize(obj: Any) -> Any:
    """Recursively resolve serialized tensors and SHM blobs in nested structures."""
    if isinstance(obj, dict):
        typ = obj.get("_type")
        if typ == "tensor_shm":
            return _tensor_from_shm(obj["meta"])
        if typ == "tensor_inline":
            import pickle

            # pickle is safe here: same-machine, trusted nodes only
            return pickle.loads(obj["data"])
        if typ == "blob_shm":
            shm_path = obj["shm_path"]
            with open(shm_path, "rb") as f:
                raw = f.read(obj["nbytes"])
            with contextlib.suppress(FileNotFoundError):
                os.unlink(shm_path)
            return msgpack.unpackb(raw, raw=False)
        # Regular dict — recurse into values
        return {k: _recursive_deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        # Fast path: a list whose first element is a scalar is plain data (token
        # ids, logprobs, strings) with nothing to resolve. Skip the per-element
        # recursion — one Python call per token, which stalls the receiving
        # event loop for seconds on large RolloutResults.
        if obj and not isinstance(obj[0], (dict, list)):
            return obj
        return [_recursive_deserialize(item) for item in obj]
    return obj


class _TorchAwareEncoder:
    """msgpack encoder that detects torch tensors in nested structures."""

    def __init__(self) -> None:
        self._has_tensor = False

    def encode(self, obj: Any) -> dict:
        if _is_tensor(obj):
            self._has_tensor = True
            import pickle

            return {
                "_type": "tensor_inline",
                "data": pickle.dumps(obj.cpu().contiguous()),  # noqa: S301 — trusted
            }
        raise TypeError(f"Cannot serialize {type(obj)}")


# ---------------------------------------------------------------------------
# Message framing
# ---------------------------------------------------------------------------


def make_request_frame(
    msg_id: str,
    src_node: str,
    dst_node: str,
    endpoint: str,
    args: tuple,
    kwargs: dict,
    shm_threshold: int = DEFAULT_SHM_THRESHOLD,
    src_addr: str | None = None,
) -> tuple[bytes, bytes, dict]:
    """Build an outbound RPC request frame.

    Returns:
        (header_part, payload_part, shm_cleanup_info)
        shm_cleanup_info maps shm_path -> meta for any SHM regions created.
    """
    payload, shm_meta, shm_path = _serialize_payload(
        {"args": args, "kwargs": kwargs},
        shm_threshold=shm_threshold,
    )
    header = {
        "msg_id": msg_id,
        "src": src_node,
        "dst": dst_node,
        "endpoint": endpoint,
        "src_addr": src_addr,  # caller's RPC endpoint for reply routing
    }
    # Frame: [header_part | payload_part]  (two ZMQ parts)
    frame_header = msgpack.packb(header, use_bin_type=True)
    cleanup: dict[str, dict] = {}
    if shm_path and shm_meta is not None:
        cleanup[shm_path] = shm_meta
    return frame_header, payload, cleanup


def parse_request_frame(parts: list[bytes]) -> dict:
    """Parse an inbound RPC request frame (2 parts: header + payload)."""
    if len(parts) < 2:
        raise ValueError(f"Expected 2 frame parts, got {len(parts)}")
    header = msgpack.unpackb(parts[0], raw=False)
    payload = _deserialize_payload(parts[1])
    header["payload"] = payload
    return header


def make_reply_frame(
    msg_id: str,
    src_node: str,
    dst_node: str,
    result: Any,
    error: str | None = None,
    shm_threshold: int = DEFAULT_SHM_THRESHOLD,
) -> tuple[bytes, bytes]:
    """Build an outbound RPC reply frame.

    Returns:
        (header_part, payload_part)
    """
    header = {
        "msg_id": msg_id,
        "src": src_node,
        "dst": dst_node,
        "error": error,
    }
    payload_obj = {"result": result} if error is None else {"error": error}
    payload_frame, _, _ = _serialize_payload(payload_obj, shm_threshold)
    frame_header = msgpack.packb(header, use_bin_type=True)
    return frame_header, payload_frame


def parse_reply_frame(parts: list[bytes]) -> tuple[Any, str | None]:
    """Parse an inbound RPC reply. Returns (result, error_or_none)."""
    header = msgpack.unpackb(parts[0], raw=False)
    payload = _deserialize_payload(parts[1])
    if header.get("error"):
        return None, header["error"]
    return payload.get("result"), None
