"""Whirlcoms — ZeroMQ RPC communication module for Whirlwind clusters.

Provides:
- Node: RPC endpoint runtime with expose decorator and heartbeat
- Orchestrator: central service registry and topology broadcast
- RpcClient: standalone RPC client with retry/reconnect logic
- call: module-level convenience function for RPC calls
"""

from whirlwind.whirlcoms.orchestrator import (
    DEFAULT_CTRL_ENDPOINT,
    DEFAULT_PUB_ENDPOINT,
    Orchestrator,
)
from whirlwind.whirlcoms.node import Node, RpcError
from whirlwind.whirlcoms.rpc import RpcClient, call
from whirlwind.whirlcoms.transport import (
    DEFAULT_SHM_THRESHOLD,
    make_request_frame,
    make_reply_frame,
    parse_request_frame,
    parse_reply_frame,
)

__all__ = [
    "DEFAULT_SHM_THRESHOLD",
    "DEFAULT_CTRL_ENDPOINT",
    "DEFAULT_PUB_ENDPOINT",
    "Orchestrator",
    "Node",
    "RpcError",
    "RpcClient",
    "call",
    "make_request_frame",
    "make_reply_frame",
    "parse_request_frame",
    "parse_reply_frame",
]
