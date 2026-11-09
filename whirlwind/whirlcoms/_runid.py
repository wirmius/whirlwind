"""Run-id namespace helper for multi-run coexistence on one machine.

When ``WW_RUN_ID`` is set, IPC socket paths are namespaced with it so multiple
WW runs can coexist without colliding on the hardcoded ``/tmp/whirlwind-*``
sockets (ctrl registry, events pub, per-node DEALER endpoints). When unset,
all values are bit-for-bit identical to the pre-namespaced defaults — single-run
behavior is unchanged.

The torchrun master port is derived from the *same* env var in
``bringup_and_run.sh`` via ``cksum`` (stable, POSIX) mapped to [10000, 10999],
and the checkpoint dir is namespaced in ``config.py``. One env var derives
every collision-prone resource.
"""
from __future__ import annotations

import os

# Captured once at import (set before any WW process starts).
RUN_ID = os.environ.get("WW_RUN_ID")


def ipc_path(name: str) -> str:
    """Return the IPC endpoint for a socket ``name``.

    Unset WW_RUN_ID -> ``ipc:///tmp/whirlwind-{name}``  (unchanged default).
    Set   WW_RUN_ID -> ``ipc:///tmp/whirlwind-{run_id}-{name}``.
    """
    if RUN_ID:
        return f"ipc:///tmp/whirlwind-{RUN_ID}-{name}"
    return f"ipc:///tmp/whirlwind-{name}"
