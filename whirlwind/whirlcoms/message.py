"""Message registry (whirlcoms v2) — endpoint → (expects, returns).

The Node owns typed serdes at the seam (``from_dict(expects)`` on recv,
``to_dict(returns)`` on reply), driven by this registry populated via
:meth:`Node.expose`. The domain message types stay with users in
``whirlwind.messages`` and subclass the primitives re-exported below.

See ``docs/whirlcoms.md`` (the v2 seam section).
"""

from __future__ import annotations

# Re-export the absorbed primitives so ``whirlwind.whirlcoms.message`` is the
# canonical import surface. ``whirlwind.messages`` does not import from
# ``whirlcoms``, so this re-export is cycle-free.
from whirlwind.messages import (  # noqa: F401  (re-exported primitives)
    BatchedMessage,
    Message,
    _from_dict,
    _to_dict,
)


class MessageRegistry:
    """Endpoint → (expects, returns) registry.

    Populated by :meth:`Node.expose`. The Node reads it at the serdes seam:
    ``expects_for`` on recv (to ``from_dict`` the request), ``returns_for`` on
    reply (to ``to_dict`` the result).
    """

    def __init__(self) -> None:
        self._endpoints: dict[str, tuple[type | None, type | None]] = {}

    def register_endpoint(
        self,
        endpoint: str,
        *,
        expects: type | None = None,
        returns: type | None = None,
    ) -> None:
        """Register (or re-register) an endpoint's message contract."""
        self._endpoints[endpoint] = (expects, returns)

    def expects_for(self, endpoint: str) -> type | None:
        """Return the endpoint's input message type (None if undeclared)."""
        return self._endpoints.get(endpoint, (None, None))[0]

    def returns_for(self, endpoint: str) -> type | None:
        """Return the endpoint's output message type (None if undeclared)."""
        return self._endpoints.get(endpoint, (None, None))[1]

    def __contains__(self, endpoint: str) -> bool:
        return endpoint in self._endpoints

    def __len__(self) -> int:
        return len(self._endpoints)
