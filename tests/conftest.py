"""Suite-wide safety net for leaked zmq contexts.

A test that fails (or forgets) before ``node.stop()`` / ``orch.stop()`` leaves
its zmq context open. When gc later collects that context together with its
sockets, pyzmq's ``Context.__del__`` -> ``destroy()`` finds its socket WeakSet
already cleared, cannot close them, and ``term()`` blocks forever — freezing
whichever test happens to trigger the collection, or pytest's own end-of-session
``gc.collect()``. That turned one failing test into a 50-minute hang.

Here every context is tracked, and after each test any context it left open is
destroyed while its sockets are still reachable (``destroy`` closes them first),
with a warning naming the leaking test.
"""

import warnings
import weakref

import pytest

try:
    import zmq
except ImportError:  # whirlcoms-free envs: nothing to track
    zmq = None

_live: "weakref.WeakSet" = weakref.WeakSet()

if zmq is not None:
    _orig_init = zmq.Context.__init__

    def _tracked_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        _live.add(self)

    zmq.Context.__init__ = _tracked_init  # zmq.asyncio.Context inits via super()


@pytest.fixture(autouse=True)
def _destroy_leaked_zmq_contexts(request):
    before = set(_live)
    yield
    leaked = [c for c in list(_live) if c not in before and not c.closed]
    for ctx in leaked:
        ctx.destroy(linger=0)
    if leaked:
        warnings.warn(
            f"{request.node.nodeid} left {len(leaked)} zmq context(s) open "
            "(missing stop()); destroyed them",
            pytest.PytestWarning,
        )
