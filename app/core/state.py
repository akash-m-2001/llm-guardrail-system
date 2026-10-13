"""Per-session event history, bounded and thread-safe.

Kept in process memory: fine for a single instance. Running several replicas
would need a shared store (e.g. Redis) so a session's history follows it.
"""
import threading
from collections import OrderedDict, deque

MAX_EVENTS_PER_SESSION = 50  # risk aggregation only needs recent history
MAX_SESSIONS = 10_000        # least-recently-used sessions are evicted beyond this

_sessions: "OrderedDict[str, deque]" = OrderedDict()
_lock = threading.Lock()


def get_session_history(session_id: str) -> list[dict]:
    with _lock:
        history = _sessions.get(session_id)
        return list(history) if history else []


def append_session_event(session_id: str, event: dict) -> None:
    with _lock:
        history = _sessions.get(session_id)
        if history is None:
            history = _sessions[session_id] = deque(maxlen=MAX_EVENTS_PER_SESSION)
        history.append(event)
        _sessions.move_to_end(session_id)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


def clear_sessions() -> None:
    with _lock:
        _sessions.clear()
