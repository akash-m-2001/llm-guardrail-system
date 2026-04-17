_sessions = {}

def get_session_history(session_id: str):
    return _sessions.get(session_id, [])

def append_session_event(session_id: str, event: dict):
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append(event)