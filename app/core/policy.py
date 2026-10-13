PROMPT_THRESHOLD = 0.5
SESSION_BLOCK_THRESHOLD = 0.75


def decide_action(prompt_score, session_score, categories, prior_events=0):
    if "harmful_content" in categories and prompt_score > PROMPT_THRESHOLD:
        return {"action": "block", "reason": "Harmful content detected"}

    # Repeated probing: only once the session has history (on the first prompt
    # the session score is just that prompt's score). Checked before sanitizing
    # so a session that keeps trying injections is blocked, not cleaned each time.
    if prior_events > 0 and session_score > SESSION_BLOCK_THRESHOLD:
        return {"action": "block", "reason": "Session risk too high"}

    if "prompt_injection" in categories and prompt_score > PROMPT_THRESHOLD:
        return {"action": "sanitize", "reason": "Possible prompt injection"}

    return {"action": "allow", "reason": "Safe prompt"}
