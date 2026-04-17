def decide_action(prompt_score, session_score, categories):

    if "harmful_content" in categories and prompt_score > 0.5:
        return {"action": "block", "reason": "Harmful content detected"}

    if "prompt_injection" in categories and prompt_score > 0.5:
        return {"action": "sanitize", "reason": "Possible prompt injection"}

    if session_score > 0.75:
        return {"action": "block", "reason": "Session risk too high"}

    return {"action": "allow", "reason": "Safe prompt"}