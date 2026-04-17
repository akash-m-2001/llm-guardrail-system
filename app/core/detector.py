def detect_risk(prompt: str):
    prompt_lower = prompt.lower()

    categories = []
    keyword_hits = []
    score = 0.0

    if "ignore" in prompt_lower or "reveal" in prompt_lower:
        categories.append("prompt_injection")
        keyword_hits.append("injection_pattern")
        score += 0.6

    if "hack" in prompt_lower or "malware" in prompt_lower:
        categories.append("harmful_content")
        keyword_hits.append("harmful_pattern")
        score += 0.7

    score = min(score, 1.0)

    return {
        "categories": categories,
        "keyword_hits": keyword_hits,
        "prompt_risk_score": score,
    }