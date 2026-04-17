def aggregate_session_risk(history, current_prompt_score):
    if not history:
        return current_prompt_score

    prev_scores = [item["final_score"] for item in history]
    avg_prev = sum(prev_scores) / len(prev_scores)

    return min(0.6 * current_prompt_score + 0.4 * avg_prev, 1.0)