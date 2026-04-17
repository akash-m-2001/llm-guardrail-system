from fastapi import APIRouter
from app.schemas import PromptRequest, GuardrailResponse, DetectionResult

from app.core.detector import detect_risk
from app.core.sanitization import sanitize_prompt
from app.core.state import get_session_history, append_session_event
from app.core.trust import aggregate_session_risk
from app.core.policy import decide_action
from app.core.ml_detector import ml_detect

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/guardrail", response_model=GuardrailResponse)
def run_guardrail(payload: PromptRequest):

    # 1. Sanitize input
    sanitized_prompt = sanitize_prompt(payload.prompt)

    # 2. Rule-based detection
    rule_detection = detect_risk(payload.prompt)

    # 3. ML detection
    ml_result = ml_detect(payload.prompt)

    # 4. Combine scores
    rule_score = rule_detection["prompt_risk_score"]
    ml_score = ml_result["confidence"]

    final_score = (0.5 * rule_score) + (0.5 * ml_score)

    # Combine categories
    categories = list(set(rule_detection["categories"] + [ml_result["label"]]))

    # 5. Session history
    history = get_session_history(payload.session_id)

    session_risk_score = aggregate_session_risk(
        history=history,
        current_prompt_score=final_score
    )

    # 6. Policy decision
    decision = decide_action(
        prompt_score=final_score,
        session_score=session_risk_score,
        categories=categories
    )

    # 7. Save event
    event = {
        "prompt": payload.prompt,
        "sanitized_prompt": sanitized_prompt,
        "rule_score": rule_score,
        "ml_score": ml_score,
        "final_score": final_score,
        "categories": categories,
        "session_risk_score": session_risk_score,
        "action": decision["action"],
    }

    append_session_event(payload.session_id, event)

    # 8. Response
    return GuardrailResponse(
        session_id=payload.session_id,
        action=decision["action"],
        reason=decision["reason"],
        detection=DetectionResult(
            categories=categories,
            keyword_hits=rule_detection["keyword_hits"],
            prompt_risk_score=final_score,
            sanitized_prompt=sanitized_prompt,
        ),
        session_risk_score=session_risk_score,
        history_count=len(history) + 1,
        metadata={
            "ml_label": ml_result["label"],
            "ml_confidence": ml_result["confidence"]
        }
    )