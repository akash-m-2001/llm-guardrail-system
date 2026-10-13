from fastapi import APIRouter

from app.core.detector import detect_risk
from app.core.ml_detector import SAFE, ml_detect
from app.core.policy import decide_action
from app.core.sanitization import normalize, sanitize_prompt
from app.core.state import append_session_event, get_session_history
from app.core.trust import aggregate_session_risk
from app.schemas import DetectionResult, GuardrailResponse, PromptRequest

router = APIRouter()

# Chosen with cross-validation on the training data only (training/choose_threshold.py):
# at 0.8 the model flags ~3% of normal prompts while catching ~70% of attacks.
ML_THRESHOLD = 0.8


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/guardrail", response_model=GuardrailResponse)
def run_guardrail(payload: PromptRequest):
    # 1. Normalize first so obfuscation ("ig​nore") can't hide from either detector.
    text = normalize(payload.prompt)

    # 2. Rule-based and ML detection.
    rules = detect_risk(text)
    ml = ml_detect(text)

    # 3. Hybrid score: rules are precise but brittle, ML generalizes but is noisy.
        # Either detector can flag on its own: the rules when a pattern matches, the
    # model only when it's confident. (The old average, 0.5*rules + 0.5*ML > 0.5,
    # meant the model could never flag anything the rules missed.)
    ml_confident = ml["risk"] > ML_THRESHOLD
    final_score = max(rules["prompt_risk_score"], ml["risk"] if ml_confident else 0.0)

    categories = list(rules["categories"])
    if ml_confident and ml["label"] != SAFE and ml["label"] not in categories:
        categories.append(ml["label"])

    # 4. Session-level risk: repeated probing raises the bar for later prompts.
    history = get_session_history(payload.session_id)
    session_risk_score = aggregate_session_risk(history=history, current_prompt_score=final_score)

    # 5. Policy decision.
    decision = decide_action(
        prompt_score=final_score, session_score=session_risk_score, categories=categories, prior_events=len(history)
    )

    sanitized_prompt, removed = text, []
    if decision["action"] == "sanitize":
        sanitized_prompt, removed = sanitize_prompt(payload.prompt)
        if not sanitized_prompt:
            # The whole prompt was the attack; there is no legitimate request left to forward.
            decision = {"action": "block", "reason": "Prompt consisted only of injected instructions"}
    if decision["action"] == "block":
        # sanitized_prompt means "safe to forward". For a blocked prompt nothing
        # is, so never hand back text a careless client could pass to the LLM.
        sanitized_prompt = ""

    append_session_event(payload.session_id, {
        "rule_score": rules["prompt_risk_score"],
        "ml_risk": ml["risk"],
        "final_score": final_score,
        "categories": categories,
        "session_risk_score": session_risk_score,
        "action": decision["action"],
    })

    return GuardrailResponse(
        session_id=payload.session_id,
        action=decision["action"],
        reason=decision["reason"],
        detection=DetectionResult(
            categories=categories,
            keyword_hits=rules["keyword_hits"],
            prompt_risk_score=round(final_score, 4),
            sanitized_prompt=sanitized_prompt,
            removed_fragments=removed,
        ),
        session_risk_score=round(session_risk_score, 4),
        history_count=len(history) + 1,
        metadata={
            "rule_score": rules["prompt_risk_score"],
            "ml_label": ml["label"],
            "ml_risk": round(ml["risk"], 4),
            "ml_probabilities": {k: round(v, 4) for k, v in ml["probabilities"].items()},
        },
    )
