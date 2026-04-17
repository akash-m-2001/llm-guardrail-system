from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class PromptRequest(BaseModel):
    session_id: str
    user_id: Optional[str]
    prompt: str


class DetectionResult(BaseModel):
    categories: List[str]
    keyword_hits: List[str]
    prompt_risk_score: float
    sanitized_prompt: str


class GuardrailResponse(BaseModel):
    session_id: str
    action: str
    reason: str
    detection: DetectionResult
    session_risk_score: float
    history_count: int
    metadata: Dict[str, Any]