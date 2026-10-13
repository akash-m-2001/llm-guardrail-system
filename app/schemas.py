from typing import Any

from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    user_id: str | None = None
    prompt: str = Field(min_length=1, max_length=20_000)


class DetectionResult(BaseModel):
    categories: list[str]
    keyword_hits: list[str]
    prompt_risk_score: float
    sanitized_prompt: str
    removed_fragments: list[str] = []


class GuardrailResponse(BaseModel):
    session_id: str
    action: str
    reason: str
    detection: DetectionResult
    session_risk_score: float
    history_count: int
    metadata: dict[str, Any]
