from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="LLM Guardrail API")

app.include_router(router)