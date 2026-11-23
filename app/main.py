"""FastAPI application entrypoint for PromptLint API."""

import os

from fastapi import FastAPI

from app.analyzer import analyze_prompt
from app.models import AnalyzeRequest, AnalyzeResponse, HealthResponse

DEFAULT_ENV: str = "DEV"
APP_ENV_VAR: str = "APP_ENV"
SERVICE_NAME: str = "promptlint-api"

app = FastAPI(
    title="PromptLint API",
    version="1.0.0",
    description="API en memoria para analizar calidad, costo y datos sensibles en prompts.",
)


def _resolve_environment() -> str:
    """Read the runtime environment name from APP_ENV, defaulting to DEV."""
    raw_env: str = os.getenv(APP_ENV_VAR, DEFAULT_ENV)
    cleaned_env: str = raw_env.strip()
    return cleaned_env or DEFAULT_ENV


@app.get("/health", response_model=HealthResponse, tags=["health"])
def get_health() -> HealthResponse:
    """Return service health status and configured environment."""
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        env=_resolve_environment(),
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
def post_analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a prompt for token cost, best practices score, and sensitive data."""
    return analyze_prompt(
        prompt=payload.prompt,
        model=payload.model,
        env=_resolve_environment(),
    )
