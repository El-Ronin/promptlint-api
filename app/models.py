"""Pydantic schemas and domain models for PromptLint API."""

from enum import StrEnum

from pydantic import BaseModel, Field


class SupportedModel(StrEnum):
    """Supported LLM targets for token and cost estimation."""

    GPT_4O = "gpt-4o"
    GEMINI_PRO = "gemini-pro"
    CLAUDE_SONNET = "claude-sonnet"


class HealthResponse(BaseModel):
    """Response payload for the health check endpoint."""

    status: str = Field(..., description="Estado actual del servicio.")
    service: str = Field(..., description="Nombre del servicio.")
    env: str = Field(..., description="Entorno de ejecución configurado.")


class AnalyzeRequest(BaseModel):
    """Request payload for analyzing a prompt."""

    prompt: str = Field(..., min_length=1, description="Texto del prompt a analizar.")
    model: SupportedModel = Field(..., description="Modelo objetivo del prompt.")


class AnalyzeResponse(BaseModel):
    """Response payload containing prompt analysis results."""

    model: SupportedModel
    estimated_tokens: int = Field(..., ge=0)
    estimated_cost_usd: float = Field(..., ge=0.0)
    score: int = Field(..., ge=0, le=100)
    suggestions: list[str]
    alerts: list[str]
    env: str
