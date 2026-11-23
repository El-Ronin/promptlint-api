"""Integration tests for FastAPI endpoints in app.main."""

from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


EXPECTED_ANALYZE_KEYS: set[str] = {
    "model",
    "estimated_tokens",
    "estimated_cost_usd",
    "score",
    "suggestions",
    "alerts",
    "env",
}


def test_health_default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /health should default env to DEV when APP_ENV is unset."""
    monkeypatch.delenv("APP_ENV", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "promptlint-api",
        "env": "DEV",
    }


def test_health_custom_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /health should reflect APP_ENV when configured."""
    monkeypatch.setenv("APP_ENV", "PROD")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "promptlint-api",
        "env": "PROD",
    }


def test_health_whitespace_env_falls_back_to_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /health should fall back to DEV when APP_ENV is whitespace only."""
    monkeypatch.setenv("APP_ENV", "   ")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "promptlint-api",
        "env": "DEV",
    }


@pytest.mark.parametrize(
    "model_name",
    ["gpt-4o", "gemini-pro", "claude-sonnet"],
)
def test_analyze_supported_models(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    """POST /analyze should return the exact contract keys for every supported model."""
    monkeypatch.delenv("APP_ENV", raising=False)
    payload = {
        "prompt": (
            "Actúa como experto en Python. Analiza este fragmento y responde "
            "en formato Markdown con recomendaciones claras."
        ),
        "model": model_name,
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == EXPECTED_ANALYZE_KEYS
    assert body["model"] == model_name
    assert body["estimated_tokens"] > 0
    assert body["estimated_cost_usd"] > 0.0
    assert body["score"] == 100
    assert body["suggestions"] == []
    assert body["alerts"] == []
    assert body["env"] == "DEV"


def test_analyze_detects_sensitive_data_and_custom_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /analyze should flag sensitive values as strings and reflect APP_ENV."""
    monkeypatch.setenv("APP_ENV", "PROD")
    synthetic_secret = "-".join(["sk", "q" * 20])
    payload = {
        "prompt": (
            f"Revisa la cuenta ops@example.com usando {synthetic_secret} "
            "y resume los cambios."
        ),
        "model": "gpt-4o",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == EXPECTED_ANALYZE_KEYS
    assert body["env"] == "PROD"
    assert len(body["alerts"]) == 2
    assert all(isinstance(alert, str) for alert in body["alerts"])
    assert any("correos electrónicos" in alert for alert in body["alerts"])
    assert any("API keys" in alert for alert in body["alerts"])
    assert synthetic_secret not in response.text


def test_analyze_rejects_unsupported_model() -> None:
    """POST /analyze should return 422 for unsupported target models."""
    response = client.post(
        "/analyze",
        json={"prompt": "Resume este texto.", "model": "llama-3"},
    )
    assert response.status_code == 422


def test_analyze_rejects_empty_prompt() -> None:
    """POST /analyze should return 422 when prompt is an empty string."""
    response = client.post(
        "/analyze",
        json={"prompt": "", "model": "gpt-4o"},
    )
    assert response.status_code == 422
