"""Unit tests for pure prompt analysis logic in app.analyzer."""

import pytest

from app.analyzer import (
    analyze_prompt,
    detect_sensitive_data,
    estimate_cost,
    estimate_tokens,
    evaluate_best_practices,
)
from app.models import SupportedModel


def test_estimate_tokens_empty_and_whitespace() -> None:
    """Whitespace-only prompts should estimate zero tokens."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \n\t  ") == 0


def test_estimate_tokens_rounds_up() -> None:
    """Token count should round up using 4 characters per token."""
    assert estimate_tokens("hola") == 1
    assert estimate_tokens("hola!") == 2
    assert estimate_tokens("  12345678  ") == 2


@pytest.mark.parametrize(
    ("tokens", "model", "expected_cost"),
    [
        (0, SupportedModel.GPT_4O, 0.0),
        (1000, SupportedModel.GPT_4O, 0.0025),
        (1000, SupportedModel.GEMINI_PRO, 0.00125),
        (1000, SupportedModel.CLAUDE_SONNET, 0.003),
        (250, SupportedModel.GPT_4O, 0.000625),
    ],
)
def test_estimate_cost_by_model(
    tokens: int, model: SupportedModel, expected_cost: float
) -> None:
    """Estimated cost should match the configured per-model pricing table."""
    assert estimate_cost(tokens, model) == pytest.approx(expected_cost)


def test_evaluate_best_practices_perfect_prompt() -> None:
    """A well-structured prompt should achieve a score of 100 with no suggestions."""
    prompt = (
        "Actúa como arquitecto de software senior. Analiza los riesgos de "
        "seguridad de este microservicio y responde en formato JSON."
    )
    score, suggestions = evaluate_best_practices(prompt)
    assert score == 100
    assert suggestions == []


def test_evaluate_best_practices_minimal_unclear_prompt() -> None:
    """A short all-caps prompt without role, format, or clarity should score 0."""
    score, suggestions = evaluate_best_practices("HOLA")
    assert score == 0
    assert len(suggestions) == 4


def test_evaluate_best_practices_missing_single_criteria() -> None:
    """Missing each single criterion should deduct 25 points and emit 1 suggestion."""
    missing_role = (
        "Analiza la arquitectura de este microservicio web con alto tráfico "
        "y devuelve los hallazgos en formato Markdown."
    )
    score_role, suggestions_role = evaluate_best_practices(missing_role)
    assert score_role == 75
    assert len(suggestions_role) == 1
    assert "rol" in suggestions_role[0].lower()

    missing_format = (
        "Actúa como arquitecto cloud senior y analiza los cuellos de botella "
        "principales de este servicio distribuido en producción."
    )
    score_fmt, suggestions_fmt = evaluate_best_practices(missing_format)
    assert score_fmt == 75
    assert len(suggestions_fmt) == 1
    assert "formato" in suggestions_fmt[0].lower()

    vague_prompt = (
        "Actúa como ingeniero DevOps, revisa el pipeline de CI/CD, agrega "
        "pruebas y devuelve una lista con cosas etc..."
    )
    score_vague, suggestions_vague = evaluate_best_practices(vague_prompt)
    assert score_vague == 75
    assert len(suggestions_vague) == 1
    assert "claros" in suggestions_vague[0].lower()

    no_action_verb = (
        "Actúa como ingeniero DevOps senior para el reporte de métricas "
        "mensuales en formato tabla."
    )
    score_verb, suggestions_verb = evaluate_best_practices(no_action_verb)
    assert score_verb == 75
    assert len(suggestions_verb) == 1


def test_evaluate_best_practices_too_short_only() -> None:
    """Short prompts with role, verb, and format should still lose length points."""
    short_prompt = "Act as QA, list bugs in JSON"
    score, suggestions = evaluate_best_practices(short_prompt)
    assert score == 75
    assert len(suggestions) == 1
    assert "50 caracteres" in suggestions[0]


def test_detect_sensitive_data_clean_prompt() -> None:
    """Prompts without sensitive data should produce no alerts."""
    prompt = "Actúa como tutor de Python y explica qué es una función pura en lista."
    assert detect_sensitive_data(prompt) == []


def test_detect_sensitive_data_emails() -> None:
    """Email addresses should be detected and reported as descriptive alerts."""
    prompt = "Envía el resumen a alice.dev@example.com y bob+ops@mail.example.org."
    alerts = detect_sensitive_data(prompt)
    assert len(alerts) == 1
    assert isinstance(alerts[0], str)
    assert "correos electrónicos" in alerts[0]
    assert "(coincidencias: 2)" in alerts[0]


@pytest.mark.parametrize(
    "phone_sample",
    [
        "+56 9 1234 5678",
        "+1-555-123-4567",
        "(555) 123-4567",
        "555-123-4567",
    ],
)
def test_detect_sensitive_data_phones(phone_sample: str) -> None:
    """Common phone number formats should trigger a descriptive phone alert."""
    alerts = detect_sensitive_data(f"Contactar al número {phone_sample} por soporte.")
    assert len(alerts) == 1
    assert "teléfono" in alerts[0]
    assert "(coincidencias: 1)" in alerts[0]


def test_detect_sensitive_data_api_keys() -> None:
    """Synthetic token patterns and key assignments should trigger an API key alert."""
    synthetic_sk = "-".join(["sk", "t" * 20])
    synthetic_gh = "_".join(["ghp", "b" * 24])
    synthetic_assigned = f"api_key={'x' * 16}"
    prompt = (
        f"Usa {synthetic_sk} o {synthetic_gh} y configura {synthetic_assigned} "
        "antes de ejecutar."
    )
    alerts = detect_sensitive_data(prompt)
    assert len(alerts) == 1
    assert "API keys" in alerts[0]
    assert "(coincidencias: 3)" in alerts[0]


def test_detect_sensitive_data_combined_categories() -> None:
    """Prompts containing multiple sensitive categories should return all alerts."""
    synthetic_aiza = "AIza" + ("z" * 22)
    prompt = (
        "Contacto: sec@empresa.cl, tel +56 9 8765 4321, "
        f"credencial {synthetic_aiza}."
    )
    alerts = detect_sensitive_data(prompt)
    assert len(alerts) == 3
    assert "correos electrónicos" in alerts[0]
    assert "teléfono" in alerts[1]
    assert "API keys" in alerts[2]


def test_analyze_prompt_end_to_end() -> None:
    """analyze_prompt should aggregate tokens, cost, score, suggestions, alerts, and env."""
    prompt = (
        "Act as a security reviewer. Analyze this config for user@example.com "
        "and return findings in JSON."
    )
    result_default = analyze_prompt(prompt=prompt, model=SupportedModel.CLAUDE_SONNET)
    assert result_default.model == SupportedModel.CLAUDE_SONNET
    assert result_default.estimated_tokens > 0
    assert result_default.estimated_cost_usd > 0.0
    assert result_default.score == 100
    assert result_default.suggestions == []
    assert len(result_default.alerts) == 1
    assert "correos electrónicos" in result_default.alerts[0]
    assert result_default.env == "DEV"

    result_staging = analyze_prompt(
        prompt=prompt,
        model=SupportedModel.CLAUDE_SONNET,
        env="STAGING",
    )
    assert result_staging.env == "STAGING"
