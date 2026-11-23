"""Pure, stateless prompt analysis logic for PromptLint API."""

import math
import re
from collections.abc import Callable

from app.models import AnalyzeResponse, SupportedModel

CHARS_PER_TOKEN: int = 4
MIN_PROMPT_CHARS: int = 50
MIN_PROMPT_WORDS: int = 8
POINTS_PER_CRITERION: int = 25

MODEL_PRICE_PER_1K_TOKENS: dict[SupportedModel, float] = {
    SupportedModel.GPT_4O: 0.0025,
    SupportedModel.GEMINI_PRO: 0.00125,
    SupportedModel.CLAUDE_SONNET: 0.003,
}

ROLE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:actúa como|actua como|eres un|eres una|como experto|como experta|"
    r"act as|you are a|you are an|as an expert)\b|\b(?:rol|role)\s*:",
    re.IGNORECASE,
)

OUTPUT_FORMAT_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:json|markdown|yaml|csv|xml|tabla|lista|viñetas|vinetas|formato|"
    r"responde en|devuelve en|output format|respond in|return as|table|bullet)\b",
    re.IGNORECASE,
)

ACTION_VERB_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:analiza|genera|resume|explica|crea|escribe|lista|clasifica|compara|"
    r"traduce|diseña|disena|evalúa|evalua|revisa|redacta|analyze|generate|"
    r"summarize|explain|create|write|list|classify|compare|translate|design|"
    r"evaluate|review)\b",
    re.IGNORECASE,
)

VAGUE_TERMS_PATTERN: re.Pattern[str] = re.compile(
    r"(?:\b(?:algo así|algo asi|cosas|whatever|stuff)\b|\betc\.{2,3})",
    re.IGNORECASE,
)

EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.){1,4}[A-Za-z]{2,10}\b"
)

PHONE_PATTERN: re.Pattern[str] = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[\s.-]\d{1,4}[\s.-]|\(\d{2,4}\)[\s.-]?|\d{3}[\s.-])"
    r"\d{3,4}[\s.-]\d{4}(?!\d)"
)

API_KEY_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?i:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*"
    r"['\"]?[A-Za-z0-9_-]{12,64}['\"]?|"
    r"\b(?:sk-[A-Za-z0-9_-]{16,64}|AIza[A-Za-z0-9_-]{20,45}|"
    r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,40})\b"
)

SENSITIVE_DATA_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        EMAIL_PATTERN,
        "Se detectaron posibles correos electrónicos en el prompt.",
    ),
    (
        PHONE_PATTERN,
        "Se detectaron posibles números de teléfono en el prompt.",
    ),
    (
        API_KEY_PATTERN,
        "Se detectaron posibles API keys o tokens en el prompt.",
    ),
)


def estimate_tokens(prompt: str) -> int:
    """Estimate approximate token count using a 4-chars-per-token heuristic."""
    cleaned: str = prompt.strip()
    if not cleaned:
        return 0
    return math.ceil(len(cleaned) / CHARS_PER_TOKEN)


def estimate_cost(tokens: int, model: SupportedModel) -> float:
    """Calculate estimated USD cost for the given token count and model."""
    price_per_1k: float = MODEL_PRICE_PER_1K_TOKENS[model]
    return round((tokens / 1000) * price_per_1k, 6)


def _has_minimum_length(prompt: str) -> bool:
    """Check whether the prompt meets minimum character and word thresholds."""
    cleaned: str = prompt.strip()
    return len(cleaned) >= MIN_PROMPT_CHARS and len(cleaned.split()) >= MIN_PROMPT_WORDS


def _has_role_definition(prompt: str) -> bool:
    """Check whether the prompt defines an explicit assistant role or persona."""
    return bool(ROLE_PATTERN.search(prompt))


def _has_output_format(prompt: str) -> bool:
    """Check whether the prompt specifies an expected output format."""
    return bool(OUTPUT_FORMAT_PATTERN.search(prompt))


def _is_clear(prompt: str) -> bool:
    """Check whether the prompt uses clear action verbs without shouting or vagueness."""
    cleaned: str = prompt.strip()
    has_letters: bool = any(char.isalpha() for char in cleaned)
    if has_letters and cleaned.isupper():
        return False
    has_action: bool = bool(ACTION_VERB_PATTERN.search(cleaned))
    has_vague_terms: bool = bool(VAGUE_TERMS_PATTERN.search(cleaned))
    return has_action and not has_vague_terms


BEST_PRACTICE_RULES: tuple[tuple[Callable[[str], bool], str], ...] = (
    (
        _has_minimum_length,
        "Amplía el prompt con más contexto (al menos 50 caracteres y 8 palabras).",
    ),
    (
        _has_role_definition,
        "Define un rol explícito para el modelo (por ejemplo: 'Actúa como arquitecto de software').",
    ),
    (
        _has_output_format,
        "Especifica el formato de salida esperado (por ejemplo: JSON, Markdown, tabla o lista).",
    ),
    (
        _is_clear,
        "Usa verbos de acción claros y evita términos vagos o escribir todo en mayúsculas.",
    ),
)


def evaluate_best_practices(prompt: str) -> tuple[int, list[str]]:
    """Evaluate prompt quality from 0 to 100 and return actionable suggestions."""
    score: int = 0
    suggestions: list[str] = []

    for check_fn, suggestion in BEST_PRACTICE_RULES:
        if check_fn(prompt):
            score += POINTS_PER_CRITERION
        else:
            suggestions.append(suggestion)

    return score, suggestions


def detect_sensitive_data(prompt: str) -> list[str]:
    """Detect potential sensitive data (emails, phones, API keys) via regex."""
    alerts: list[str] = []

    for pattern, message in SENSITIVE_DATA_RULES:
        matches_count: int = len(pattern.findall(prompt))
        if matches_count > 0:
            alerts.append(f"{message} (coincidencias: {matches_count})")

    return alerts


def analyze_prompt(
    prompt: str,
    model: SupportedModel,
    env: str = "DEV",
) -> AnalyzeResponse:
    """Run full in-memory analysis of a prompt for a target LLM model."""
    estimated_tokens: int = estimate_tokens(prompt)
    estimated_cost_usd: float = estimate_cost(estimated_tokens, model)
    score, suggestions = evaluate_best_practices(prompt)
    alerts: list[str] = detect_sensitive_data(prompt)

    return AnalyzeResponse(
        model=model,
        estimated_tokens=estimated_tokens,
        estimated_cost_usd=estimated_cost_usd,
        score=score,
        suggestions=suggestions,
        alerts=alerts,
        env=env,
    )
