"""
context/budgeting.py — Dynamic token budget per model
======================================================
Covers all major model families:
  - Claude (Anthropic):  sonnet, haiku, opus
  - Gemini (Google):     2.5-pro, 2.5-flash, 1.5-pro
  - OpenAI Chat:         gpt-4o, gpt-4.1, gpt-4.5, gpt-4, gpt-3.5
  - OpenAI Reasoning:    o1, o1-mini, o3, o3-mini, o4-mini
  - OpenAI Codex:        codex-mini-latest, code-davinci-002, gpt-4o-mini
  - Generic fallback:    128k context
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenBudget:
    model_name: str
    context_window: int
    reserved_output: int
    reserved_reasoning: int   # hidden CoT tokens (o1/o3/o4 family)
    reserved_tools: int
    inject_budget: int        # context_window - all reserves


# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (context_window, reserved_output, reserved_reasoning, reserved_tools)
_MODEL_TABLE: dict[str, tuple[int, int, int, int]] = {
    # ── Claude ──────────────────────────────────────────────────────────────
    "claude-opus":          (200_000, 15_000,  0,      10_000),
    "claude-sonnet":        (200_000, 30_000,  25_000, 10_000),
    "claude-haiku":         (200_000, 10_000,  0,       8_000),

    # ── Gemini ──────────────────────────────────────────────────────────────
    "gemini-2.5-pro":       (2_000_000, 64_000, 0,     20_000),
    "gemini-2.5-flash":     (1_000_000, 32_000, 0,     16_000),
    "gemini-1.5-pro":       (2_000_000, 8_192,  0,     20_000),
    "gemini-1.5-flash":     (1_000_000, 8_192,  0,     16_000),
    "gemini":               (1_000_000, 32_000, 0,     16_000),  # generic

    # ── OpenAI GPT-5 series ─────────────────────────────────────────────────
    # Context windows / output limits are estimates based on API announcements.
    # Update when OpenAI releases official specs.
    "gpt-5.5":              (1_000_000, 32_768,  0,     16_000),  # multimodal flagship
    "gpt-5.4":              (  512_000, 32_768,  0,     12_000),
    "gpt-5-mini":           (  256_000, 16_384,  0,      8_000),
    "gpt-5":                (  512_000, 32_768,  0,     12_000),  # base gpt-5

    # ── OpenAI Chat ─────────────────────────────────────────────────────────
    "gpt-4.5":              (128_000, 16_384,  0,      8_000),
    "gpt-4.1":              (128_000, 16_384,  0,      8_000),
    "gpt-4o":               (128_000, 16_384,  0,      8_000),
    "gpt-4o-mini":          (128_000,  4_096,  0,      4_000),
    "gpt-4-turbo":          (128_000, 16_384,  0,      8_000),
    "gpt-4":                ( 32_768,  4_096,  0,      4_000),
    "gpt-3.5-turbo":        ( 16_385,  4_096,  0,      3_000),

    # ── OpenAI Reasoning (o-series) ──────────────────────────────────────────
    # reserved_reasoning captures the hidden CoT budget estimate
    "o4-mini":              (200_000, 16_384, 64_000,  8_000),
    "o3":                   (200_000, 16_384, 80_000,  8_000),
    "o3-mini":              (200_000, 16_384, 64_000,  8_000),
    "o1":                   (200_000, 16_384, 80_000,  8_000),
    "o1-mini":              (128_000,  4_096, 32_000,  4_000),
    "o1-preview":           (128_000,  4_096, 32_000,  4_000),

    # ── OpenAI Codex ────────────────────────────────────────────────────────
    # codex-mini-latest: the new hosted reasoning code model (May 2025)
    "codex-mini-latest":    (200_000, 16_384, 64_000,  8_000),
    "codex-mini":           (200_000, 16_384, 64_000,  8_000),
    # Legacy completion-only Codex (deprecated but still referenced)
    "code-davinci-002":     (  8_001,  4_096,  0,      1_000),
    "code-cushman-001":     (  2_048,  2_048,  0,        500),

    # ── Generic fallback ────────────────────────────────────────────────────
    "default":              (128_000,  8_192,  0,       6_000),
}


def _match(model: str) -> tuple[int, int, int, int]:
    """
    Match a model string to the closest entry in _MODEL_TABLE.
    Checks exact → prefix → keyword substring → default.
    """
    ml = model.lower().strip()

    # 1. Exact match
    if ml in _MODEL_TABLE:
        return _MODEL_TABLE[ml]

    # 2. Prefix match (longest wins)
    candidates = [k for k in _MODEL_TABLE if ml.startswith(k)]
    if candidates:
        return _MODEL_TABLE[max(candidates, key=len)]

    # 3. Keyword substring scan (ordered by specificity)
    keyword_order = [
        "codex-mini-latest", "codex-mini", "codex",
        "code-davinci", "code-cushman",
        "o4-mini", "o3-mini", "o1-mini", "o1-preview",
        "o4", "o3", "o1",
        # GPT-5 must come before gpt-4 to avoid partial match
        "gpt-5.5", "gpt-5.4", "gpt-5-mini", "gpt-5",
        "gpt-4.5", "gpt-4.1", "gpt-4o-mini", "gpt-4o",
        "gpt-4-turbo", "gpt-4",
        "gpt-3.5",
        "gemini-2.5-pro", "gemini-2.5-flash",
        "gemini-1.5-pro", "gemini-1.5-flash",
        "gemini",
        "claude-opus", "claude-haiku", "claude-sonnet", "claude",
    ]
    for kw in keyword_order:
        if kw in ml:
            return _MODEL_TABLE.get(kw, _MODEL_TABLE["default"])

    return _MODEL_TABLE["default"]


def get_budget(model: str) -> TokenBudget:
    """
    Return a TokenBudget for the given model string.

    :param model: Any model identifier string (case-insensitive).
    :returns: TokenBudget with inject_budget = usable context tokens.
    """
    ctx, out, res, tools = _match(model)
    inject = max(ctx - out - res - tools, 0)
    return TokenBudget(
        model_name=model,
        context_window=ctx,
        reserved_output=out,
        reserved_reasoning=res,
        reserved_tools=tools,
        inject_budget=inject,
    )
