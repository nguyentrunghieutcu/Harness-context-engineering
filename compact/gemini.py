"""
compact/gemini.py — Gemini-specific format helpers
===================================================
Gemini uses role + parts[] instead of role + content.
Parts can be: text str, inline_data (images), function_call, function_response.
"""

from __future__ import annotations

import re
from typing import Any


def _extract_text_from_parts(parts: Any) -> str:
    """Flatten Gemini parts array to plain text."""
    if not isinstance(parts, list):
        return str(parts) if parts else ""
    texts: list[str] = []
    for p in parts:
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, dict):
            if "text" in p:
                texts.append(p["text"])
            elif p.get("inline_data"):
                texts.append("[inline_data]")
            elif p.get("function_call"):
                fn = p["function_call"].get("name", "")
                texts.append(f"[function_call: {fn}]")
            elif p.get("function_response"):
                fn = p["function_response"].get("name", "")
                texts.append(f"[function_response: {fn}]")
    return " ".join(texts)


def summarize_gemini_history(
    old_messages: list[dict],
    model: str = "",
) -> str:
    """
    Produce a compact plain-text summary of old_messages in Gemini format.

    :param old_messages: Messages being replaced (excludes retained turns).
    :param model:        Target Gemini model string.
    :returns: Plain-text summary.
    """
    if not old_messages:
        return "No prior conversation history."

    user_intents: list[str] = []
    model_actions: list[str] = []
    fn_calls_seen: list[str] = []

    for m in old_messages:
        role = m.get("role", "")  # "user" | "model"
        parts = m.get("parts", [])

        # Collect function calls
        for p in parts if isinstance(parts, list) else []:
            if isinstance(p, dict):
                if p.get("function_call"):
                    fn = p["function_call"].get("name", "")
                    if fn:
                        fn_calls_seen.append(fn)

        text = _extract_text_from_parts(parts).strip()

        if role == "user" and text:
            first = re.split(r"[.!?\n]", text)[0].strip()
            if first:
                user_intents.append(first[:120])
        elif role == "model" and text:
            first = re.split(r"[.!?\n]", text)[0].strip()
            if first:
                model_actions.append(first[:120])

    lines: list[str] = []
    if user_intents:
        lines.append(f"User requests: {'; '.join(user_intents[:5])}")
    if model_actions:
        lines.append(f"Model actions: {'; '.join(model_actions[:5])}")
    if fn_calls_seen:
        unique = list(dict.fromkeys(fn_calls_seen))[:8]
        lines.append(f"Functions called: {', '.join(unique)}")

    return "\n".join(lines) if lines else (
        "Prior conversation turns omitted for context length optimization (Gemini)."
    )
