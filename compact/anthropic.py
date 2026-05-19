"""
compact/anthropic.py — Anthropic-specific format helpers
=========================================================
Provides a standalone function for summarizing Anthropic-format
old messages. The main compaction entry-point is in summarizer.py.

Anthropic content blocks may be:
  - str
  - list[dict] with type: "text" | "image" | "tool_use" | "tool_result"
"""

from __future__ import annotations

import re
from typing import Any


def _extract_text(content: Any) -> str:
    """Flatten Anthropic content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type", "")
            if t == "text":
                parts.append(block.get("text", ""))
            elif t == "image":
                parts.append("[image]")
            elif t == "tool_use":
                parts.append(f"[tool: {block.get('name', '')}]")
            elif t == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str):
                    parts.append(f"[tool_result: {inner[:80]}]")
        return " ".join(parts)
    return str(content) if content is not None else ""


def summarize_anthropic_history(
    old_messages: list[dict],
    model: str = "",
) -> str:
    """
    Produce a compact plain-text summary of old_messages.

    :param old_messages: Messages being replaced (excludes retained turns).
    :param model:        Target Claude model string (for future tuning).
    :returns: Plain-text summary injected into the summary message.
    """
    if not old_messages:
        return "No prior conversation history."

    user_intents: list[str] = []
    assistant_actions: list[str] = []
    tool_calls_seen: list[str] = []

    for m in old_messages:
        role = m.get("role", "")
        text = _extract_text(m.get("content") or "").strip()

        if role == "user" and text:
            first = re.split(r"[.!?\n]", text)[0].strip()
            if first:
                user_intents.append(first[:120])

        elif role == "assistant":
            content = m.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_calls_seen.append(block.get("name", ""))
                        elif block.get("type") == "text":
                            t = block.get("text", "").strip()
                            if t:
                                first = re.split(r"[.!?\n]", t)[0].strip()
                                if first:
                                    assistant_actions.append(first[:120])
            elif text:
                first = re.split(r"[.!?\n]", text)[0].strip()
                if first:
                    assistant_actions.append(first[:120])

    lines: list[str] = []
    if user_intents:
        lines.append(f"User requests: {'; '.join(user_intents[:5])}")
    if assistant_actions:
        lines.append(f"Assistant actions: {'; '.join(assistant_actions[:5])}")
    if tool_calls_seen:
        unique = list(dict.fromkeys(tool_calls_seen))[:8]
        lines.append(f"Tools used: {', '.join(unique)}")

    return "\n".join(lines) if lines else (
        "Prior conversation turns omitted for context length optimization (Anthropic)."
    )
