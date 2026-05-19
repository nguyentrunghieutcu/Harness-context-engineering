"""
compact/summarizer.py — Auto-format conversation compaction
============================================================
Detects OpenAI / Anthropic / Gemini message format and dispatches
to the appropriate provider-specific compaction logic.
"""

from __future__ import annotations

import json


# ── Format detection ──────────────────────────────────────────────────────────

def _detect_format(messages: list[dict]) -> str:
    """
    Heuristic format detection.

    Returns: "openai" | "anthropic" | "gemini" | "unknown"
    """
    if not messages:
        return "unknown"

    sample = messages[:5]

    # Gemini: uses 'parts' + 'role' (model | user)
    for m in sample:
        if isinstance(m, dict) and "parts" in m:
            return "gemini"

    # Anthropic: uses 'role' (user | assistant) + possible 'content' list
    # with 'type' == 'text' | 'image' | 'tool_use' | 'tool_result'
    for m in sample:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "text", "image", "tool_use", "tool_result"
                ):
                    return "anthropic"

    # OpenAI: uses 'role' with system | developer | tool | function
    openai_exclusive = {"system", "developer", "tool", "function"}
    for m in sample:
        if isinstance(m, dict) and m.get("role") in openai_exclusive:
            return "openai"

    # Fallback: if messages have 'role' at all, treat as OpenAI (most common)
    for m in sample:
        if isinstance(m, dict) and "role" in m:
            return "openai"

    return "unknown"


# ── Provider-specific compaction ──────────────────────────────────────────────

def _compact_openai(
    messages: list[dict],
    retain_turns: int,
    model: str,
    max_recent_tool_tokens: int,
) -> list[dict]:
    from compact.openai import compact_openai_messages
    return compact_openai_messages(
        messages,
        retain_turns=retain_turns,
        model=model,
        max_recent_tool_tokens=max_recent_tool_tokens,
    )


def _compact_anthropic(messages: list[dict], retain_turns: int) -> list[dict]:
    """
    Compact Anthropic-format messages.
    Anthropic does not have a 'system' role in the messages array (it's a
    separate API param), so the body is all user/assistant turns.
    """
    from compact.anthropic import summarize_anthropic_history
    retain_turns = max(0, int(retain_turns))
    retain_msgs = retain_turns * 2
    if len(messages) <= retain_msgs:
        return messages

    if retain_msgs == 0:
        old_msgs = messages
        recent = []
    else:
        old_msgs = messages[:-retain_msgs]
        recent = messages[-retain_msgs:]

    summary = summarize_anthropic_history(old_msgs)

    summary_message = {
        "role": "user",
        "content": (
            f"<conversation_summary>\n"
            f"{summary}\n"
            f"</conversation_summary>"
        ),
    }
    return [summary_message] + recent


def _compact_gemini(messages: list[dict], retain_turns: int) -> list[dict]:
    """
    Compact Gemini-format messages (role + parts[]).
    """
    from compact.gemini import summarize_gemini_history
    retain_turns = max(0, int(retain_turns))
    retain_msgs = retain_turns * 2
    if len(messages) <= retain_msgs:
        return messages

    if retain_msgs == 0:
        old_msgs = messages
        recent = []
    else:
        old_msgs = messages[:-retain_msgs]
        recent = messages[-retain_msgs:]

    summary = summarize_gemini_history(old_msgs)

    summary_message = {
        "role": "user",
        "parts": [
            {
                "text": (
                    f"<conversation_summary>\n"
                    f"{summary}\n"
                    f"</conversation_summary>"
                )
            }
        ],
    }
    return [summary_message] + recent


# ── Public entry-point ────────────────────────────────────────────────────────

def compact_messages(
    messages_json: str,
    retain_turns: int = 2,
    model: str = "",
    max_recent_tool_tokens: int = 800,
) -> list[dict]:
    """
    Parse messages_json, auto-detect format, and compact in-place.

    :param messages_json: JSON array string of conversation messages.
    :param retain_turns:  Number of recent turn-pairs (user+assistant) to keep.
    :param model:         Model hint for provider-specific formatting.
    :param max_recent_tool_tokens: Cap retained OpenAI tool/function payloads.
    :returns: Compacted message list.
    """
    try:
        messages: list[dict] = json.loads(messages_json)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(messages, list) or not messages:
        return []

    fmt = _detect_format(messages)

    if fmt == "openai":
        return _compact_openai(
            messages,
            retain_turns,
            model=model,
            max_recent_tool_tokens=max_recent_tool_tokens,
        )
    elif fmt == "anthropic":
        return _compact_anthropic(messages, retain_turns)
    elif fmt == "gemini":
        return _compact_gemini(messages, retain_turns)
    else:
        # Best-effort: treat as OpenAI
        return _compact_openai(
            messages,
            retain_turns,
            model=model,
            max_recent_tool_tokens=max_recent_tool_tokens,
        )
