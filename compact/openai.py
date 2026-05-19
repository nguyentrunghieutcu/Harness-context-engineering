"""
compact/openai.py — OpenAI / Codex format compaction
=====================================================
Handles both Chat Completions format and legacy Completions format.

Supported model families:
  - gpt-4o, gpt-4, gpt-3.5-turbo  (chat)
  - o1, o3, o3-mini, o4-mini       (reasoning)
  - codex (code-davinci-002, etc.)  (legacy completions → lifted to chat)
  - gpt-4.1, gpt-4.5               (latest chat)

Message roles recognised:
  system | user | assistant | tool | function | developer
"""

from __future__ import annotations

import re
from typing import Any


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    """
    Flatten OpenAI content to plain text.

    OpenAI content can be:
      - str
      - list[dict]  (multi-modal: [{type: "text", text: "..."}, {type: "image_url", ...}])
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("type", "")
                if t == "text":
                    parts.append(part.get("text", ""))
                elif t == "image_url":
                    parts.append("[image]")
                elif t == "input_audio":
                    parts.append("[audio]")
                elif t == "refusal":
                    parts.append(f"[refusal: {part.get('refusal', '')}]")
        return " ".join(parts)
    return str(content) if content is not None else ""


def _is_openai_format(messages: list[dict]) -> bool:
    """
    Heuristic: at least one message has 'role' key matching OpenAI roles.
    """
    openai_roles = {"system", "user", "assistant", "tool",
                    "function", "developer"}
    for m in messages[:5]:
        if isinstance(m, dict) and m.get("role") in openai_roles:
            return True
    return False


def _summarize_chunk(messages: list[dict]) -> str:
    """
    Build a structured summary of the omitted message chunk.
    Extracts user intents + assistant key responses without calling an LLM.
    """
    user_intents: list[str] = []
    assistant_actions: list[str] = []
    tool_calls_seen: list[str] = []
    code_snippets: int = 0

    for m in messages:
        role = m.get("role", "")
        text = _extract_text(m.get("content") or "").strip()

        if role == "system":
            continue  # system prompts handled separately

        elif role in ("user", "developer"):
            if text:
                # Condense: first sentence or first 120 chars
                first = re.split(r"[.!?\n]", text)[0].strip()
                if first:
                    user_intents.append(first[:120])

        elif role == "assistant":
            # Count code blocks as signals
            code_snippets += len(re.findall(r"```", text)) // 2

            # Tool calls in structured format
            tool_calls = m.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function", {}).get("name", "")
                if fn:
                    tool_calls_seen.append(fn)

            if text and not tool_calls:
                first = re.split(r"[.!?\n]", text)[0].strip()
                if first:
                    assistant_actions.append(first[:120])

        elif role == "tool":
            # Tool responses: just note the tool name
            name = m.get("name") or ""
            if name:
                tool_calls_seen.append(f"{name}→result")

        elif role == "function":
            name = m.get("name") or ""
            if name:
                tool_calls_seen.append(f"{name}→result")

    lines: list[str] = []
    if user_intents:
        lines.append(f"User requests: {'; '.join(user_intents[:5])}")
    if assistant_actions:
        lines.append(f"Assistant actions: {'; '.join(assistant_actions[:5])}")
    if tool_calls_seen:
        unique_calls = list(dict.fromkeys(tool_calls_seen))[:8]
        lines.append(f"Tools used: {', '.join(unique_calls)}")
    if code_snippets:
        lines.append(f"Code blocks generated: {code_snippets}")

    return "\n".join(lines) if lines else (
        "Prior conversation turns omitted for context length optimization."
    )


def _cap_summary(text: str, max_tokens: int) -> str:
    """
    Truncate summary to stay within max_tokens.
    Uses tiktoken if available, else len/3.5 heuristic.
    """
    try:
        import tiktoken as _tiktoken
        enc = _tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        # Decode truncated token list back to string
        return enc.decode(tokens[:max_tokens]) + "..."
    except Exception:
        # Fallback: approx 3.5 chars/token
        char_limit = max_tokens * 3
        return text[:char_limit] + ("..." if len(text) > char_limit else "")


def _summary_token_cap(model: str) -> int:
    """
    Max tokens for the <conversation_summary> block, by model family.
    Codex legacy is very small; GPT-4.x/o-series are generous; large context models can handle more.
    """
    ml = model.lower()
    # Legacy Codex (8k context) — keep summary tiny
    if any(kw in ml for kw in ("code-davinci", "code-cushman", "codex-001", "codex-002")):
        return 100
    # codex-mini / gpt-4o-mini / small models
    if any(kw in ml for kw in ("codex-mini", "gpt-4o-mini", "gpt-3.5", "o1-mini", "o3-mini")):
        return 200
    # Standard GPT-4.x, o1, o3, o4-mini, gpt-5.x
    if any(kw in ml for kw in ("gpt-4", "gpt-5", "o1", "o3", "o4")):
        return 350
    # Large-context models (Claude, Gemini) — be generous
    return 500


def _count_tokens(text: str) -> int:
    try:
        import tiktoken as _tiktoken
        enc = _tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return max(1, int(len(text) / 3.5)) if text else 0


def _cap_text_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or _count_tokens(text) <= max_tokens:
        return text
    try:
        import tiktoken as _tiktoken
        enc = _tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text, disallowed_special=())
        return enc.decode(tokens[:max_tokens]) + "..."
    except Exception:
        return text[:max_tokens * 3] + "..."


def _compact_recent_tool_outputs(
    messages: list[dict],
    max_recent_tool_tokens: int,
) -> list[dict]:
    """
    Keep recent message structure intact, but cap very large tool/function
    payloads so retained turns cannot carry raw logs or diffs indefinitely.
    """
    if max_recent_tool_tokens <= 0:
        return messages

    compacted: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role not in ("tool", "function"):
            compacted.append(message)
            continue

        content = message.get("content")
        text = _extract_text(content)
        if _count_tokens(text) <= max_recent_tool_tokens:
            compacted.append(message)
            continue

        capped = _cap_text_tokens(text, max_recent_tool_tokens)
        next_message = dict(message)
        next_message["content"] = (
            "[tool output compacted; original exceeded "
            f"{max_recent_tool_tokens} tokens]\n{capped}"
        )
        compacted.append(next_message)

    return compacted


# ── Public API ────────────────────────────────────────────────────────────────

def summarize_openai_history(
    old_messages: list[dict],
    model: str = "",
) -> str:
    """
    Produce a compact plain-text summary of old_messages.
    Used by compact_messages() to build the <conversation_summary> block.

    :param old_messages: Messages being replaced (already sliced — excludes
                         the system prompt and recent retained turns).
    :param model:        Target model string (for any model-specific tweaks).
    :returns: Plain-text summary injected into the summary message.
    """
    if not old_messages:
        return "No prior conversation history."
    return _summarize_chunk(old_messages)


def compact_openai_messages(
    messages: list[dict],
    retain_turns: int = 2,
    model: str = "",
    max_recent_tool_tokens: int = 800,
) -> list[dict]:
    """
    Compact an OpenAI Chat Completions message array.

    Strategy:
      1. Separate system / developer messages (always kept first).
      2. Keep the last `retain_turns` user+assistant exchange pairs intact.
      3. Replace everything in between with a single summary user message
         containing a <conversation_summary> XML block.
      4. For reasoning models (o1/o3/o4) inject summary as a 'user' role
         (they don't support 'system' injections in all contexts).

    :param messages:      Full message list in OpenAI format.
    :param retain_turns:  Number of recent exchange pairs to keep.
    :param model:         Model name for format tuning.
    :param max_recent_tool_tokens: Cap retained tool/function payloads.
    :returns: Compacted message list.
    """
    if not messages:
        return messages

    # ── 1. Peel off leading system/developer messages ─────────────────────────────
    prefix: list[dict] = []
    body: list[dict] = []
    in_prefix = True
    for m in messages:
        if in_prefix and m.get("role") in ("system", "developer"):
            prefix.append(m)
        else:
            in_prefix = False
            body.append(m)

    # ── 2. Determine retain window ────────────────────────────────────────────
    # Each "turn" = one user message + one assistant message (2 items)
    retain_turns = max(0, int(retain_turns))
    retain_msgs = retain_turns * 2
    if len(body) <= retain_msgs:
        return prefix + _compact_recent_tool_outputs(
            body, max_recent_tool_tokens=max_recent_tool_tokens)

    if retain_msgs == 0:
        old_body = body
        recent_body = []
    else:
        old_body = body[:-retain_msgs]
        recent_body = body[-retain_msgs:]

    # ── 3. Build + cap summary ─────────────────────────────────────────────────
    raw_summary = summarize_openai_history(old_body, model=model)
    max_tok = _summary_token_cap(model)
    summary_text = _cap_summary(raw_summary, max_tok)

    is_reasoning = any(
        tag in model.lower() for tag in ("o1", "o3", "o4")
    )
    # All injections use 'user' role (safe for both chat and reasoning models)
    _ = is_reasoning  # kept for future per-model tweaks

    summary_message: dict = {
        "role": "user",
        "content": (
            f"<conversation_summary>\n"
            f"{summary_text}\n"
            f"</conversation_summary>"
        ),
    }

    recent_body = _compact_recent_tool_outputs(
        recent_body, max_recent_tool_tokens=max_recent_tool_tokens)

    return prefix + [summary_message] + recent_body
