"""
context/assembler.py — Provider-aware context assembly
=======================================================
Formats retrieved chunks + memory entries into a system prompt.

Format strategy by model:
  - Claude (Anthropic):  XML  — <retrieved_context> / <long_term_memory>
  - Gemini (Google):     JSON — wrapped in ```json fence
  - OpenAI Chat (GPT):   Markdown — fenced code blocks, clean headings
  - OpenAI Codex:        Compact Markdown — minimise overhead tokens;
                          only emit file path + critical code snippet
  - OpenAI Reasoning:    Markdown — same as chat but no `<system>` injection
                          (summary injected as user message by compactor)
"""

from __future__ import annotations

import json


# ── Memory formatting helper ──────────────────────────────────────────────────

def _format_memory_compact(entries: list[dict]) -> str:
    """
    Compact single-line memory format — saves ~50 tokens per entry vs XML.
    Format: [type|key] value
    """
    if not entries:
        return ""
    lines = [f"[{m['type']}|{m['key']}] {m['value']}" for m in entries]
    return "\n".join(lines)


def _format_memory_xml(entries: list[dict]) -> str:
    """Full XML memory block — for Claude structured mode."""
    if not entries:
        return ""
    lines = []
    for m in entries:
        lines.append(
            f"  <memory key='{m['key']}' type='{m['type']}'>"
            f"{m['value']}</memory>"
        )
    return "\n".join(lines)


# ── Model family detection ────────────────────────────────────────────────────

def _is_gemini(model: str) -> bool:
    return "gemini" in model.lower()


def _is_claude(model: str) -> bool:
    return "claude" in model.lower() or "anthropic" in model.lower()


def _is_codex(model: str) -> bool:
    ml = model.lower()
    return any(kw in ml for kw in (
        "codex", "code-davinci", "code-cushman",
    ))


def _is_reasoning(model: str) -> bool:
    """OpenAI o-series reasoning models."""
    ml = model.lower()
    return any(ml.startswith(p) or f"-{p}" in ml for p in (
        "o1", "o3", "o4",
    ))


def _is_openai(model: str) -> bool:
    ml = model.lower()
    return any(kw in ml for kw in ("gpt-", "o1", "o3", "o4", "codex", "code-"))


# ── Formatters ────────────────────────────────────────────────────────────────

class ContextAssembler:
    def __init__(self) -> None:
        pass

    def assemble(
        self,
        base_prompt: str,
        chunks: list,
        memory_entries: list,
        query: str,
        output_mode: str,
        model: str,
    ) -> str:
        """
        Assemble the enriched system prompt.

        :param base_prompt:     Original system prompt.
        :param chunks:          Retrieved code/doc chunks.
        :param memory_entries:  Memory entries from long-term store.
        :param query:           Current user task.
        :param output_mode:     "concise" | "structured" | "code_only" | "minimal"
        :param model:           Target model string.
        :returns: Assembled prompt string.
        """
        if _is_gemini(model):
            return self._assemble_gemini(
                base_prompt, chunks, memory_entries, query, output_mode)

        if _is_claude(model):
            return self._assemble_claude(
                base_prompt, chunks, memory_entries, query, output_mode)

        if _is_codex(model):
            return self._assemble_codex(
                base_prompt, chunks, memory_entries, query, output_mode)

        if _is_openai(model) or _is_reasoning(model):
            return self._assemble_openai(
                base_prompt, chunks, memory_entries, query, output_mode,
                reasoning=_is_reasoning(model))

        # Fallback → OpenAI markdown
        return self._assemble_openai(
            base_prompt, chunks, memory_entries, query, output_mode)

    # ── Claude XML ────────────────────────────────────────────────────────────
    def _assemble_claude(
        self,
        base_prompt: str,
        chunks: list,
        memory_entries: list,
        query: str,
        output_mode: str,
    ) -> str:
        # Memory: XML only for structured mode; compact otherwise
        if output_mode == "structured":
            mem_block = _format_memory_xml(memory_entries)
            mem_section = f"<long_term_memory>\n{mem_block}\n</long_term_memory>" if mem_block else ""
        else:
            mem_block = _format_memory_compact(memory_entries)
            mem_section = f"<memory>\n{mem_block}\n</memory>" if mem_block else ""

        chunk_xml = ""
        for c in chunks:
            if c.content.startswith("<context>"):  # already compressed
                chunk_xml += c.content + "\n"
            else:
                chunk_xml += (
                    f"  <chunk path='{c.path}' symbol='{c.symbol}'>"
                    f"<content>{c.content}</content></chunk>\n"
                )

        parts = [base_prompt]
        if chunk_xml:
            parts.append(f"<retrieved_context>\n{chunk_xml}</retrieved_context>")
        if mem_section:
            parts.append(mem_section)
        if query:
            parts.append(f"<task>\n{query}\n</task>")
        return "\n\n".join(parts)

    # ── Gemini JSON ───────────────────────────────────────────────────────────
    def _assemble_gemini(
        self,
        base_prompt: str,
        chunks: list,
        memory_entries: list,
        query: str,
        output_mode: str,
    ) -> str:
        context_data = {
            "query": query,
            "chunks": [
                {"path": c.path, "symbol": c.symbol, "content": c.content}
                for c in chunks
            ],
            "memory": memory_entries,
        }
        ctx_str = json.dumps(context_data, ensure_ascii=False, indent=2)
        return f"{base_prompt}\n\n[CONTEXT]\n```json\n{ctx_str}\n```"

    # ── OpenAI Markdown (Chat + Reasoning) ────────────────────────────────────
    def _assemble_openai(
        self,
        base_prompt: str,
        chunks: list,
        memory_entries: list,
        query: str,
        output_mode: str,
        reasoning: bool = False,
    ) -> str:
        """
        Clean Markdown format for GPT / o-series.

        Uses fenced code blocks per chunk (language inferred from path).
        Memory entries in compact single-line format to minimise token overhead.
        """
        parts = [base_prompt]

        if chunks:
            parts.append("## Retrieved Context")
            for c in chunks:
                lang = _infer_language(c.path)
                header = f"### `{c.path}`"
                if c.symbol and c.symbol != "module":
                    header += f" — `{c.symbol}`"
                if output_mode == "minimal":
                    snippet = "\n".join(c.content.splitlines()[:20])
                    parts.append(f"{header}\n```{lang}\n{snippet}\n```")
                else:
                    parts.append(f"{header}\n```{lang}\n{c.content}\n```")

        if memory_entries:
            mem_str = _format_memory_compact(memory_entries)
            parts.append(f"## Memory\n{mem_str}")

        if query:
            parts.append(f"## Task\n{query}")

        return "\n\n".join(parts)

    # ── Codex Compact Markdown ────────────────────────────────────────────────
    def _assemble_codex(
        self,
        base_prompt: str,
        chunks: list,
        memory_entries: list,
        query: str,
        output_mode: str,
    ) -> str:
        """
        Ultra-compact format for Codex models.

        Codex has a small context window (8k legacy / 200k codex-mini).
        For legacy Codex we minimise prose and only surface file path +
        critical code.  For codex-mini we use the standard OpenAI format.
        """
        parts = [base_prompt]

        if chunks:
            parts.append("# Context")
            for c in chunks:
                lang = _infer_language(c.path)
                # Only first 40 lines to save tokens for legacy Codex
                snippet = "\n".join(c.content.splitlines()[:40])
                symbol_tag = f" [{c.symbol}]" if c.symbol and c.symbol != "module" else ""
                parts.append(
                    f"// {c.path}{symbol_tag}\n```{lang}\n{snippet}\n```"
                )

        if memory_entries:
            parts.append("# Memory")
            for m in memory_entries:
                parts.append(f"// [{m['type']}] {m['key']}: {m['value']}")

        parts.append(f"# Task\n{query}")
        return "\n\n".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".sh": "bash",
    ".zsh": "bash",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
}


def _infer_language(path: str) -> str:
    """Guess a markdown fence language tag from file extension."""
    if not path:
        return ""
    dot = path.rfind(".")
    if dot == -1:
        return ""
    ext = path[dot:].lower()
    return _EXT_LANG.get(ext, "")
