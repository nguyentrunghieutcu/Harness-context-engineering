"""
chunking/compressor.py — Context compressor
============================================
Compresses chunks into a compact representation depending on the target
model provider:
  - Claude  → XML <context> blocks
  - OpenAI  → Markdown comments (minimal token overhead)
  - Codex   → Single-line summary + truncated code (max 30 lines)
  - Gemini  → JSON object
"""

from __future__ import annotations

import json


class ContextCompressor:
    def __init__(self) -> None:
        pass

    def compress(
        self,
        chunks: list,
        model: str = "",
    ) -> list[str]:
        """
        Compress each chunk to its minimal useful representation.

        :param chunks: List of Chunk objects.
        :param model:  Target model string for format selection.
        :returns: List of compressed strings (one per chunk).
        """
        ml = model.lower()

        if "gemini" in ml:
            return [self._compress_gemini(c) for c in chunks]
        if any(kw in ml for kw in ("codex", "code-davinci", "code-cushman")):
            return [self._compress_codex(c) for c in chunks]
        if "claude" in ml or "anthropic" in ml:
            return [self._compress_claude(c) for c in chunks]
        # Default: OpenAI markdown (also used for gpt-*, o1/o3/o4)
        return [self._compress_openai(c) for c in chunks]

    # ── Claude XML ────────────────────────────────────────────────────────────
    def _compress_claude(self, c) -> str:
        summary = c.summary if c.summary else f"{c.type} {c.symbol}"
        critical = "\n".join(c.content.splitlines()[:60])
        return (
            f"<context>\n"
            f"  <summary>{summary}</summary>\n"
            f"  <critical_code>\n{critical}\n  </critical_code>\n"
            f"</context>"
        )

    # ── OpenAI Markdown ───────────────────────────────────────────────────────
    def _compress_openai(self, c) -> str:
        summary = c.summary if c.summary else f"{c.type} `{c.symbol}`"
        critical = "\n".join(c.content.splitlines()[:60])
        lang = _ext_to_lang(c.path)
        return (
            f"<!-- {c.path} | {summary} -->\n"
            f"```{lang}\n{critical}\n```"
        )

    # ── Codex — ultra-compact ─────────────────────────────────────────────────
    def _compress_codex(self, c) -> str:
        summary = c.summary if c.summary else f"{c.type} {c.symbol}"
        # Only 30 lines for legacy Codex (tiny context)
        critical = "\n".join(c.content.splitlines()[:30])
        lang = _ext_to_lang(c.path)
        return (
            f"// {c.path} [{summary}]\n"
            f"```{lang}\n{critical}\n```"
        )

    # ── Gemini JSON ───────────────────────────────────────────────────────────
    def _compress_gemini(self, c) -> str:
        return json.dumps({
            "path": c.path,
            "symbol": c.symbol,
            "summary": c.summary or f"{c.type} {c.symbol}",
            "snippet": "\n".join(c.content.splitlines()[:60]),
        }, ensure_ascii=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

_EXT_LANG: dict[str, str] = {
    ".py": "python", ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "jsx", ".go": "go",
    ".rs": "rust", ".java": "java", ".cs": "csharp",
    ".cpp": "cpp", ".c": "c", ".h": "c",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".sh": "bash", ".zsh": "bash", ".md": "markdown",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".html": "html", ".css": "css", ".sql": "sql",
}


def _ext_to_lang(path: str) -> str:
    if not path:
        return ""
    dot = path.rfind(".")
    return _EXT_LANG.get(path[dot:].lower(), "") if dot != -1 else ""
