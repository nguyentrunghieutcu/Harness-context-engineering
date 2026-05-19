"""
server.py — Harness context engineering v5.0
=========================================
Context Orchestration + Mini-RAG Engine

MCP Tools exposed:
  retrieve_context       — hybrid retrieval (semantic + BM25 + graph)
  compact_conversation   — auto-format compact with summary injection
  memory_save            — save to episodic / semantic / procedural memory
  memory_search          — search across memory tiers
  memory_inject          — inject relevant memory into prompt
  memory_delete          — delete by key
  memory_list            — list recent entries
  memory_evict           — LRU eviction
  memory_stats           — DB statistics
  estimate_tokens        — token count utility
  get_token_budget       — model-specific budget info
"""

from __future__ import annotations
from compact.summarizer import compact_messages
from context.sanitizer import sanitize, sanitize_chunks
from context.assembler import ContextAssembler
from context.budgeting import get_budget
from memory.vector_store import VectorStore
from memory.episodic import EpisodicMemory, SemanticMemory, ProceduralMemory, MemoryStore
from chunking.compressor import ContextCompressor
from chunking.treesitter_chunker import ASTChunker, Chunk, count_tokens
from retrieval.cache import RetrievalCache
from retrieval.graph import DependencyGraph
from retrieval.reranker import Reranker
from retrieval.bm25 import BM25Index
from retrieval.embeddings import EmbeddingEngine
from mcp.server.fastmcp import FastMCP

import os
import sys
import json
import logging
import time

sys.path.insert(0, os.path.dirname(__file__))


# ── Sub-modules ─────────────────────────────────────────────────────────


# ── Logging ─────────────────────────────────────────────────────────────
# Primary log: provider-neutral location under ~/.mcp-harness/
LOG_DIR = os.path.expanduser("~/.mcp-harness")
LOG_FILE = os.path.join(LOG_DIR, "harness-v3.log")
os.makedirs(LOG_DIR, exist_ok=True)

# Legacy symlink: keep ~/.gemini/mcp-harness-v3.log pointing to the real file
# so existing `tail -f` commands still work.
_LEGACY_DIR = os.path.expanduser("~/.gemini")
_LEGACY_LINK = os.path.join(_LEGACY_DIR, "mcp-harness-v3.log")
try:
    os.makedirs(_LEGACY_DIR, exist_ok=True)
    if not os.path.exists(_LEGACY_LINK):
        os.symlink(LOG_FILE, _LEGACY_LINK)
except OSError:
    pass  # non-critical — skip if symlink can't be created

_log_handlers: list[logging.Handler] = [
    logging.FileHandler(LOG_FILE, encoding="utf-8"),
]

# If a stale legacy file already exists and is not the primary log symlink,
# write to it as well so older `tail -f ~/.gemini/...` commands keep working.
try:
    if os.path.exists(_LEGACY_LINK) and not os.path.samefile(LOG_FILE, _LEGACY_LINK):
        _log_handlers.append(logging.FileHandler(_LEGACY_LINK, encoding="utf-8"))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger("harness.server")


def _summary_token_count(messages: list[dict]) -> int:
    """Count tokens in the generated conversation summary, if present."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and "<conversation_summary>" in content:
            return count_tokens(content)

        parts = message.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and "<conversation_summary>" in text:
                    return count_tokens(text)

    return 0


def _coerce_target_ratio(target_ratio: float) -> float:
    """Return a sane prompt target ratio in (0, 1]."""
    try:
        ratio = float(target_ratio)
    except (TypeError, ValueError):
        return 0.20
    if ratio <= 0:
        return 0.20
    return min(ratio, 1.0)


def _prompt_token_limit(
    *,
    context_window: int,
    inject_budget: int,
    target_ratio: float,
    max_prompt_tokens: int,
) -> int:
    """Resolve the global prompt cap, never exceeding the inject budget."""
    if max_prompt_tokens and max_prompt_tokens > 0:
        return max(1, min(int(max_prompt_tokens), inject_budget))

    ratio_limit = int(context_window * _coerce_target_ratio(target_ratio))
    return max(1, min(ratio_limit, inject_budget))


def _optimizer_report_comment(
    *,
    before_tokens: int,
    after_tokens: int,
    target_tokens: int,
    chunks_before: int,
    chunks_after: int,
    memory_before: int,
    memory_after: int,
    status: str,
) -> str:
    saved_tokens = max(before_tokens - after_tokens, 0)
    saved_percent = (
        round(saved_tokens * 100 / before_tokens, 1)
        if before_tokens > 0 else 0.0
    )
    return (
        "<!-- prompt_optimizer "
        f"status={status} before={before_tokens} after={after_tokens} "
        f"saved={saved_tokens} saved_percent={saved_percent}% "
        f"target={target_tokens} chunks={chunks_after}/{chunks_before} "
        f"memory={memory_after}/{memory_before} -->"
    )


def _with_optimizer_report(
    prompt: str,
    *,
    before_tokens: int,
    target_tokens: int,
    chunks_before: int,
    chunks_after: int,
    memory_before: int,
    memory_after: int,
    status: str,
) -> str:
    """Prepend a compact report and make its `after` token count exact."""
    provisional = _optimizer_report_comment(
        before_tokens=before_tokens,
        after_tokens=count_tokens(prompt),
        target_tokens=target_tokens,
        chunks_before=chunks_before,
        chunks_after=chunks_after,
        memory_before=memory_before,
        memory_after=memory_after,
        status=status,
    )
    after_tokens = count_tokens(f"{provisional}\n{prompt}")
    final = _optimizer_report_comment(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        target_tokens=target_tokens,
        chunks_before=chunks_before,
        chunks_after=chunks_after,
        memory_before=memory_before,
        memory_after=memory_after,
        status=status,
    )
    return f"{final}\n{prompt}"


def _fit_prompt_to_limit(
    *,
    base_prompt: str,
    chunks: list[Chunk],
    memory_entries: list[dict],
    query: str,
    output_mode: str,
    model: str,
    target_tokens: int,
    before_tokens: int,
    include_report: bool,
    base_prompt_policy: str,
) -> tuple[str, list[Chunk], list[dict], str]:
    """
    Enforce the global prompt cap against the fully assembled output.

    If retrieved context and memory cannot be trimmed enough because the base
    prompt itself exceeds the cap, return a clear rejection unless callers opt
    into unsafe base prompt truncation with base_prompt_policy="truncate".
    """
    selected_chunks = list(chunks)
    selected_memory = list(memory_entries)
    chunks_before = len(selected_chunks)
    memory_before = len(selected_memory)

    def render(status: str) -> tuple[str, int]:
        prompt = assembler.assemble(
            base_prompt=base_prompt,
            chunks=selected_chunks,
            memory_entries=selected_memory,
            query=query,
            output_mode=output_mode,
            model=model,
        )
        if include_report:
            prompt = _with_optimizer_report(
                prompt,
                before_tokens=before_tokens,
                target_tokens=target_tokens,
                chunks_before=chunks_before,
                chunks_after=len(selected_chunks),
                memory_before=memory_before,
                memory_after=len(selected_memory),
                status=status,
            )
        return prompt, count_tokens(prompt)

    did_trim = False
    prompt, tokens = render("ok")
    while tokens > target_tokens and (selected_chunks or selected_memory):
        did_trim = True
        if selected_chunks:
            selected_chunks.pop()
        elif selected_memory:
            selected_memory.pop()
        prompt, tokens = render("trimmed")

    if tokens <= target_tokens:
        return prompt, selected_chunks, selected_memory, "trimmed" if did_trim else "ok"

    if base_prompt_policy.lower() == "truncate":
        return _fit_with_truncated_base_prompt(
            base_prompt=base_prompt,
            chunks=[],
            memory_entries=[],
            query=query,
            output_mode=output_mode,
            model=model,
            target_tokens=target_tokens,
            before_tokens=before_tokens,
            include_report=include_report,
            chunks_before=chunks_before,
            memory_before=memory_before,
        )

    base_tokens = count_tokens(base_prompt)
    rejection = (
        "Error: prompt_optimizer rejected output because the base prompt alone "
        "does not fit the global prompt target.\n"
        f"base_prompt_tokens={base_tokens:,}; target_tokens={target_tokens:,}; "
        "set a higher max_prompt_tokens/target_ratio, reduce base_prompt, or "
        "use base_prompt_policy='truncate' if lossy truncation is acceptable."
    )
    if include_report:
        rejection = _with_optimizer_report(
            rejection,
            before_tokens=before_tokens,
            target_tokens=target_tokens,
            chunks_before=chunks_before,
            chunks_after=0,
            memory_before=memory_before,
            memory_after=0,
            status="rejected_base_prompt",
        )
    return rejection, [], [], "rejected_base_prompt"


def _fit_with_truncated_base_prompt(
    *,
    base_prompt: str,
    chunks: list[Chunk],
    memory_entries: list[dict],
    query: str,
    output_mode: str,
    model: str,
    target_tokens: int,
    before_tokens: int,
    include_report: bool,
    chunks_before: int,
    memory_before: int,
) -> tuple[str, list[Chunk], list[dict], str]:
    """Last-resort lossy base prompt truncation."""
    marker = "\n\n[base_prompt truncated by prompt_optimizer]\n"
    available = max(1, target_tokens - count_tokens(query) - 120)
    while available > 0:
        truncated_base = _truncate_to_tokens(base_prompt, available) + marker
        prompt = assembler.assemble(
            base_prompt=truncated_base,
            chunks=chunks,
            memory_entries=memory_entries,
            query=query,
            output_mode=output_mode,
            model=model,
        )
        if include_report:
            prompt = _with_optimizer_report(
                prompt,
                before_tokens=before_tokens,
                target_tokens=target_tokens,
                chunks_before=chunks_before,
                chunks_after=0,
                memory_before=memory_before,
                memory_after=0,
                status="truncated_base_prompt",
            )
        if count_tokens(prompt) <= target_tokens:
            return prompt, [], [], "truncated_base_prompt"
        available = int(available * 0.8)

    rejection = (
        "Error: prompt_optimizer could not fit even the truncated base prompt "
        f"under target_tokens={target_tokens:,}."
    )
    return rejection, [], [], "rejected_base_prompt"


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Token-aware truncation with a character fallback."""
    if count_tokens(text) <= max_tokens:
        return text
    try:
        import tiktoken as _tiktoken
        enc = _tiktoken.get_encoding("cl100k_base")
        return enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])
    except Exception:
        return text[:max_tokens * 3]


def _messages_token_count(messages: list[dict]) -> int:
    return count_tokens(json.dumps(messages, ensure_ascii=False))

# ── Singletons ──────────────────────────────────────────────────────────
mcp = FastMCP("Harness context engineering")
embedder = EmbeddingEngine()
bm25 = BM25Index()
reranker = Reranker()
graph = DependencyGraph()
cache = RetrievalCache(ttl=600)
chunker = ASTChunker()
compressor = ContextCompressor()
vec_store = VectorStore()
assembler = ContextAssembler()
mem_store = MemoryStore.instance()
episodic = EpisodicMemory(mem_store)
semantic = SemanticMemory(mem_store)
procedural = ProceduralMemory(mem_store)

# ── In-memory chunk index (rebuilt on each retrieve call for new paths) ─
_indexed_paths: set[str] = set()
_all_chunks: list[Chunk] = []


def _ensure_indexed(
        paths: list[str],
        force_reindex: bool = False) -> list[Chunk]:
    """
    Build/update the chunk index for the given paths.
    Skips paths already indexed unless force_reindex=True.
    """
    new_paths = [p for p in paths if p not in _indexed_paths or force_reindex]
    if not new_paths and _all_chunks:
        return _all_chunks

    t0 = time.time()
    chunks = chunker.chunk_paths(new_paths)

    if not chunks:
        return _all_chunks

    # Safety: sanitize before indexing
    sanitize_chunks(chunks)

    # Embed all chunks in batch
    texts = [c.content[:512] for c in chunks]
    vecs = embedder.embed_batch(texts)
    for c, v in zip(chunks, vecs):
        c.embedding = v

    # Update vector store
    vec_store.upsert(chunks)

    # Update BM25 index (full rebuild over all chunks)
    _all_chunks.extend(chunks)
    bm25.build(_all_chunks)

    # Update dependency graph
    graph.build(_all_chunks)

    _indexed_paths.update(new_paths)
    logger.info(
        f"[Server] indexed {len(chunks)} new chunks in "
        f"{time.time() - t0:.1f}s | total={len(_all_chunks)}")
    return _all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def retrieve_context(
    base_prompt: str,
    paths: list[str],
    query: str,
    model: str = "claude-sonnet-4",
    output_mode: str = "concise",
    retrieve_top_n: int = 50,
    rerank_top_k: int = 12,
    memory_top_k: int = 4,
    min_memory_sim: float = 0.18,
    graph_expand: bool = True,
    compress: bool = True,
    force_reindex: bool = False,
    target_ratio: float = 0.20,
    max_prompt_tokens: int = 0,
    include_report: bool = True,
    base_prompt_policy: str = "reject",
) -> str:
    """
    MAIN TOOL — Hybrid RAG retrieval pipeline.

    Pipeline: index → hybrid score → rerank → graph expand → compress → assemble
    → enforce global prompt budget.

    :param base_prompt:    System prompt to build on.
    :param paths:          File/folder paths to index and retrieve from.
    :param query:          Current user task or question.
    :param model:          Target model (affects token budget + output format).
    :param output_mode:    "concise" | "structured" | "code_only" | "minimal"
    :param retrieve_top_n: Candidate pool before reranking (default 50).
    :param rerank_top_k:   Final chunks after reranking (default 12).
    :param memory_top_k:   Memory entries to inject (default 4).
    :param min_memory_sim: Minimum similarity for memory injection.
    :param graph_expand:   Expand retrieval via dependency graph.
    :param compress:       Compress long chunks before injecting.
    :param force_reindex:  Force re-chunking even for already-indexed paths.
    :param target_ratio:   Max share of model context window for final prompt.
                            Ignored when max_prompt_tokens > 0.
    :param max_prompt_tokens: Absolute final prompt cap. 0 uses target_ratio.
    :param include_report: Prepend compact token savings report to output.
    :param base_prompt_policy: "reject" or "truncate" if base_prompt alone
                               exceeds the global cap.
    """
    try:
        budget = get_budget(model)
        target_tokens = _prompt_token_limit(
            context_window=budget.context_window,
            inject_budget=budget.inject_budget,
            target_ratio=target_ratio,
            max_prompt_tokens=max_prompt_tokens,
        )
        logger.info(
            f"model={model!r} budget={budget.context_window:,} "
            f"inject={budget.inject_budget:,} target={target_tokens:,}")
        logger.info(f"retrieve_context | query={query[:100]!r} | paths={paths}")

        # ── 1. Cache check ───────────────────────────────────────────────────
        cached = cache.get(query, paths)
        if cached:
            logger.info(f"  [Cache HIT] {len(cached)} chunks")
            final_chunks = cached
        else:
            # ── 2. Index / retrieve ──────────────────────────────────────────
            all_chunks = _ensure_indexed(paths, force_reindex)
            if not all_chunks:
                logger.info("  No chunks indexed from provided paths")
                final_chunks = []
            else:
                # ── 3. Hybrid scoring ────────────────────────────────────────
                query_vec = embedder.embed_text(query)
                vec_results = vec_store.search(query_vec, top_k=retrieve_top_n)
                bm25_scores = bm25.score(query)

                # Build chunk lookup by id
                chunk_by_id = {c.id: c for c in all_chunks}
                bm25_by_idx = {c.id: s for c, s in zip(bm25.chunks, bm25_scores)}

                candidates: list[Chunk] = []
                for r in vec_results:
                    c = chunk_by_id.get(r["id"])
                    if c is None:
                        continue

                    sem_score = float(r.get("score", 0))
                    bm25_score = bm25_by_idx.get(c.id, 0.0)
                    rec_score = 0.0    # recency not tracked per chunk
                    prio_score = c.priority / 10.0
                    grph_score = graph.symbol_score(query, c)

                    final = (
                        sem_score * 0.55 +
                        bm25_score * 0.20 +
                        rec_score * 0.10 +
                        prio_score * 0.10 +
                        grph_score * 0.05
                    )
                    c._score = final   # type: ignore[attr-defined]
                    candidates.append(c)

                candidates.sort(
                    key=lambda x: getattr(
                        x, "_score", 0), reverse=True)

                # ── 4. Rerank top-N ──────────────────────────────────────────
                reranked = reranker.rerank(
                    query, candidates[:retrieve_top_n], top_k=rerank_top_k)

                # ── 5. Graph expansion ───────────────────────────────────────
                if graph_expand:
                    extras = graph.expand(reranked, max_extra=4)
                    for e in extras:
                        if e.id not in {c.id for c in reranked}:
                            e._score = 0.3   # type: ignore[attr-defined]
                            reranked.append(e)

                final_chunks = reranked
                cache.set(query, paths, final_chunks)

        # ── 6. Compress candidates (before budget enforcement) ───────────────
        # Compress first so budget is enforced on ACTUAL output token counts,
        # not inflated pre-compression counts.
        if compress:
            compressed_strings = compressor.compress(final_chunks, model=model)
            for c, cs in zip(final_chunks, compressed_strings):
                c.content = cs
                c.tokens = count_tokens(cs)  # re-count post-compression

        # ── 7. Memory retrieval ──────────────────────────────────────────────
        memory_entries = mem_store.search(
            query, top_k=memory_top_k, min_sim=min_memory_sim)
        logger.info(f"  {len(memory_entries)} memory entries injected")

        # ── 8. Assemble once without trimming for before/after telemetry ─────
        before_prompt = assembler.assemble(
            base_prompt=base_prompt,
            chunks=final_chunks,
            memory_entries=memory_entries,
            query=query,
            output_mode=output_mode,
            model=model,
        )
        before_tokens = count_tokens(before_prompt)

        # ── 9. Enforce global prompt budget on the actual assembled output ───
        result, selected_chunks, selected_memory, status = _fit_prompt_to_limit(
            base_prompt=base_prompt,
            chunks=final_chunks,
            memory_entries=memory_entries,
            query=query,
            output_mode=output_mode,
            model=model,
            target_tokens=target_tokens,
            before_tokens=before_tokens,
            include_report=include_report,
            base_prompt_policy=base_prompt_policy,
        )
        after_tokens = count_tokens(result)
        saved_tokens = max(before_tokens - after_tokens, 0)
        saved_percent = (
            round(saved_tokens * 100 / before_tokens, 1)
            if before_tokens > 0 else 0.0
        )
        logger.info(
            "  prompt_optimizer | "
            f"status={status} before={before_tokens:,} after={after_tokens:,} "
            f"saved={saved_tokens:,} saved_percent={saved_percent}% "
            f"target={target_tokens:,} chunks={len(selected_chunks)}/{len(final_chunks)} "
            f"memory={len(selected_memory)}/{len(memory_entries)}"
        )
        return result

    except Exception as e:
        logger.error(f"  ❌ {e}", exc_info=True)
        return f"Error in retrieve_context: {e}"


@mcp.tool()
def compact_conversation(
    messages_json: str,
    retain_turns: int = 2,
    model: str = "",
    max_recent_tool_tokens: int = 800,
) -> str:
    """
    Compact conversation history with structured summary injection.
    Auto-detects OpenAI / Anthropic / Gemini format.
    Replaces old turns with <conversation_summary> XML block.

    :param messages_json: JSON array of messages.
    :param retain_turns:  Number of recent turns to keep intact (default 2).
                          0 is allowed and compacts the full non-prefix body.
    :param model:         Model hint for provider-specific formatting
                          (e.g. "gpt-4o", "codex-mini-latest", "o3",
                           "claude-sonnet-4", "gemini-2.5-pro").
    :param max_recent_tool_tokens: Cap retained OpenAI tool/function payloads.
    """
    try:
        try:
            before_messages = json.loads(messages_json)
            before_tokens = (
                _messages_token_count(before_messages)
                if isinstance(before_messages, list) else 0
            )
        except (json.JSONDecodeError, ValueError):
            before_tokens = 0

        result = compact_messages(
            messages_json,
            retain_turns,
            model=model,
            max_recent_tool_tokens=max_recent_tool_tokens,
        )
        after_tokens = _messages_token_count(result)
        saved_tokens = max(before_tokens - after_tokens, 0)
        saved_percent = (
            round(saved_tokens * 100 / before_tokens, 1)
            if before_tokens > 0 else 0.0
        )
        logger.info(
            f"compact_messages | retain_turns={retain_turns} | "
            f"summary_tokens={_summary_token_count(result):,} | "
            f"before={before_tokens:,} after={after_tokens:,} "
            f"saved={saved_tokens:,} saved_percent={saved_percent}%")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"  ❌ {e}")
        return f"Error: {e}"


# ── Memory tools ────────────────────────────────────────────────────────

@mcp.tool()
def memory_save(
    key: str,
    value: str,
    mtype: str = "semantic",
    tags: str = "",
) -> str:
    """
    Save knowledge to long-term memory.

    :param key:   Short identifying key (used for search).
    :param value: Content to remember.
    :param mtype: Memory tier — "episodic" | "semantic" | "procedural"
    :param tags:  Comma-separated labels for filtering.
    """
    try:
        value = sanitize(value, escape_xml=False)
        store = {
            "episodic": episodic,
            "semantic": semantic,
            "procedural": procedural}.get(
            mtype,
            semantic)
        result = store.save(key, value, tags)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_search(
    query: str,
    mtype: str = "",
    top_k: int = 5,
    min_sim: float = 0.12,
) -> str:
    """
    Search long-term memory by semantic similarity.

    :param query:   Search query.
    :param mtype:   Filter by tier: "episodic" | "semantic" | "procedural" | "" (all)
    :param top_k:   Max results.
    :param min_sim: Minimum similarity threshold (0–1).
    """
    try:
        from memory.episodic import MemoryType
        mt = None
        if mtype:
            try:
                mt = MemoryType(mtype)
            except ValueError:
                pass
        results = mem_store.search(
            query, mtype=mt, top_k=top_k, min_sim=min_sim)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_inject(
    base_prompt: str,
    query: str,
    top_k: int = 4,
    min_sim: float = 0.18,
    output_mode: str = "concise",
    model: str = "claude-sonnet-4",
    target_ratio: float = 0.20,
    max_prompt_tokens: int = 0,
    include_report: bool = True,
    base_prompt_policy: str = "reject",
) -> str:
    """
    Inject relevant memory entries into a system prompt (without file retrieval).
    Use when you only need memory context, not workspace files.

    :param base_prompt:  System prompt.
    :param query:        Current task.
    :param top_k:        Max memory entries.
    :param min_sim:      Minimum similarity.
    :param output_mode:  "concise" | "structured" | "code_only" | "minimal"
    :param model:        Target model for format selection.
    :param target_ratio: Max share of model context window for final prompt.
    :param max_prompt_tokens: Absolute final prompt cap. 0 uses target_ratio.
    :param include_report: Prepend compact token savings report to output.
    :param base_prompt_policy: "reject" or "truncate" for oversized base_prompt.
    """
    try:
        entries = mem_store.search(query, top_k=top_k, min_sim=min_sim)
        budget = get_budget(model)
        target_tokens = _prompt_token_limit(
            context_window=budget.context_window,
            inject_budget=budget.inject_budget,
            target_ratio=target_ratio,
            max_prompt_tokens=max_prompt_tokens,
        )
        before_prompt = assembler.assemble(
            base_prompt=base_prompt, chunks=[], memory_entries=entries,
            query=query, output_mode=output_mode, model=model,
        )
        result, _, _, status = _fit_prompt_to_limit(
            base_prompt=base_prompt,
            chunks=[],
            memory_entries=entries,
            query=query,
            output_mode=output_mode,
            model=model,
            target_tokens=target_tokens,
            before_tokens=count_tokens(before_prompt),
            include_report=include_report,
            base_prompt_policy=base_prompt_policy,
        )
        logger.info(
            "  memory_inject optimizer | "
            f"status={status} after={count_tokens(result):,} "
            f"target={target_tokens:,}"
        )
        return result
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_delete(key: str, mtype: str = "semantic") -> str:
    """Delete a memory entry by exact key and type."""
    try:
        from memory.episodic import MemoryType
        mt = MemoryType(mtype)
        deleted = mem_store.delete(mt, key)
        return json.dumps({"deleted": deleted, "key": key, "mtype": mtype})
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_list(mtype: str = "", limit: int = 30) -> str:
    """List recent memory entries, optionally filtered by type."""
    try:
        from memory.episodic import MemoryType
        mt = MemoryType(mtype) if mtype else None
        return json.dumps(mem_store.list_keys(mt, limit), ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_evict(keep_top: int = 300) -> str:
    """Evict least-recently-used memory entries to free space."""
    try:
        before = mem_store.stats()
        removed = mem_store.evict_lru(keep_top)
        after = mem_store.stats()
        return json.dumps(
            {"removed": removed, "before": before, "after": after})
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_stats() -> str:
    """Return statistics per memory tier (count, tokens, hits)."""
    try:
        return json.dumps(mem_store.stats(), ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def estimate_tokens(text: str) -> str:
    """
    Estimate token count for any text.
    Useful for checking prompt size before sending to model.
    """
    tok = count_tokens(text)
    chars = len(text)
    # Representative models across all supported providers
    _benchmark_models = [
        # Claude
        "claude-sonnet-4",
        "claude-opus-4",
        # Gemini
        "gemini-2.5-pro",
        # OpenAI GPT-5
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5",
        # OpenAI Chat
        "gpt-4o",
        "gpt-4.1",
        "gpt-4o-mini",
        # OpenAI Reasoning
        "o3",
        "o1",
        "o4-mini",
        # OpenAI Codex
        "codex-mini-latest",
    ]
    budgets = {
        m: get_budget(m).inject_budget
        for m in _benchmark_models
    }
    return json.dumps({
        "tokens": tok,
        "chars": chars,
        "ratio_chars_per_tok": round(chars / tok, 2) if tok > 0 else 0,
        "fits_in": {
            m: tok <= b
            for m, b in budgets.items()
        },
    }, ensure_ascii=False)


@mcp.tool()
def get_token_budget(model: str = "claude-sonnet-4") -> str:
    """
    Return the dynamic token budget for a specific model.
    Shows context window, reserved headroom, and available inject budget.
    """
    try:
        b = get_budget(model)
        return json.dumps({
            "model": b.model_name,
            "context_window": b.context_window,
            "reserved_output": b.reserved_output,
            "reserved_reasoning": b.reserved_reasoning,
            "reserved_tools": b.reserved_tools,
            "inject_budget": b.inject_budget,
        }, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def invalidate_cache() -> str:
    """Clear the retrieval cache. Call when workspace files have changed."""
    cache.invalidate_all()
    return json.dumps({"status": "cache cleared"})


@mcp.tool()
def reindex_paths(paths: list[str]) -> str:
    """
    Force re-chunking and re-indexing of given paths.
    Use after significant file edits to refresh the vector index.
    """
    logger.info(f"[Server] force reindex: {paths}")
    try:
        chunks = _ensure_indexed(paths, force_reindex=True)
        return json.dumps({"status": "reindexed", "total_chunks": len(chunks)})
    except Exception as e:
        return f"Error: {e}"


# ── Entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("═" * 70)
    logger.info("Harness context engineering v5.0 starting")
    logger.info(f"  Log (primary) : {LOG_FILE}")
    logger.info(f"  Log (legacy)  : {_LEGACY_LINK}  → symlink")
    logger.info(f"  tail -f {LOG_FILE}")
    logger.info("  Tools  : retrieve_context · compact_conversation")
    logger.info("           memory_save · memory_search · memory_inject")
    logger.info(
        "           memory_delete · memory_list · memory_evict · memory_stats")
    logger.info("           estimate_tokens · get_token_budget")
    logger.info("           invalidate_cache · reindex_paths")
    logger.info(
        "  Models : Claude · Gemini · GPT-4o/4.1/4.5 · o1/o3/o4 · Codex")
    logger.info(
        "  Stack  : numpy vector store + BM25 + AST chunker")
    logger.info(
        "           local reranker + dependency graph + safety layer")
    logger.info("═" * 70)
    mcp.run()
