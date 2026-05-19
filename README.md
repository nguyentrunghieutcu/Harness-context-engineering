# Harness context engineering v5.0

> **MCP Tool Server** chạy local — tối ưu hóa context window cho AI assistant (Codex / Claude) với Hybrid RAG, Memory Tiers, dynamic token budgeting, và global prompt optimization.

---

## 🧩 Tech Stack

| Thành phần | Chi tiết |
|---|---|
| **Ngôn ngữ** | Python 3.10+ |
| **Framework** | [`FastMCP`](https://github.com/jlowin/fastmcp) — MCP Server qua `stdio` transport |
| **Embeddings** | **Local TF-IDF + LSA** (Scikit-learn) — Không download, 128-dim |
| **Vector Store** | **In-memory Numpy** (Cosine Similarity) — Nhanh, không phụ thuộc DB external |
| **BM25** | Full-text sparse retrieval (rank-bm25) |
| **Reranker** | **Local Hybrid Reranker** (BM25 + Keyword Overlap) |
| **Chunker** | AST-aware chunker (`treesitter_chunker`) |
| **Memory DB** | SQLite via `memory/episodic.py` — 3 tiers |
| **Logging** | `~/.mcp-harness/harness-v3.log` |
| **Config** | `.mcp.json` + `~/.gemini/antigravity/mcp_config.json` |
| **Entry point** | `server.py` |

---

## 📁 Cấu trúc thư mục

```
compact-token/
├── server.py                  # Entry point — FastMCP server, 13 tools
├── .mcp.json                  # MCP config cho workspace
│
├── chunking/
│   ├── treesitter_chunker.py  # AST-aware code chunker + token counter
│   └── compressor.py          # XML/JSON context compressor
│
├── retrieval/
│   ├── embeddings.py          # Sentence-transformer embedding engine
│   ├── bm25.py                # BM25 sparse index
│   ├── reranker.py            # Cross-encoder reranker
│   ├── graph.py               # Dependency graph (symbol expansion)
│   └── cache.py               # TTL retrieval cache
│
├── memory/
│   ├── episodic.py            # MemoryStore: Episodic / Semantic / Procedural tiers
│   └── vector_store.py        # LanceDB vector store wrapper
│
├── context/
│   ├── assembler.py           # Prompt assembler (multi-mode output)
│   ├── budgeting.py           # Dynamic token budget per model
│   └── sanitizer.py           # Input/chunk safety sanitizer
│
└── compact/
    ├── summarizer.py          # Conversation compactor (multi-format)
    ├── anthropic.py           # Anthropic message format handler
    ├── openai.py              # OpenAI message format handler
    └── gemini.py              # Gemini message format handler
```

---

## 🎯 13 MCP Tools

Các tool này được gọi từ MCP client (Codex / Claude Desktop), không phải CLI shell trực tiếp. Payload bên dưới là JSON arguments truyền vào tool.

### 🔍 Retrieval

#### `retrieve_context`
> **MAIN TOOL** — Hybrid RAG pipeline đầy đủ.

Pipeline: `index → hybrid score (BM25 + semantic + graph) → rerank → graph expand → compress → assemble`

| Param | Default | Mô tả |
|---|---|---|
| `base_prompt` | — | System prompt gốc |
| `paths` | — | Danh sách file/folder cần index |
| `query` | — | Task hiện tại |
| `model` | `claude-sonnet-4` | Model đích (ảnh hưởng token budget) |
| `output_mode` | `concise` | `concise` / `structured` / `code_only` / `minimal` |
| `retrieve_top_n` | `50` | Pool candidates trước rerank |
| `rerank_top_k` | `12` | Chunks sau rerank |
| `memory_top_k` | `4` | Memory entries inject thêm |
| `graph_expand` | `true` | Mở rộng qua dependency graph |
| `compress` | `true` | Nén chunks thành XML |
| `force_reindex` | `false` | Force re-chunk ngay cả khi đã index |
| `target_ratio` | `0.20` | Giới hạn prompt cuối theo % context window |
| `max_prompt_tokens` | `0` | Giới hạn tuyệt đối; `0` dùng `target_ratio` |
| `include_report` | `true` | Thêm report token ngắn vào output |
| `base_prompt_policy` | `reject` | `reject` hoặc `truncate` nếu base prompt quá lớn |

Budget được enforce trên prompt cuối đã assemble:
`base_prompt + retrieved_context + memory + query + wrapper`. Nếu base prompt
tự nó vượt target, tool sẽ reject với báo cáo rõ, trừ khi bật
`base_prompt_policy: "truncate"`.

**Use case:**
- Trước khi sửa một flow lớn, lấy đúng file/chunk liên quan thay vì nhét toàn bộ repo vào context.
- Hỏi kiến trúc hoặc dependency của một module.
- Chuẩn bị context ngắn gọn cho model có context window nhỏ hơn.

**Ví dụ:**
```json
{
  "base_prompt": "You are a senior backend engineer. Use retrieved context only when relevant.",
  "paths": ["src/", "server.py"],
  "query": "optimize auth flow",
  "model": "gpt-5.5",
  "output_mode": "structured",
  "retrieve_top_n": 50,
  "rerank_top_k": 12,
  "memory_top_k": 4,
  "graph_expand": true,
  "compress": true,
  "target_ratio": 0.2
}
```

#### `reindex_paths`
Force re-chunk và re-index các path chỉ định. Dùng sau khi sửa file lớn.

**Use case:** vừa refactor hoặc tạo file mới, cần index lại để lần `retrieve_context` sau thấy nội dung mới.

**Ví dụ:**
```json
{
  "paths": ["src/auth/", "server.py"]
}
```

#### `invalidate_cache`
Xóa TTL retrieval cache. Dùng khi workspace thay đổi nhiều.

**Use case:** kết quả retrieval cũ không còn đúng vì vừa đổi nhiều file hoặc đổi nhánh.

**Ví dụ:**
```json
{}
```

---

### 💬 Conversation

#### `compact_conversation`
> Nén lịch sử hội thoại — auto-detect OpenAI / Anthropic / Gemini format.

- Thay thế các turns cũ bằng `<conversation_summary>` XML block
- Giữ nguyên **N turns gần nhất** (default: 2)
- Tiết kiệm **40–70% tokens** trong conversation dài

| Param | Default | Mô tả |
|---|---|---|
| `messages_json` | — | JSON array của messages |
| `retain_turns` | `2` | Số turns gần nhất giữ nguyên |
| `model` | — | Model hint cho summary cap |
| `max_recent_tool_tokens` | `800` | Cap tool/function output vẫn nằm trong turns được giữ |

`retain_turns=0` hợp lệ và sẽ compact toàn bộ body sau phần
system/developer prefix. Tool cũng log `before`, `after`, `saved_tokens`,
và `saved_percent`.

**Use case:**
- Hội thoại dài nhưng vẫn muốn giữ quyết định, bug đã gặp, tool đã chạy.
- Trước khi chuyển model hoặc tiếp tục task dài với context gọn hơn.
- Dùng cho OpenAI / Anthropic / Gemini message arrays mà không cần tự viết formatter riêng.

**Ví dụ:**
```json
{
  "messages_json": "[{\"role\":\"user\",\"content\":\"Fix auth bug\"},{\"role\":\"assistant\",\"content\":\"Inspected auth middleware\"},{\"role\":\"user\",\"content\":\"Now optimize login\"},{\"role\":\"assistant\",\"content\":\"Plan ready\"}]",
  "retain_turns": 1,
  "model": "gpt-5.5"
}
```

---

### 🧠 Memory

| Tool | Mô tả |
|---|---|
| `memory_save` | Lưu knowledge vào memory tier (`episodic` / `semantic` / `procedural`) |
| `memory_search` | Tìm kiếm bằng semantic similarity |
| `memory_inject` | Inject memory entries vào system prompt (không cần RAG) |
| `memory_delete` | Xóa entry theo key + type |
| `memory_list` | Liệt kê entries gần nhất |
| `memory_evict` | LRU eviction — giữ top-N entries |
| `memory_stats` | Thống kê theo tier (count, tokens, hits) |

**3 Memory Tiers:**
- `episodic` — sự kiện, lỗi đã gặp, quyết định cụ thể
- `semantic` — kiến thức, patterns, quy tắc dự án
- `procedural` — quy trình, workflow, cách làm

#### `memory_save`
Lưu một mẩu knowledge vào memory store.

**Use case:** lưu convention của repo, quyết định kỹ thuật, hoặc lỗi đã debug xong để các lần sau retrieve/inject lại.

**Ví dụ:**
```json
{
  "key": "auth-refresh-token-policy",
  "value": "Refresh token rotation is mandatory; never reuse old refresh tokens after successful refresh.",
  "mtype": "semantic",
  "tags": "auth,security"
}
```

#### `memory_search`
Tìm memory bằng semantic similarity.

**Use case:** kiểm tra repo đã từng có quyết định hoặc ghi chú liên quan trước khi sửa code.

**Ví dụ:**
```json
{
  "query": "refresh token rotation",
  "mtype": "semantic",
  "top_k": 5,
  "min_sim": 0.12
}
```

#### `memory_inject`
Inject memory liên quan vào `base_prompt` mà không cần đọc source code.

**Use case:** task cần project rules hoặc decision history, nhưng không cần RAG trên file.

**Ví dụ:**
```json
{
  "base_prompt": "Follow project conventions and answer concisely.",
  "query": "implement auth refresh flow",
  "top_k": 4,
  "min_sim": 0.18,
  "output_mode": "concise",
  "model": "gpt-5.5"
}
```

#### `memory_delete`
Xóa entry theo `key` và `mtype`.

**Use case:** loại bỏ memory sai, cũ, hoặc không còn áp dụng.

**Ví dụ:**
```json
{
  "key": "old-auth-rule",
  "mtype": "semantic"
}
```

#### `memory_list`
Liệt kê keys gần nhất, có thể lọc theo tier.

**Use case:** audit nhanh memory đang lưu gì trước khi evict hoặc delete.

**Ví dụ:**
```json
{
  "mtype": "semantic",
  "limit": 20
}
```

#### `memory_evict`
Xóa các memory ít dùng nhất, giữ lại top-N theo LRU.

**Use case:** dọn memory store khi quá nhiều ghi chú cũ làm retrieval nhiễu.

**Ví dụ:**
```json
{
  "keep_top": 300
}
```

#### `memory_stats`
Xem thống kê số lượng, token, lượt hit theo tier.

**Use case:** kiểm tra memory có phình quá lớn hoặc tier nào đang được dùng nhiều.

**Ví dụ:**
```json
{}
```

---

### 📊 Utilities

| Tool | Mô tả |
|---|---|
| `estimate_tokens` | Ước tính số token cho một đoạn text bất kỳ |
| `get_token_budget` | Xem token budget cho model cụ thể (context window, headroom, inject budget) |

#### `estimate_tokens`
Ước tính token và kiểm tra đoạn text có fit trong các model phổ biến không.

**Use case:** trước khi inject prompt dài, kiểm tra kích thước thay vì đoán theo số ký tự.

**Ví dụ:**
```json
{
  "text": "Long prompt or retrieved context here..."
}
```

#### `get_token_budget`
Trả về context window, reserved output/reasoning/tools và inject budget cho model.

**Use case:** chọn model phù hợp hoặc debug vì sao `retrieve_context` chỉ chọn một phần chunks.

**Ví dụ:**
```json
{
  "model": "gpt-5.5"
}
```

---

## 🔄 Luồng hoạt động

```
AI assistant
    └── gọi MCP tool qua stdio
        └── server.py (FastMCP)
            │
            ├── retrieve_context()
            │   ├── AST Chunker       → chunk file theo cú pháp
            │   ├── Embedding Engine  → sentence-transformers vectors
            │   ├── BM25 Index        → sparse keyword scoring
            │   ├── Hybrid Score      → 55% semantic + 20% BM25 + 15% priority/graph
            │   ├── Reranker          → cross-encoder rerank
            │   ├── Graph Expander    → dependency symbol expansion
            │   ├── Compressor        → XML context compression
            │   ├── Memory Store      → inject relevant memories
            │   └── Assembler         → build final enriched prompt
            │
            ├── compact_conversation()
            │   └── Summarizer        → detect format → summarize → inject XML block
            │
            └── memory_*()
                └── MemoryStore (SQLite + LanceDB) → Episodic / Semantic / Procedural
```

---

## ⚙️ Cấu hình

**`.mcp.json`** (workspace-level):
```json
{
  "mcpServers": {
    "harness-context-optimizer": {
      "command": "<repo-path>/.venv/bin/python",
      "args": ["<repo-path>/server.py"],
      "env": { "PYTHONUTF8": "1" }
    }
  }
}
```

**`claude_desktop_config.json`** (`~/Library/Application Support/Claude/`):
```json
{
  "mcpServers": {
    "harness-context-optimizer": {
      "command": "<repo-path>/.venv/bin/python",
      "args": ["<repo-path>/server.py"],
      "env": { "PYTHONUTF8": "1" }
    }
  }
}
```

---

## 🛠 Lệnh hữu ích

```bash
# Xem log realtime
tail -f ~/.mcp-harness/harness-v3.log

# Chạy server thủ công để test
cd Harness-context-engineering
.venv/bin/python server.py

# Cài dependencies
.venv/bin/pip install -r requirements.txt
```

---

## 📌 Ghi chú quan trọng

- Log file chính: `~/.mcp-harness/harness-v3.log`
- Legacy symlink: `~/.gemini/mcp-harness-v3.log` vẫn được tạo tự động nếu hệ thống cho phép
- Config path `~/.gemini/antigravity/mcp_config.json` phải trỏ đến `server.py` (không phải file cũ `compact_conversation_history.py`)
- Server tự động load model embeddings khi khởi động lần đầu (có thể mất vài giây)
