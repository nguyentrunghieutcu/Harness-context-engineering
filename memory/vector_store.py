"""
vector_store.py — In-memory numpy vector store
===============================================
No LanceDB, no PyArrow, no fixed dimension constraint.
Stores chunk embeddings in RAM and performs brute-force cosine search.
Fast enough for typical workspaces (<10k chunks).
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    """
    In-memory cosine-similarity vector store.
    Replaces lancedb to remove the fixed 384-dim Hugging Face constraint.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._paths: list[str] = []
        self._matrix: np.ndarray | None = None  # shape (N, D)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _index_of(self, id_: str) -> int | None:
        try:
            return self._ids.index(id_)
        except ValueError:
            return None

    # ── Public API ───────────────────────────────────────────────────────────

    def upsert(self, chunks: list) -> None:
        """Insert or overwrite chunk embeddings."""
        for c in chunks:
            if not c.embedding:
                continue
            vec = np.array(c.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm  # pre-normalise for cosine via dot

            idx = self._index_of(c.id)
            if idx is not None:
                # Update in-place
                self._matrix[idx] = vec  # type: ignore[index]
                self._paths[idx] = c.path
            else:
                # Append
                self._ids.append(c.id)
                self._paths.append(c.path)
                if self._matrix is None:
                    self._matrix = vec.reshape(1, -1)
                else:
                    # Pad/trim to current dim if needed
                    d = self._matrix.shape[1]
                    if vec.shape[0] < d:
                        vec = np.pad(vec, (0, d - vec.shape[0]))
                    elif vec.shape[0] > d:
                        vec = vec[:d]
                    self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])

    def search(self, query_vec: list[float], top_k: int = 50) -> list[dict]:
        """Return top-k chunks by cosine similarity."""
        if self._matrix is None or len(self._ids) == 0:
            return []

        q = np.array(query_vec, dtype=np.float32)
        # Match dimension
        d = self._matrix.shape[1]
        if q.shape[0] < d:
            q = np.pad(q, (0, d - q.shape[0]))
        elif q.shape[0] > d:
            q = q[:d]

        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        # Brute-force cosine (dot of normalised vectors)
        scores = self._matrix @ q  # shape (N,)
        n = min(top_k, len(self._ids))
        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            {"id": self._ids[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def count(self) -> int:
        return len(self._ids)
