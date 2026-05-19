"""
embeddings.py — Local TF-IDF + LSA embedding engine
=====================================================
No Hugging Face, no model downloads. Uses scikit-learn TF-IDF + TruncatedSVD
(Latent Semantic Analysis) to produce dense 128-dim vectors locally.

The vectorizer is fitted lazily on the first batch, then updated incrementally
via partial_fit-style re-fit when new texts arrive.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

_DIM = 128


class EmbeddingEngine:
    """Local LSA-based embedding engine (TF-IDF → SVD → L2-normalise)."""

    def __init__(self, dim: int = _DIM):
        self.dim = dim
        self._corpus: list[str] = []
        self._pipe: Pipeline | None = None

    # ── Fitting ──────────────────────────────────────────────────────────────

    def _fit(self, texts: list[str]) -> None:
        """(Re-)fit the pipeline on the accumulated corpus."""
        n_components = min(self.dim, len(texts) - 1) if len(texts) > 1 else 1
        self._pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=20_000,
                sublinear_tf=True,
            )),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ("norm", Normalizer(copy=False)),
        ])
        self._pipe.fit(texts)

    def _ensure_fitted(self, texts: list[str]) -> None:
        """Add texts to corpus and refit if needed."""
        before = len(self._corpus)
        for t in texts:
            if t not in self._corpus:
                self._corpus.append(t)
        if self._pipe is None or len(self._corpus) > before:
            if len(self._corpus) >= 2:
                self._fit(self._corpus)

    def _transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts → dense vectors. Falls back to zeros if not fitted."""
        if self._pipe is None or len(self._corpus) < 2:
            dim = min(self.dim, max(len(self._corpus), 1))
            return np.zeros((len(texts), dim), dtype=np.float32)
        vecs = self._pipe.transform(texts).astype(np.float32)
        # Pad/trim to fixed dim
        if vecs.shape[1] < self.dim:
            pad = np.zeros(
                (vecs.shape[0], self.dim - vecs.shape[1]), dtype=np.float32)
            vecs = np.hstack([vecs, pad])
        return vecs[:, : self.dim]

    # ── Public API ───────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        self._ensure_fitted([text])
        return self._transform([text])[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_fitted(texts)
        return self._transform(texts).tolist()

    def similarity(self, a: str, b: str) -> float:
        vecs = self._transform([a, b])
        dot = float(np.dot(vecs[0], vecs[1]))
        # Vectors are L2-normalised so dot == cosine similarity
        return max(-1.0, min(1.0, dot))
