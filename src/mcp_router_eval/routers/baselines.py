"""Baseline routers (§5.1): **BM25** (lexical), **NaiveRAG** (dense), **HybridRAG** (fusion).

All three do **pure ranking only** (ADR 0018 + amendment) — they emit ``ranked_tools`` + a ``top_k``;
the shared :mod:`~mcp_router_eval.routers.closure` stage expands the dependency closure. Every router
consumes the **same** per-tool document text via :func:`tool_document` (ADR 0020), so the comparison
isolates method, not input.

- **BM25** — strong lexical baseline, tuned ``k1=0.9, b=0.4`` (Pyserini, ADR 0018), not library
  defaults; a weak baseline would make the thesis comparison meaningless (BEIR).
- **NaiveRAG** — dense cosine over the embedding provider (LocalEmbedder BGE, ADR 0003); the 573 tool
  vectors are computed once and served from the provider's versioned cache thereafter.
- **HybridRAG** — convex-combination fusion of BM25 + NaiveRAG (ADR 0019):
  ``α·norm(dense) + (1−α)·norm(sparse)`` over min-max-normalized scores (ADR 0018), ``α`` default 0.5.

The GraphRAG-traversal baseline and the GNN routers stay stubs.
"""
from __future__ import annotations

import re

import numpy as np

from mcp_router_eval.contracts import ToolSpec
from mcp_router_eval.data.loader import Dataset
from mcp_router_eval.embedding.base import Embedder
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.routers.base import (
    RankResult,
    Router,
    minmax_normalize,
    normalize_confidence,
    ranked_from_scores,
)

__all__ = [
    "BM25Router",
    "NaiveRAGRouter",
    "HybridRAGRouter",
    "DEFAULT_TOP_K",
    "tool_document",
]

#: How many top-ranked tools a router selects before shared closure expansion.
DEFAULT_TOP_K: int = 10

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. ``snake_case`` tool_ids split on ``_`` into their words."""
    return _TOKEN.findall(text.lower())


def tool_document(spec: ToolSpec) -> str:
    """The shared per-tool document text every router consumes (ADR 0020).

    Tool name (as words) plus its parameter descriptions. The dataset ships no free-text tool
    description, so the searchable text is the tool_id (which is descriptive — ``download_audible_book``)
    plus the JSON-Schema property ``description`` strings. BM25 tokenizes this; the dense routers embed
    it — the **same** text, so the comparison isolates method, not input.
    """
    props = spec.schema_.get("properties", {}) or {}
    descriptions = " ".join(
        p["description"] for p in props.values() if isinstance(p, dict) and p.get("description")
    )
    return f"{spec.tool_id.replace('_', ' ')} {descriptions}".strip()


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (zero rows left as zero → cosine 0), for cosine similarity."""
    mat = np.asarray(mat, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _rank_result(
    tool_ids: list[str],
    scores: np.ndarray,
    *,
    top_k: int,
    query_text: str,
    query_id: str,
    router_name: str,
) -> RankResult:
    """Assemble a :class:`RankResult` from a score vector (shared by every router — no duplication)."""
    ranked = ranked_from_scores(tool_ids, scores)
    selected = [ts.tool_id for ts in ranked[:top_k]]
    confidence = normalize_confidence([ts.score for ts in ranked[:top_k]])
    return RankResult(
        query_id=query_id,
        query_text=query_text,
        ranked_tools=ranked,
        top_k=selected,
        confidence=confidence,
        router_name=router_name,
    )


class BM25Router(Router):
    """Lexical BM25 router over tool documents (tuned ``k1=0.9, b=0.4``). Pure ranking (ADR 0018)."""

    name = "bm25"

    def __init__(
        self,
        dataset: Dataset,
        *,
        k1: float = 0.9,
        b: float = 0.4,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        from rank_bm25 import BM25Okapi

        # Fixed tool order (sorted tool_id) → deterministic corpus indexing and tie-breaking.
        self._tool_ids: list[str] = sorted(dataset.tools)
        self._top_k = top_k
        corpus = [_tokenize(tool_document(dataset.tools[t])) for t in self._tool_ids]
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b)

    @property
    def tool_ids(self) -> list[str]:
        return list(self._tool_ids)

    def raw_scores(self, query_text: str) -> np.ndarray:
        """BM25 scores for every tool, aligned to :attr:`tool_ids` (for fusion)."""
        return np.asarray(self._bm25.get_scores(_tokenize(query_text)), dtype=float)

    def rank(self, query_text: str, query_id: str) -> RankResult:
        return _rank_result(
            self._tool_ids, self.raw_scores(query_text),
            top_k=self._top_k, query_text=query_text, query_id=query_id, router_name=self.name,
        )


class NaiveRAGRouter(Router):
    """Dense cosine router: embed the shared tool document, cosine-rank vs the query (ADR 0003/0020)."""

    name = "naive_rag"

    def __init__(
        self,
        dataset: Dataset,
        embedder: Embedder | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._tool_ids: list[str] = sorted(dataset.tools)
        self._top_k = top_k
        self._embedder = embedder if embedder is not None else LocalEmbedder()
        # ADR 0020: embed the SAME text BM25 indexes. Cached by the provider → 573 vectors computed once.
        self._documents = [tool_document(dataset.tools[t]) for t in self._tool_ids]
        self._tool_matrix = _l2_normalize(self._embedder.encode(self._documents))  # unit rows for cosine

    @property
    def tool_ids(self) -> list[str]:
        return list(self._tool_ids)

    @property
    def documents(self) -> list[str]:
        """The per-tool document texts embedded (identical to what BM25 indexes; ADR 0020)."""
        return list(self._documents)

    def raw_scores(self, query_text: str) -> np.ndarray:
        """Cosine similarity of the query against every tool vector, aligned to :attr:`tool_ids`."""
        q = self._embedder.encode([query_text])[0]
        q = q / (np.linalg.norm(q) or 1.0)
        return self._tool_matrix @ q

    def rank(self, query_text: str, query_id: str) -> RankResult:
        return _rank_result(
            self._tool_ids, self.raw_scores(query_text),
            top_k=self._top_k, query_text=query_text, query_id=query_id, router_name=self.name,
        )


class HybridRAGRouter(Router):
    """Convex-combination fusion of BM25 (sparse) + NaiveRAG (dense) — ADR 0019.

    ``score = α·norm(dense) + (1−α)·norm(sparse)`` over min-max-normalized scores (ADR 0018). At
    ``α=0`` the ranking reduces to BM25's, at ``α=1`` to NaiveRAG's (min-max is affine, so it preserves
    each ranker's order and confidence at the endpoints).
    """

    name = "hybrid_rag"
    DEFAULT_ALPHA = 0.5

    def __init__(
        self,
        bm25: BM25Router,
        naive: NaiveRAGRouter,
        *,
        alpha: float = DEFAULT_ALPHA,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if bm25.tool_ids != naive.tool_ids:
            raise ValueError("BM25 and NaiveRAG must share tool order to fuse their scores")
        self._bm25 = bm25
        self._naive = naive
        self._alpha = float(alpha)
        self._top_k = top_k
        self._tool_ids = bm25.tool_ids

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def tool_ids(self) -> list[str]:
        return list(self._tool_ids)

    def raw_scores(self, query_text: str) -> np.ndarray:
        """Convex combination of min-max-normalized dense + sparse scores (ADR 0019)."""
        sparse = minmax_normalize(self._bm25.raw_scores(query_text))
        dense = minmax_normalize(self._naive.raw_scores(query_text))
        return self._alpha * dense + (1.0 - self._alpha) * sparse

    def rank(self, query_text: str, query_id: str) -> RankResult:
        return _rank_result(
            self._tool_ids, self.raw_scores(query_text),
            top_k=self._top_k, query_text=query_text, query_id=query_id, router_name=self.name,
        )
