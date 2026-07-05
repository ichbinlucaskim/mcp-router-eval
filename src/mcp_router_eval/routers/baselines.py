"""Baseline routers (§5.1): **BM25** (lexical), **NaiveRAG** (dense), **HybridRAG** (fusion),
**Traversal** (Graph RAG-Tool Fusion).

BM25 / NaiveRAG / HybridRAG do **pure ranking only** (ADR 0018 + amendment) — they emit
``ranked_tools`` + a ``top_k`` and the shared :mod:`~mcp_router_eval.routers.closure` stage expands the
dependency closure. Every router consumes the **same** per-tool document text via :func:`tool_document`
(ADR 0020), so the comparison isolates method, not input.

- **BM25** — strong lexical baseline, tuned ``k1=0.9, b=0.4`` (Pyserini, ADR 0018).
- **NaiveRAG** — dense cosine over the embedding provider (LocalEmbedder BGE, ADR 0003); 573 tool
  vectors computed once and served from the provider cache.
- **HybridRAG** — convex-combination fusion of BM25 + NaiveRAG (ADR 0019):
  ``α·norm(dense) + (1−α)·norm(sparse)``, ``α`` default 0.5.
- **Traversal** — reproduces Graph RAG-Tool Fusion's **standard** (no-rerank) method (ADR 0021 +
  amendment): hybrid initial retrieval → per-tool DFS of ``PARAMETER_*`` dependencies →
  **block-interleaved** order. It is the **one** router with its own expansion (intrinsic to the
  method), and it **still** passes through the shared closure stage afterward. The GNN routers stay stubs.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from mcp_router_eval.contract_layer.invariants import Dep
from mcp_router_eval.contracts import ORDERING_RELATIONS, ToolScore, ToolSpec
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
    "TraversalRouter",
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
    """Assemble a :class:`RankResult` from a score vector (shared by the scoring routers)."""
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


# Traversal defaults (Graph RAG-Tool Fusion, ADR 0021 amendment). All are config, recorded per run.
DEFAULT_INITIAL_K: int = 3   # paper's initial hybrid retrieval top-k
DEFAULT_D_LIMIT: int = 3     # per-tool DFS depth limit
DEFAULT_FINAL_TOP_K: int = DEFAULT_TOP_K


def _dfs_dependencies(
    tool: str, tool_deps: Mapping[str, Sequence[Dep]], d_limit: int
) -> list[str]:
    """DFS pre-order of ``tool``'s ``PARAMETER_*`` dependencies, up to per-tool depth ``d_limit``.

    Reuses the loader dependency data (``tool_deps``) and :data:`ORDERING_RELATIONS` (ADR 0013): only
    ``PARAMETER_*`` edges are traversed — ``TOOL_*`` neighbors are never followed (avoids the
    all-neighbor noise/blow-up GraphRunner warns about). Direct dependencies are depth 1. Neighbors are
    visited in a stable sorted order so the DFS is **deterministic**; a source already seen in this DFS
    is not revisited (defensive — the ``PARAMETER_*`` sub-graph is acyclic, ADR 0012).
    """
    out: list[str] = []
    seen: set[str] = set()

    def visit(node: str, depth: int) -> None:
        if depth > d_limit:
            return
        for dep in sorted(tool_deps.get(node, ()), key=lambda d: (d.source, d.relation.value)):
            if dep.relation in ORDERING_RELATIONS and dep.source not in seen:
                seen.add(dep.source)
                out.append(dep.source)
                visit(dep.source, depth + 1)

    visit(tool, 1)
    return out


class TraversalRouter(Router):
    """Graph RAG-Tool Fusion, standard/no-rerank (ADR 0021 + 2026-07-05 amendment).

    Algorithm 1: initial hybrid retrieval (top ``k``) → for each retrieved tool, DFS its ``PARAMETER_*``
    dependencies up to ``d_limit`` (append-if-new) → **block-interleaved** order
    ``[v1, deps(v1), v2, deps(v2), …]`` de-duplicated (first occurrence wins) and truncated to
    ``final_top_k``. This interleaving *is* ``ranked_tools`` — it preserves the initial vector order and
    inserts each tool's dependencies right after it (not a plain closure add, not a score recompute). No
    LLM reranking (the paper's reranker is optional/gpt-4o; excluded for determinism, ADR 0015).

    Still passes through the shared closure stage afterward (ADR 0021), so final ``selected_tools``
    closure-completeness is guaranteed identically to every other router.
    """

    name = "traversal"

    def __init__(
        self,
        hybrid: HybridRAGRouter,
        tool_deps: Mapping[str, Sequence[Dep]],
        *,
        k: int = DEFAULT_INITIAL_K,
        d_limit: int = DEFAULT_D_LIMIT,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> None:
        self._hybrid = hybrid
        self._tool_deps = tool_deps
        self._k = k
        self._d_limit = d_limit
        self._final_top_k = final_top_k

    @property
    def params(self) -> dict:
        """The reproducibility parameters (recorded per run; ADR 0021 amendment)."""
        return {"k": self._k, "d_limit": self._d_limit, "final_top_k": self._final_top_k}

    def rank(self, query_text: str, query_id: str) -> RankResult:
        initial = self._hybrid.rank(query_text, query_id).ranked_tools
        vector_top = [ts.tool_id for ts in initial[: self._k]]

        # Block-interleaving: each vector tool immediately followed by its DFS dependencies.
        interleaved: list[str] = []
        seen: set[str] = set()
        for tool in vector_top:
            if tool not in seen:
                seen.add(tool)
                interleaved.append(tool)
            for dep in _dfs_dependencies(tool, self._tool_deps, self._d_limit):
                if dep not in seen:
                    seen.add(dep)
                    interleaved.append(dep)
        interleaved = interleaved[: self._final_top_k]

        # Order IS the ranking → positional descending scores (traversal is order-based, not scored).
        n = len(interleaved)
        ranked = [ToolScore(tool_id=t, score=float(n - i), rank=i) for i, t in enumerate(interleaved)]
        # Confidence stays score-based (ADR 0018): from the initial hybrid retrieval scores.
        confidence = normalize_confidence([ts.score for ts in initial[: self._k]])
        return RankResult(
            query_id=query_id,
            query_text=query_text,
            ranked_tools=ranked,
            top_k=interleaved,
            confidence=confidence,
            router_name=self.name,
        )
