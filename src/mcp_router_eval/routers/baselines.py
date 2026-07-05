"""Baseline routers (§5.1). First implemented: **BM25** — a strong lexical baseline (ADR 0018).

BM25 is a formidable baseline (BEIR), so it is tuned, not left on library defaults: ``k1 = 0.9,
b = 0.4`` (Pyserini's values, ADR 0018) rather than ``rank_bm25``'s ``1.5 / 0.75``. A weak baseline
would make the thesis comparison meaningless.

Per ADR 0018 (+ amendment) the router does **pure ranking only** — it produces ``ranked_tools`` and a
``top_k``; the shared :mod:`~mcp_router_eval.routers.closure` stage does dependency-closure expansion.

The remaining baselines (naive/hybrid vector RAG, GraphRAG-traversal) and the GNN routers stay stubs.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from mcp_router_eval.contracts import ToolSpec
from mcp_router_eval.data.loader import Dataset
from mcp_router_eval.routers.base import RankResult, Router, normalize_confidence, ranked_from_scores

__all__ = ["BM25Router", "DEFAULT_TOP_K", "tool_document"]

#: How many top-ranked tools the router selects before shared closure expansion.
DEFAULT_TOP_K: int = 10

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. ``snake_case`` tool_ids split on ``_`` into their words."""
    return _TOKEN.findall(text.lower())


def tool_document(spec: ToolSpec) -> str:
    """Lexical document for a tool: its name (as words) plus its parameter descriptions.

    The dataset ships no free-text tool description, so the searchable text is the tool_id (which is
    descriptive — ``download_audible_book``) plus the JSON-Schema property ``description`` strings.
    """
    props = spec.schema_.get("properties", {}) or {}
    descriptions = " ".join(
        p["description"] for p in props.values() if isinstance(p, dict) and p.get("description")
    )
    return f"{spec.tool_id.replace('_', ' ')} {descriptions}".strip()


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
        # Fixed tool order (sorted tool_id) → deterministic corpus indexing and tie-breaking.
        self._tool_ids: list[str] = sorted(dataset.tools)
        self._top_k = top_k
        corpus = [_tokenize(tool_document(dataset.tools[t])) for t in self._tool_ids]
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def rank(self, query_text: str, query_id: str) -> RankResult:
        """Score every tool against ``query_text`` and return the full ranking + top-k (no closure)."""
        scores: Sequence[float] = self._bm25.get_scores(_tokenize(query_text))
        ranked = ranked_from_scores(self._tool_ids, scores)
        top_k = [ts.tool_id for ts in ranked[: self._top_k]]
        confidence = normalize_confidence([ts.score for ts in ranked[: self._top_k]])
        return RankResult(
            query_id=query_id,
            query_text=query_text,
            ranked_tools=ranked,
            top_k=top_k,
            confidence=confidence,
            router_name=self.name,
        )
