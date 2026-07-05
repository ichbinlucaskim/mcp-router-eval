"""Router interface — PURE RANKING (ADR 0018 + 2026-07-05 amendment).

A router's only job is to **rank** tools for a query: it emits ``ranked_tools`` (the full ranking, for
retrieval metrics) and a ``top_k`` selection. It does **not** expand the dependency closure — that is a
single shared post-processing stage (``routers/closure.py``) applied identically to every router, so
ablation A swaps only the ranker and holds expansion fixed (ADR 0018 amendment).

This module freezes the interface and provides the router-common helpers every router reuses so the
signals in the final ``RouteResult`` are comparable across heterogeneous routers (ADR 0018):

* :func:`normalize_confidence` — min-max normalization of the top-k candidate scores into ``[0,1]``,
  with the documented ``M_q == m_q`` degenerate rule (RIRAG). The **same** method for every router, so
  the gate and cross-router comparison are fair.
* :data:`HOMOPHILY_NA` — the neutral sentinel non-GNN routers write into ``homophily_local`` (a
  GNN-only neighbor-similarity signal that is meaningless for lexical/vector routers). It exists only
  to satisfy the contract and must never be read as a computed value.

``RankResult`` is the router's raw output; the shared closure stage consumes it and assembles the final
:class:`~mcp_router_eval.contracts.RouteResult`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mcp_router_eval.contracts import ToolScore

__all__ = [
    "HOMOPHILY_NA",
    "RankResult",
    "Router",
    "minmax_normalize",
    "normalize_confidence",
    "ranked_from_scores",
]

#: Sentinel written into ``RouteResult.homophily_local`` by non-GNN routers (ADR 0018). ``homophily``
#: is a GNN-specific signal; ``0.0`` here means "not applicable", NOT a computed neighbor similarity.
HOMOPHILY_NA: float = 0.0


@dataclass(frozen=True)
class RankResult:
    """A router's pure-ranking output (pre-closure). The shared stage turns this into a RouteResult."""

    query_id: str
    query_text: str
    ranked_tools: list[ToolScore]  # full ranking, rank 0 = best (feeds retrieval metrics)
    top_k: list[str]               # top-k tool_ids, best-first (pre-closure selection)
    confidence: float              # normalized to [0,1] (ADR 0018)
    router_name: str               # ablation bookkeeping


class Router(ABC):
    """Abstract router: rank tools for a query. RANKING ONLY — no closure expansion (ADR 0018)."""

    #: Stable router identity for ablation bookkeeping; concrete routers override.
    name: str = "router"

    @abstractmethod
    def rank(self, query_text: str, query_id: str) -> RankResult:
        """Rank all tools for ``query_text`` and return the full ranking + top-k (no closure)."""


def minmax_normalize(scores: Sequence[float]) -> np.ndarray:
    """Min-max normalize scores to ``[0,1]``: ``(s − m)/(M − m)`` (RIRAG, ADR 0018).

    The single min-max implementation, reused for both the confidence self-estimate and hybrid
    convex-combination fusion (ADR 0019). Degenerate rule (ADR 0018): an all-equal window (``M == m``)
    has no spread, so every element maps to the constant ``1.0`` rather than dividing by zero; an empty
    input returns an empty array.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def normalize_confidence(topk_scores: Sequence[float]) -> float:
    """Summarize the top-k candidate scores to a single ``[0,1]`` confidence (ADR 0018).

    Min-max normalizes the top-k window (:func:`minmax_normalize`) and returns the **mean** of the
    normalized scores — one comparable self-estimate, computed the identical way for every router.
    Empty or all-equal windows return the documented constant ``1.0`` (degenerate rule).
    """
    normed = minmax_normalize(topk_scores)
    if normed.size == 0:
        return 1.0
    return float(normed.mean())


def ranked_from_scores(tool_ids: Sequence[str], scores: Sequence[float]) -> list[ToolScore]:
    """Build a descending ``ranked_tools`` list from parallel ``tool_ids`` / ``scores``.

    Ties break by the given ``tool_ids`` order (``np.argsort`` ``kind="stable"`` on negated scores), so
    the ranking is **deterministic** for a fixed input order.
    """
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    return [
        ToolScore(tool_id=tool_ids[i], score=float(scores[i]), rank=rank)
        for rank, i in enumerate(order)
    ]
