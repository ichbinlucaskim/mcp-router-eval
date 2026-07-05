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

__all__ = ["HOMOPHILY_NA", "RankResult", "Router", "normalize_confidence", "ranked_from_scores"]

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


def normalize_confidence(topk_scores: Sequence[float]) -> float:
    """Min-max normalize the top-k candidate scores and summarize to a single ``[0,1]`` confidence.

    Uses ``(s − m)/(M − m)`` over the top-k window (RIRAG, ADR 0018) and returns the **mean** of the
    normalized top-k — one comparable self-estimate, computed the identical way for every router.

    Degenerate rule (ADR 0018): when the window is empty or all scores are equal (``M == m``), there is
    no spread to normalize, so a documented constant ``1.0`` is returned instead of dividing by zero.
    """
    arr = np.asarray(topk_scores, dtype=float)
    if arr.size == 0:
        return 1.0
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return 1.0
    return float(((arr - lo) / (hi - lo)).mean())


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
