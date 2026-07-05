"""GNNRouter — the trained GNN as a Router (ADR 0018 / 0022 / 0027). Stage 3, final GNN piece.

Wraps a trained :class:`~mcp_router_eval.routers.gnn_models.GNNScorer` so the GNN stands beside the four
baselines behind the same :class:`~mcp_router_eval.routers.base.Router` contract:

- **Pure ranking** (ADR 0018): ``rank(query)`` computes the node embeddings **once** (cached per router
  instance — the trained weights and graph are fixed), embeds the query with the **same BGE provider**
  the baselines use, and scores by **late-cosine matmul** (``node @ query``; ADR 0022 amendment — no
  query-node fusion MLP). The shared closure stage runs separately, exactly like the other routers.
- **confidence**: min-max normalized to ``[0,1]`` (ADR 0018), the same helper the baselines use.
- **homophily_local** (ADR 0027, **GNN-only real value**): for a tool ``t`` with ``PARAMETER_*``
  dependencies, the mean cosine between ``t``'s GNN-refined embedding and its dependency neighbors'
  embeddings; a tool with **no** ``PARAMETER_*`` deps yields the ADR-0018 **sentinel** (not a fabricated
  ``0``). :meth:`route` fills the ``RouteResult`` scalar with the mean over the selected set's
  real values. Baseline routers keep emitting the sentinel — unchanged.

Deterministic: a fixed checkpoint + a fixed query → identical scores. Works for all three backbones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mcp_router_eval.contracts import ORDERING_RELATIONS, RouteResult
from mcp_router_eval.data.graph_build import ToolGraph
from mcp_router_eval.data.loader import Dataset
from mcp_router_eval.embedding.base import Embedder
from mcp_router_eval.routers.base import (
    HOMOPHILY_NA,
    RankResult,
    Router,
    normalize_confidence,
    ranked_from_scores,
)
from mcp_router_eval.routers.baselines import DEFAULT_TOP_K
from mcp_router_eval.routers.closure import assemble_route_result, expand_closure
from mcp_router_eval.routers.gnn_models import GNNScorer, node_feature_matrix
from mcp_router_eval.routers.gnn_train import GNNTrainConfig, build_scorer

__all__ = ["GNNRouter"]


class GNNRouter(Router):
    """A trained GNN scorer exposed as a pure-ranking Router (ADR 0018/0022/0027)."""

    def __init__(
        self,
        scorer: GNNScorer,
        graph: ToolGraph,
        dataset: Dataset,
        embedder: Embedder,
        *,
        backbone: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._scorer = scorer.eval()  # inference mode: no dropout → deterministic
        self._embedder = embedder
        self._tool_ids = list(graph.node_ids)
        self._id_to_index = dict(graph.id_to_index)
        self._tool_deps = dataset.tool_deps
        self._x = node_feature_matrix(graph, dataset, embedder)
        self._edge_index = graph.data.edge_index
        self._edge_type = graph.data.edge_type
        self._top_k = top_k
        self._node: torch.Tensor | None = None  # cached L2-normalized node embeddings
        self.name = f"gnn_{backbone}" if backbone else "gnn"

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        dataset: Dataset,
        graph: ToolGraph,
        embedder: Embedder,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> GNNRouter:
        """Rebuild the scorer from the checkpoint's saved config, load weights, wrap as a router."""
        ckpt = torch.load(Path(path), weights_only=False)
        config = GNNTrainConfig(**ckpt["config"])
        in_dim = node_feature_matrix(graph, dataset, embedder).shape[1]
        scorer = build_scorer(config, in_dim, query_dim=embedder.dim)
        scorer.load_state_dict(ckpt["state_dict"])
        return cls(scorer, graph, dataset, embedder, backbone=config.backbone, top_k=top_k)

    # ---- node embeddings: one forward, cached per instance ----------------- #
    def _node_embeddings(self) -> torch.Tensor:
        if self._node is None:
            with torch.no_grad():
                self._node = self._scorer.node_embeddings(self._x, self._edge_index, self._edge_type)
        return self._node  # [N, d], L2-normalized (unit rows) → cosine via dot product

    # ---- pure ranking (ADR 0018) ------------------------------------------ #
    def rank(self, query_text: str, query_id: str) -> RankResult:
        node = self._node_embeddings()
        with torch.no_grad():
            q_raw = torch.as_tensor(self._embedder.encode([query_text])[0], dtype=torch.float)
            q = self._scorer.query_embedding(q_raw)          # [d], unit
            scores = (node @ q).cpu().numpy()                # [N] late cosine (matmul, no fusion MLP)
        ranked = ranked_from_scores(self._tool_ids, scores)
        top_k = [ts.tool_id for ts in ranked[: self._top_k]]
        confidence = normalize_confidence([ts.score for ts in ranked[: self._top_k]])
        return RankResult(
            query_id=query_id, query_text=query_text, ranked_tools=ranked,
            top_k=top_k, confidence=confidence, router_name=self.name,
        )

    # ---- homophily_local (ADR 0027, GNN-only) ----------------------------- #
    def tool_homophily(self, tool_id: str) -> float:
        """Mean cosine of ``tool_id`` to its ``PARAMETER_*`` dependency neighbors; sentinel if none."""
        ti = self._id_to_index.get(tool_id)
        if ti is None:
            return HOMOPHILY_NA
        dep_idx = [
            self._id_to_index[d.source]
            for d in self._tool_deps.get(tool_id, ())
            if d.relation in ORDERING_RELATIONS and d.source in self._id_to_index
        ]
        if not dep_idx:
            return HOMOPHILY_NA  # no PARAMETER_* deps → ADR-0018 sentinel (not a fabricated 0)
        node = self._node_embeddings()
        with torch.no_grad():
            sims = node[ti] @ node[dep_idx].T  # unit rows → cosine
        return float(sims.mean())

    def _has_param_deps(self, tool_id: str) -> bool:
        return any(
            d.relation in ORDERING_RELATIONS and d.source in self._id_to_index
            for d in self._tool_deps.get(tool_id, ())
        )

    def homophily_local(self, tools: list[str]) -> float:
        """Scalar ``RouteResult.homophily_local``: mean of the per-tool values over dependency-having
        selected tools (ADR 0027); the sentinel if none of them has ``PARAMETER_*`` deps."""
        vals = [self.tool_homophily(t) for t in tools if self._has_param_deps(t)]
        return float(np.mean(vals)) if vals else HOMOPHILY_NA

    # ---- full pipeline: pure ranking → shared closure → RouteResult ------- #
    def route(self, query_text: str, query_id: str) -> RouteResult:
        """Rank, then pass through the shared closure stage with the GNN-computed homophily (ADR 0021)."""
        rank_result = self.rank(query_text, query_id)
        selected, _ = expand_closure(rank_result.top_k, self._tool_deps)
        return assemble_route_result(
            rank_result, self._tool_deps, homophily_local=self.homophily_local(selected)
        )
