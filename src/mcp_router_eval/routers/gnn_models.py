"""GNN backbones + late-cosine scorer + node-feature filling (ADR 0010 / 0020 / 0022 / 0025).

Stage 1 of the GNN router: model *definitions* and graph preparation only — **no training loop**
(stage 2) and **no `GNNRouter`** integration (stage 3). Every choice here follows the locked ADRs:

- **Three 2-layer backbones** (ADR 0010 + 0025): R-GCN (`RGCNConv`, `num_relations=4`, ADR 0006/0013),
  GAT (`GATConv`, heads default 2), GraphSAGE (`SAGEConv`, the lower-bound control). Fixed at **2
  layers**, hidden **64**, dropout **0.5**, **no residuals** (ADR 0025); ``hidden`` / ``heads`` /
  ``dropout`` are constructor args so stage-2 can search {32,64,128} / {2,4} / {0.0,0.3,0.5}.
- **Node features** fill graph_build's `x[:, 1:]` placeholder (ADR 0003/0020): `x[:, 0]` stays
  ``is_core``; `x[:, 1:]` = BGE embedding of each tool's ``tool_document()`` (the same text every router
  uses), served from the embedding provider's versioned cache.
- **Late-cosine scoring** (ADR 0022 amendment / two-tower): ``score(query, tool) = cos(query_embed,
  GNN-refined node_embed)`` with per-tower L2 normalization and an **optional per-tower** linear
  projection to a shared space. There is **no** MLP that fuses the query and node embeddings.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, RGCNConv, SAGEConv

from mcp_router_eval.data.graph_build import NUM_RELATIONS, ToolGraph
from mcp_router_eval.data.loader import Dataset
from mcp_router_eval.embedding.base import Embedder
from mcp_router_eval.routers.baselines import tool_document

__all__ = [
    "DEFAULT_HIDDEN",
    "DEFAULT_DROPOUT",
    "DEFAULT_HEADS",
    "DEFAULT_PROJ_DIM",
    "GNNEncoder",
    "RGCNEncoder",
    "GATEncoder",
    "SAGEEncoder",
    "GNNScorer",
    "node_feature_matrix",
]

DEFAULT_HIDDEN: int = 64      # ADR 0025 (search {32, 64, 128})
DEFAULT_DROPOUT: float = 0.5  # ADR 0025 (search {0.0, 0.3, 0.5})
DEFAULT_HEADS: int = 2        # ADR 0025 (GAT only; search {2, 4})
DEFAULT_PROJ_DIM: int = 128   # shared space for the optional per-tower projection


# --------------------------------------------------------------------------- #
# Node features — resolve graph_build's x[:, 1:] embedding placeholder (ADR 0003/0020)
# --------------------------------------------------------------------------- #
def node_feature_matrix(graph: ToolGraph, dataset: Dataset, embedder: Embedder) -> torch.Tensor:
    """Complete node features ``[N, 1 + emb_dim]``: ``is_core`` (col 0) + BGE embedding (cols 1:).

    Embeds each tool's ``tool_document()`` (the SAME text BM25/NaiveRAG use, ADR 0020) in **node order**
    (``graph.node_ids``), reusing the provider cache. Leaves ``x[:, 0]`` (``is_core``) untouched.
    """
    docs = [tool_document(dataset.tools[tid]) for tid in graph.node_ids]
    emb = torch.as_tensor(embedder.encode(docs), dtype=torch.float)  # [N, emb_dim], cached
    is_core = graph.data.x[:, :1].to(torch.float)                    # [N, 1] structural feature
    return torch.cat([is_core, emb], dim=1)                          # [N, 1 + emb_dim]


# --------------------------------------------------------------------------- #
# Backbones — a common 2-layer encoder base + three subclasses
# --------------------------------------------------------------------------- #
class GNNEncoder(nn.Module):
    """2-layer GNN encoder producing graph-refined node embeddings ``[N, out_dim]`` (ADR 0025).

    Fixed structure (ADR 0025): exactly 2 message-passing layers, ReLU + feature dropout between them,
    **no residual connections**. Subclasses only supply the two conv layers via :meth:`_build_convs`.
    """

    #: R-GCN needs ``edge_type`` in its forward; GAT/SAGE do not — keeps the call signature uniform.
    uses_edge_type: bool = False

    def __init__(
        self,
        in_dim: int,
        hidden: int = DEFAULT_HIDDEN,
        *,
        out_dim: int | None = None,
        dropout: float = DEFAULT_DROPOUT,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim if out_dim is not None else hidden
        self.dropout = dropout
        self.num_layers = 2  # fixed (ADR 0025)
        self.conv1, self.conv2 = self._build_convs(in_dim, hidden, self.out_dim, **backbone_kwargs)

    def _build_convs(self, in_dim: int, hidden: int, out_dim: int, **kw):
        raise NotImplementedError

    def _run_conv(self, conv, x, edge_index, edge_type):
        return conv(x, edge_index, edge_type) if self.uses_edge_type else conv(x, edge_index)

    def forward(self, x, edge_index, edge_type=None):
        h = self._run_conv(self.conv1, x, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self._run_conv(self.conv2, h, edge_index, edge_type)  # no residual (ADR 0025)
        return h


class RGCNEncoder(GNNEncoder):
    """Relational GCN over the 4 typed dependency edges (ADR 0006/0013)."""

    uses_edge_type = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = DEFAULT_HIDDEN,
        *,
        out_dim: int | None = None,
        dropout: float = DEFAULT_DROPOUT,
        num_relations: int = NUM_RELATIONS,
    ) -> None:
        super().__init__(in_dim, hidden, out_dim=out_dim, dropout=dropout, num_relations=num_relations)

    def _build_convs(self, in_dim, hidden, out_dim, *, num_relations=NUM_RELATIONS):
        self.num_relations = num_relations
        return (
            RGCNConv(in_dim, hidden, num_relations=num_relations),
            RGCNConv(hidden, out_dim, num_relations=num_relations),
        )


class GATEncoder(GNNEncoder):
    """Graph Attention Network. Layer 1 concatenates heads, layer 2 averages (standard GAT)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = DEFAULT_HIDDEN,
        *,
        out_dim: int | None = None,
        dropout: float = DEFAULT_DROPOUT,
        heads: int = DEFAULT_HEADS,
    ) -> None:
        super().__init__(in_dim, hidden, out_dim=out_dim, dropout=dropout, heads=heads)

    def _build_convs(self, in_dim, hidden, out_dim, *, heads=DEFAULT_HEADS):
        self.heads = heads
        return (
            GATConv(in_dim, hidden, heads=heads, concat=True),        # -> [N, hidden*heads]
            GATConv(hidden * heads, out_dim, heads=heads, concat=False),  # -> [N, out_dim] (avg heads)
        )


class SAGEEncoder(GNNEncoder):
    """GraphSAGE — the lower-bound structural control (ADR 0010)."""

    def _build_convs(self, in_dim, hidden, out_dim):
        return SAGEConv(in_dim, hidden), SAGEConv(hidden, out_dim)


# --------------------------------------------------------------------------- #
# Late-cosine two-tower scorer (ADR 0022 amendment) — NO query-node fusion MLP
# --------------------------------------------------------------------------- #
class GNNScorer(nn.Module):
    """Score tools by late cosine between the query embedding and GNN-refined node embeddings.

    ``score(query, tool) = cos( query_tower(query) , node_tower(GNN(x)) )``. The two towers are
    independent (ADR 0022 amendment): the only cross-tower interaction is the output cosine. The
    optional per-tower linear projections map each side into a shared space (``proj_dim``); each
    projection touches **one** tower only. There is **no** module that ingests query and node embeddings
    jointly — this is a two-tower design, not a cross-encoder.

    ``proj_dim=None`` uses no projection and requires ``encoder.out_dim == query_dim`` (the towers are
    already in one space); otherwise both towers project to ``proj_dim``.
    """

    def __init__(self, encoder: GNNEncoder, query_dim: int, *, proj_dim: int | None = DEFAULT_PROJ_DIM):
        super().__init__()
        self.encoder = encoder
        self.query_dim = query_dim
        node_dim = encoder.out_dim
        if proj_dim is None:
            if node_dim != query_dim:
                raise ValueError(
                    f"proj_dim=None requires encoder.out_dim ({node_dim}) == query_dim ({query_dim}); "
                    "set proj_dim to project into a shared space."
                )
            self.query_proj = None
            self.node_proj = None
            self.shared_dim = query_dim
        else:
            self.query_proj = nn.Linear(query_dim, proj_dim)  # query tower only
            self.node_proj = nn.Linear(node_dim, proj_dim)    # node tower only
            self.shared_dim = proj_dim

    @property
    def has_projection(self) -> bool:
        return self.node_proj is not None

    def node_embeddings(self, x, edge_index, edge_type=None) -> torch.Tensor:
        """L2-normalized node-tower embeddings ``[N, shared_dim]`` (graph refinement + optional proj)."""
        h = self.encoder(x, edge_index, edge_type)
        if self.node_proj is not None:
            h = self.node_proj(h)
        return F.normalize(h, p=2, dim=-1)

    def query_embedding(self, query_vec: torch.Tensor) -> torch.Tensor:
        """L2-normalized query-tower embedding (optional projection); ``[shared_dim]`` or ``[B, dim]``."""
        q = query_vec
        if self.query_proj is not None:
            q = self.query_proj(q)
        return F.normalize(q, p=2, dim=-1)

    def score(self, query_vec: torch.Tensor, x, edge_index, edge_type=None) -> torch.Tensor:
        """Per-node late-cosine scores ``[N]`` for a single query embedding ``query_vec`` (``[query_dim]``)."""
        node = self.node_embeddings(x, edge_index, edge_type)  # [N, d], unit rows
        q = self.query_embedding(query_vec)                    # [d], unit
        return node @ q                                        # cosine (both L2-normed) -> [N]

    def forward(self, query_vec, x, edge_index, edge_type=None):
        return self.score(query_vec, x, edge_index, edge_type)
