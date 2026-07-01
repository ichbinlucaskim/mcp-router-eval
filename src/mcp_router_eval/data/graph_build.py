"""graph_build — turn loader output into a PyG typed-edge graph the R-GCN/GAT router consumes.

**Representation (verified against PyG guidance — Discussion #4925 + Heterogeneous Learning docs).**
The graph has **one node type** (tool) and **4 edge types**. ``RGCNConv``/``RGATConv`` expect a
*single* node feature matrix ``x``, a *single* ``edge_index``, and an ``edge_type`` vector — **not**
``HeteroData``'s per-type dicts (that is for multiple *node* types, which we do not have). So we build
one :class:`torch_geometric.data.Data` with ``x``, ``edge_index``, ``edge_type`` and
``num_relations = 4``. (If a heterogeneous view is ever needed, ``Data.to_heterogeneous(edge_type=…)``
exists — we are not boxed in.)

**Edge-type integers (stable mapping, ADR 0006/0013):** ``param_direct=0, param_indirect=1,
tool_direct=2, tool_indirect=3`` — see :data:`EDGE_TYPE_TO_INT`.

**Edge direction.** A tool *depends on* its dependency; the dependent should aggregate from its
dependency neighborhood, so messages flow **dependency → dependent** (``edge_index = [src=dependency,
dst=dependent]``). This realizes "a tool's identity depends on its dependency neighborhood."

**Reverse edges — DECISION (Option B).** We keep the **4 directed relations only, no reverse edges**.
Rationale: the thesis is that a tool's identity flows *from* its dependencies; adding reverse edges
would double ``num_relations`` to 8 and blur that directionality before we have evidence it helps.
Reverse edges are recorded as a **future ablation** — the mechanism is to append the flipped
``edge_index`` with new relation ids 4..7 (PyG also ships ``transforms.ToUndirected`` for the untyped
case).

**Node features — embeddings are a documented PLACEHOLDER (ADR 0003), not a PyG convention.**
``x`` currently carries only the **structural** ``is_core`` column (``x[:, 0]``). Text embeddings
(local BGE, behind the embedding provider interface) are produced in their own later step and will be
concatenated as ``x[:, 1:]``. This ordering is a **project sequencing decision**, explicitly *not* a
claimed PyG best practice (no "placeholder-slot" convention was established by search).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

from mcp_router_eval.contracts import EdgeType
from mcp_router_eval.data.loader import Dataset

__all__ = [
    "EDGE_TYPE_TO_INT",
    "INT_TO_EDGE_TYPE",
    "NUM_RELATIONS",
    "IS_CORE_COL",
    "ToolGraph",
    "build_graph",
]

#: Stable EdgeType → integer id used in ``Data.edge_type`` (ADR 0006/0013). Do not reorder.
EDGE_TYPE_TO_INT: dict[EdgeType, int] = {
    EdgeType.PARAM_DIRECT: 0,
    EdgeType.PARAM_INDIRECT: 1,
    EdgeType.TOOL_DIRECT: 2,
    EdgeType.TOOL_INDIRECT: 3,
}
INT_TO_EDGE_TYPE: dict[int, EdgeType] = {v: k for k, v in EDGE_TYPE_TO_INT.items()}
NUM_RELATIONS: int = 4
IS_CORE_COL: int = 0  # x[:, 0] is the is_core structural feature


@dataclass(frozen=True)
class ToolGraph:
    """Frozen graph-object contract the router (and later code) consume.

    Conventions (stable):
      * ``node_ids[i]`` is the tool_id at node index ``i``; nodes are ordered by **sorted tool_id**
        (deterministic, ADR 0008). ``id_to_index`` is the inverse.
      * ``data.x`` layout: column ``IS_CORE_COL`` (0) = ``is_core`` (0/1 float). Text-embedding
        columns are appended later as ``x[:, 1:]`` (ADR 0003; currently absent).
      * ``data.edge_index`` = ``[src=dependency, dst=dependent]``; ``data.edge_type`` ∈ {0,1,2,3}
        per :data:`EDGE_TYPE_TO_INT`. Directed; no reverse edges (Option B).
      * ``num_relations`` = 4 (for ``RGCNConv``/``RGATConv``).
    """

    data: Data
    node_ids: tuple[str, ...]
    id_to_index: dict[str, int]
    num_relations: int = NUM_RELATIONS
    feature_columns: dict[str, int] = field(default_factory=lambda: {"is_core": IS_CORE_COL})

    def index_of(self, tool_id: str) -> int:
        return self.id_to_index[tool_id]

    def tool_at(self, index: int) -> str:
        return self.node_ids[index]


def build_graph(dataset: Dataset) -> ToolGraph:
    """Build the single-``Data`` typed-edge PyG graph from a loaded :class:`Dataset`.

    Nodes = tools (sorted by tool_id). Edges = every dependency, directed dependency→dependent, typed
    by relation. Node features = ``is_core`` only for now (embedding columns appended later).
    """
    node_ids = tuple(sorted(dataset.tools))
    id_to_index = {tid: i for i, tid in enumerate(node_ids)}

    # x: [N, 1] structural feature (is_core). Embedding columns appended in the embedding step.
    x = torch.tensor(
        [[float(dataset.tools[tid].is_core)] for tid in node_ids], dtype=torch.float
    )

    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []
    for tid in node_ids:
        u = id_to_index[tid]  # dependent
        for dep in dataset.tool_deps[tid]:
            v = id_to_index[dep.source]  # dependency
            src.append(v)  # message flows dependency -> dependent
            dst.append(u)
            etype.append(EDGE_TYPE_TO_INT[dep.relation])

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.num_relations = NUM_RELATIONS
    return ToolGraph(data=data, node_ids=node_ids, id_to_index=id_to_index)
