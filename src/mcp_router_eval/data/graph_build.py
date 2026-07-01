"""Build the PyG tool graph: nodes=tools, node features=tool-doc embeddings + is_core.

Typed edges use the 4 real dependence types (param x {direct,indirect}, tool x {direct,indirect}),
so num_relations=4 (ADR 0006, 0007). "core" is encoded as an ``is_core`` node feature, not an edge.

STUB.
"""

raise NotImplementedError("data.graph_build: not implemented yet (T3.1)")
