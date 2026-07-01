"""GNN routers: GraphSAGE control + query-conditioned R-GCN/GAT (§5.2).

R-GCN uses num_relations=4 typed messages (ADR 0006); the query is injected as a conditioning
vector. Emits score(v|q) over all tools, then closure expansion (T3.3).

STUB.
"""

raise NotImplementedError("routers.gnn: not implemented yet (T3.2)")
