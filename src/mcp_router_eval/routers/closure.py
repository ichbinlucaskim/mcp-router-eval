"""Shared closure-expansion stage (ADR 0018 amendment 2026-07-05).

A single post-processing stage applied **identically to every router**: it takes a router's ``top_k``
and the loader's dependency data and expands it by adding the transitive ``PARAMETER_*`` dependency
closure, then assembles the final :class:`~mcp_router_eval.contracts.RouteResult`. Because expansion
lives here — not inside any router — every router (BM25, vector, traversal, GNN) receives the exact
same closure for a given ``top_k``, so ablation A isolates *ranking* quality and holds expansion fixed.

Only ``PARAMETER_*`` relations expand the selection (a tool needs an argument produced by another tool
— a genuine closure constraint, ADR 0013); ``TOOL_*`` relations are representation-only and are never
pulled in. Ordering/closure semantics reuse :data:`ORDERING_RELATIONS` and the loader's
:func:`~mcp_router_eval.data.loader.topo_order`, the same helpers the executor and invariants use.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from mcp_router_eval.contract_layer.invariants import Dep
from mcp_router_eval.contracts import ORDERING_RELATIONS, Edge, RouteResult
from mcp_router_eval.data.loader import topo_order
from mcp_router_eval.routers.base import HOMOPHILY_NA, RankResult

__all__ = ["expand_closure", "assemble_route_result"]


def expand_closure(
    top_k: Sequence[str], tool_deps: Mapping[str, Sequence[Dep]]
) -> tuple[list[str], list[Edge]]:
    """Expand ``top_k`` with its transitive ``PARAMETER_*`` closure (ADR 0018 amendment / ADR 0013).

    Args:
        top_k: the router's selected tool_ids (pre-closure).
        tool_deps: loader dependency map; only ``PARAMETER_*`` (``ORDERING_RELATIONS``) entries expand.

    Returns:
        ``(selected, closure_edges)`` where ``selected`` is the closed set in topological order
        (dependencies first, :func:`topo_order`), and ``closure_edges`` are the ``PARAMETER_*`` edges
        among the closed set (``Edge(src=dependent, dst=dependency, type=relation)``, sorted for
        determinism). Independent of *which* router produced ``top_k``.
    """
    selected: set[str] = set(top_k)
    # Fixpoint: pull in every PARAMETER_* dependency source until nothing new is added.
    changed = True
    while changed:
        changed = False
        for tool in sorted(selected):
            for dep in tool_deps.get(tool, ()):
                if dep.relation in ORDERING_RELATIONS and dep.source not in selected:
                    selected.add(dep.source)
                    changed = True

    edges: list[Edge] = []
    for tool in sorted(selected):
        for dep in tool_deps.get(tool, ()):
            if dep.relation in ORDERING_RELATIONS and dep.source in selected:
                edges.append(Edge(src=tool, dst=dep.source, type=dep.relation))
    edges.sort(key=lambda e: (e.src, e.dst, e.type.value))

    ordered = topo_order(selected, tool_deps)  # deterministic; deps first (ADR 0012)
    return ordered, edges


def assemble_route_result(
    rank_result: RankResult,
    tool_deps: Mapping[str, Sequence[Dep]],
    *,
    homophily_local: float = HOMOPHILY_NA,
) -> RouteResult:
    """Assemble the final ``RouteResult`` from a router's ranking + the shared closure (ADR 0018).

    Fills ``selected_tools`` / ``closure_edges`` from :func:`expand_closure`, carries the router's
    ``ranked_tools`` / normalized ``confidence`` / ``router_name`` through unchanged, and sets
    ``homophily_local`` (the GNN passes its computed value; non-GNN routers use the
    :data:`~mcp_router_eval.routers.base.HOMOPHILY_NA` sentinel default).
    """
    selected, closure_edges = expand_closure(rank_result.top_k, tool_deps)
    return RouteResult(
        query_id=rank_result.query_id,
        query_text=rank_result.query_text,
        ranked_tools=rank_result.ranked_tools,
        selected_tools=selected,
        closure_edges=closure_edges,
        confidence=rank_result.confidence,
        homophily_local=homophily_local,
        router_name=rank_result.router_name,
    )
