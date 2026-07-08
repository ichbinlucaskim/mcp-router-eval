"""Closure-depth slicing (ADR 0005 + 0024) — PURE functions, no router/executor runs.

Buckets queries by the **size of their PARAMETER_* dependency closure** (ADR 0005): **shallow (2–3)**
vs **deep (≥6)**, with a **medium (4–5)** middle. Depth replaces the single-vs-composite split because
ToolLinkOS has no single-tool queries; the deep bucket isolates the composite regime where the
dependency-aware GNN is expected to win.

Slicing applies **within the test split** (ADR 0024): :func:`slice_by_depth` partitions an
already-selected set of :class:`~mcp_router_eval.eval.metrics.QueryResult` (the test-split results) into
buckets, so every metric in :mod:`eval.metrics` can be reported per bucket. Closure depth itself is
computed by :func:`closure_size` from the ``PARAMETER_*`` sub-graph (reusing the router closure helper).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from mcp_router_eval.contract_layer.invariants import Dep
from mcp_router_eval.eval.metrics import QueryResult
from mcp_router_eval.routers.closure import expand_closure

__all__ = ["SHALLOW", "MEDIUM", "DEEP", "depth_bucket", "closure_size", "slice_by_depth", "partition"]

SHALLOW = "shallow"   # 2–3 tools in the PARAMETER_* closure
MEDIUM = "medium"     # 4–5
DEEP = "deep"         # >= 6


def depth_bucket(depth: int) -> str:
    """Map a closure size to its slice label (ADR 0005): deep ≥6, shallow 2–3, else medium."""
    if depth >= 6:
        return DEEP
    if 2 <= depth <= 3:
        return SHALLOW
    return MEDIUM


def closure_size(seed_tools: Sequence[str], tool_deps: Mapping[str, Sequence[Dep]]) -> int:
    """Number of tools in the ``PARAMETER_*`` closure of ``seed_tools`` (reuses the shared expander).

    A pure function of the gold set + dependency data — it does **not** run any router or executor.
    """
    selected, _ = expand_closure(list(seed_tools), tool_deps)
    return len(selected)


def slice_by_depth(results: Sequence[QueryResult]) -> dict[str, list[QueryResult]]:
    """Partition results into ``{shallow, medium, deep}`` by each result's ``closure_depth`` (ADR 0005).

    Always returns all three keys (possibly empty), so callers can report every bucket uniformly.
    """
    buckets: dict[str, list[QueryResult]] = {SHALLOW: [], MEDIUM: [], DEEP: []}
    for r in results:
        buckets[depth_bucket(r.closure_depth)].append(r)
    return buckets


def partition(results: Sequence[QueryResult], key) -> dict:
    """Generic grouping helper: ``{key(r): [results…]}`` (e.g. by ``router_name`` for per-router views)."""
    out: dict = {}
    for r in results:
        out.setdefault(key(r), []).append(r)
    return out
