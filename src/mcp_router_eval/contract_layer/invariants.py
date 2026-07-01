"""Contract-layer invariant checks (§3.2, T1.2) — the first consumer of the frozen contracts.

Given a :class:`~mcp_router_eval.contracts.RouteResult` and **injected** ground-truth dependency
info, produce an :class:`~mcp_router_eval.contracts.InvariantReport`
(``closure_complete``, ``dangling_params``, ``violations``).

**Independent of the loader (design constraint).** This module never imports or calls
``data.loader`` / ``data.graph_build``. The dependency table is passed in as a parameter, so the
contract layer can be unit-tested with hand-built fixtures and stays decoupled from data plumbing.
The loader will later produce the same ``tool_deps`` shape from normalized data.

**Only ``PARAMETER_*`` relations matter here (ADR 0013).** ``PARAMETER_*_DEPENDS_ON`` means "this tool
needs an argument value produced by another tool" — a genuine closure/ordering constraint. ``TOOL_*``
relations are conceptual association and are *excluded* from closure and dangling-param checks. The
filter uses :data:`~mcp_router_eval.contracts.ORDERING_RELATIONS`, so a ``TOOL_*`` dependency can never
cause a violation.

**No ``core ⇒ leaf`` assumption.** ``is_core`` is not consulted; a core tool is checked exactly like
any other, so a core tool that has its own param-deps is still validated (inspection: 30/50 core
tools have dependencies).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from mcp_router_eval.contracts import (
    ORDERING_RELATIONS,
    EdgeType,
    InvariantReport,
    RouteResult,
)

__all__ = ["Dep", "check_invariants"]


class Dep(NamedTuple):
    """One ground-truth dependency of a tool, as injected into :func:`check_invariants`.

    A tool's full dependency list may contain any of the 4 relation types; ``check_invariants``
    keeps only those in :data:`ORDERING_RELATIONS` (the ``PARAMETER_*`` subset).

    Attributes:
        source: tool_id (== tool name) that the owning tool depends on.
        param: the required parameter name this dependency sources. ``PARAMETER_*`` edges always
            carry one (per inspection); may be ``None`` for non-parameter relations.
        relation: which of the 4 :class:`EdgeType` relations this dependency is.
    """

    source: str
    param: str | None
    relation: EdgeType


def check_invariants(
    route: RouteResult,
    tool_deps: Mapping[str, Sequence[Dep]],
) -> InvariantReport:
    """Validate the closure of ``route.selected_tools`` against injected dependency info.

    Args:
        route: the router's output; only ``selected_tools`` is consulted here.
        tool_deps: ground-truth dependencies keyed by tool_id. Each value is the tool's full
            dependency list (any relation type); non-``PARAMETER_*`` entries are ignored. A selected
            tool absent from this map is treated as having no dependencies.

    Returns:
        InvariantReport with:
          * ``closure_complete`` — True iff every ``PARAMETER_*`` dependency of every selected tool
            is itself in ``selected_tools``.
          * ``dangling_params`` — ``"tool.param"`` for each required param whose sourcing tool is not
            selected (Scenario B, the thesis-critical case). Sorted, de-duplicated.
          * ``violations`` — human-readable messages for every missing dependency and dangling param.
            Sorted, de-duplicated.
    """
    selected = set(route.selected_tools)
    dangling: set[str] = set()
    violations: set[str] = set()
    closure_complete = True

    for tool in sorted(selected):
        for dep in tool_deps.get(tool, ()):  # PARAMETER_* only
            if dep.relation not in ORDERING_RELATIONS:
                continue
            if dep.source in selected:
                continue
            # Missing param-source dependency: fails closure ...
            closure_complete = False
            violations.add(f"missing dependency {dep.source} required by {tool}")
            # ... and leaves the consuming parameter dangling.
            if dep.param is not None:
                token = f"{tool}.{dep.param}"
                dangling.add(token)
                violations.add(f"dangling param {token}")

    return InvariantReport(
        closure_complete=closure_complete,
        dangling_params=sorted(dangling),
        violations=sorted(violations),
    )
