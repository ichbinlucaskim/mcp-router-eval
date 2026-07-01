"""Deterministic failure attribution (§3.4, T1.3) — the project's differentiator.

Given a finished run, assign blame to **ROUTING / CONTRACT / EXECUTION** deterministically, or
``NONE`` on success. This is the piece that turns "the task failed" into "the *router* failed" vs
"the *contract* failed" vs "the *executor* failed".

**Reuse via data, not via call.** This module imports **only**
:mod:`mcp_router_eval.contracts`. It does *not* import or call ``invariants`` / ``data.loader`` /
``data.graph_build``. The :class:`~mcp_router_eval.contracts.InvariantReport` is passed **in** as
already-computed data: the pipeline runs the invariant check once and hands the report here, reusing
that work through the frozen contract rather than a function call. The gold tool set
(``required_tools``) is likewise injected, keeping attribution loader-independent.

**Upstream-wins priority.** When a run fails, several signals often fire at once — a tool the router
never surfaced (ROUTING) will *also* show up as a broken closure (CONTRACT) and a failed call
(EXECUTION) downstream. Attributing to the **earliest** cause in the ROUTING → CONTRACT → EXECUTION
chain is what makes blame **actionable** (fix the router, and the downstream symptoms disappear) and
**non-ambiguous** (exactly one cause per failure). The strict ordering below is therefore the whole
point, not an implementation detail.
"""
from __future__ import annotations

from collections.abc import Sequence

from mcp_router_eval.contracts import (
    Attribution,
    Blame,
    ExecResult,
    InvariantReport,
    Outcome,
    RouteResult,
)

__all__ = ["attribute"]


def attribute(
    route: RouteResult,
    result: ExecResult,
    invariant_report: InvariantReport,
    required_tools: Sequence[str],
) -> Attribution:
    """Deterministically attribute a run's outcome to a single blame target.

    Args:
        route: router output (``selected_tools`` is the routing evidence).
        result: executor output (``completed`` is the success verdict; ``call_trace`` the exec evidence).
        invariant_report: the already-computed contract-layer report (injected, not recomputed here).
        required_tools: the gold tool set (``golden_function_names``), injected.

    Returns:
        contracts.Attribution — ``{query_id, outcome, blame, evidence}``.

    Raises:
        ValueError: if ``route.query_id`` and ``result.query_id`` disagree (a programming error —
            the two halves of one run must share an id).
    """
    if route.query_id != result.query_id:
        raise ValueError(
            f"query_id mismatch: route={route.query_id!r} result={result.query_id!r} "
            "(route and result must belong to the same run)"
        )
    query_id = route.query_id

    # Outcome first: success is purely the (structural-proxy) completion verdict.
    if result.completed:
        return Attribution(
            query_id=query_id,
            outcome=Outcome.SUCCESS,
            blame=Blame.NONE,
            evidence="completed=True; no failure to attribute.",
        )

    # --- FAILURE: assign blame by the FIRST matching cause, upstream → downstream. ---
    selected = set(route.selected_tools)

    # 1) ROUTING — a required tool was never surfaced by the router.
    missing = sorted(t for t in required_tools if t not in selected)
    if missing:
        return Attribution(
            query_id=query_id,
            outcome=Outcome.FAILURE,
            blame=Blame.ROUTING,
            evidence=f"required tool(s) missing from selection: {missing}",
        )

    # 2) CONTRACT — routing had the tools, but the closure is broken (incomplete / dangling param).
    if not invariant_report.closure_complete or invariant_report.dangling_params:
        if invariant_report.dangling_params:
            detail = f"dangling params {sorted(invariant_report.dangling_params)}"
        else:
            detail = "closure incomplete"
        if invariant_report.violations:
            detail += f"; violations={sorted(invariant_report.violations)}"
        return Attribution(
            query_id=query_id,
            outcome=Outcome.FAILURE,
            blame=Blame.CONTRACT,
            evidence=f"contract closure violation: {detail}",
        )

    # 3) EXECUTION — tools present and closure intact, but the run still failed.
    #    Prefer the first failed call (stable by call_trace order); this also covers an out-of-order
    #    call, which surfaces as a ToolCall with ok=False (e.g. a required arg not yet produced).
    first_failed = next((c for c in result.call_trace if not c.ok), None)
    if first_failed is not None:
        evidence = (
            f"execution failure: tool {first_failed.tool_id!r} call failed: "
            f"{first_failed.error or 'no error message'}"
        )
    else:
        evidence = (
            "execution failure: completed=False with all required tools present and closure intact "
            "(no ROUTING/CONTRACT cause)."
        )
    return Attribution(
        query_id=query_id,
        outcome=Outcome.FAILURE,
        blame=Blame.EXECUTION,
        evidence=evidence,
    )
