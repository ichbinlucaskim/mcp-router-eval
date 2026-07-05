"""Deterministic mock runner — the executor's PRIMARY layer (ADR 0015).

ToolLinkOS tools are fictional and do not run, and instances carry no gold arguments. This runner
therefore evaluates a **structural proxy** for completion (ADR 0004; see
``docs/completion-scoring-examples.md`` for the worked A/B/C scenarios this implements): it synthesizes
type-valid arguments, calls the plan's tools in the presented order, validates each call's args against
the tool's built JSON Schema, and returns an ``ExecResult`` whose ``completed`` verdict is

    completed == (every required tool invoked)
             AND (call order respects PARAMETER_* dependencies)
             AND (every call succeeds: args type-valid AND every PARAMETER_*-sourced required arg
                  is *available* — its producing tool is present and ran earlier)

**Argument availability is structural, not semantic (ADR 0016 §5).** The runner never threads a
producer's *output value* into a consumer's argument (that is agent-reasoning territory, demonstrated
only via the SDK replay adapter). It only models whether a dependency-sourced required argument
*could* have been produced: if the producing tool is missing from the plan (Scenario B) or is
scheduled to run later (Scenario C), that argument is **unsourced**, so the call fails — exactly as the
scoring doc's call traces show (``ok: false``, "arg not sourced"). Available sourced args still get a
plain synthetic dummy, never the real value.

Everything here is **deterministic** (ADR 0015/0016): identical ``ExecPlan`` in → identical
``ExecResult`` out, *except* the measured latency numbers (ADR 0017: latency is real wall-clock, not
fabricated). That determinism is what lets ablation A isolate the router by holding the executor fixed.

Design notes
------------
* **Argument synthesis (ADR 0016).** Required fields only, canonical per-type dummies, honoring
  ``enum`` (first value) and ``default`` (when present). No RNG. Every ``ToolCall`` is marked
  ``synthetic=True``.
* **Order (ADR 0012).** The runner executes ``plan.bound_tools`` *in the order the plan presents them*
  — an executor runs the plan it is given. The contract layer is expected to present a topologically
  valid order (built via :func:`~mcp_router_eval.data.loader.topo_order`, reused here as the canonical
  reference). The verdict then independently checks that the presented order respects every
  ``PARAMETER_*`` precedence constraint; a reversed plan (Scenario C) fails this check. Only
  ``PARAMETER_*`` edges order; ``TOOL_*`` edges are representation-only (ADR 0013).
* **Failure injection (ADR 0017).** Failures are produced deterministically by point-injection —
  ``arg_overrides`` forces a schema-invalid argument on a chosen tool — never randomly.

This is the mock (structural) layer only; the ``claude-agent-sdk`` replay adapter
(``executor/claude_exec.py``) is a separate, off-critical-path demonstration (ADR 0015).
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from jsonschema import Draft202012Validator

from mcp_router_eval.contract_layer.invariants import Dep
from mcp_router_eval.contracts import (
    ORDERING_RELATIONS,
    ExecPlan,
    ExecResult,
    LatencyMs,
    ToolCall,
)
from mcp_router_eval.data.loader import topo_order

__all__ = ["run", "synthesize_args", "canonical_dummy"]

#: Canonical, fixed dummy value per JSON-Schema primitive type (ADR 0016). Fixed (no RNG) so that
#: synthesis is deterministic. ``object``/``array`` are handled structurally in :func:`_dummy_for`.
_TYPE_DUMMY: dict[str, object] = {
    "string": "synthetic",
    "integer": 0,
    "number": 0.0,
    "boolean": False,
    "array": [],
    "object": {},
}


def canonical_dummy(type_name: str | None) -> object:
    """The fixed dummy for a primitive JSON-Schema ``type`` (ADR 0016).

    Unknown/absent types fall back to the string dummy — a schema that does not constrain a required
    field's type accepts a string, so this stays type-valid.
    """
    if type_name in _TYPE_DUMMY:
        # Return fresh containers so callers can never mutate the shared defaults.
        return {} if type_name == "object" else ([] if type_name == "array" else _TYPE_DUMMY[type_name])
    return _TYPE_DUMMY["string"]


def _dummy_for(pspec: Mapping) -> object:
    """Synthesize one type-valid value for a property spec (ADR 0016).

    Precedence: ``default`` (a value that is valid by construction) → first ``enum`` value → the
    canonical per-type dummy. ``object`` recurses minimally over *its* required properties.
    """
    if "default" in pspec:
        return pspec["default"]
    enum = pspec.get("enum")
    if enum:
        return enum[0]
    if pspec.get("type") == "object":
        return synthesize_args(pspec)
    return canonical_dummy(pspec.get("type"))


def synthesize_args(schema: Mapping) -> dict:
    """Deterministically synthesize the **required** arguments for a tool's JSON Schema (ADR 0016).

    Only ``required`` properties are emitted (minimal, type-valid); ``enum``/``default`` are honored.
    Same schema → identical dict (no RNG).
    """
    props: Mapping = schema.get("properties", {}) or {}
    required: Sequence[str] = schema.get("required", []) or []
    return {name: _dummy_for(props.get(name, {})) for name in required}


def _order_violations(
    call_order: Sequence[str], tool_deps: Mapping[str, Sequence[Dep]]
) -> list[tuple[str, str]]:
    """``(dependency, dependent)`` pairs where a PARAMETER_* dependency is called *after* its dependent.

    Both endpoints must be present in ``call_order`` (a missing dependency is a closure/CONTRACT
    concern, not an ordering one). Empty list ⇒ the presented order is a valid topological order.
    """
    present = set(call_order)
    pos = {tid: i for i, tid in enumerate(call_order)}
    violations: list[tuple[str, str]] = []
    for tool in call_order:
        for dep in tool_deps.get(tool, ()):
            if dep.relation not in ORDERING_RELATIONS:
                continue  # TOOL_* edges do not order (ADR 0013)
            if dep.source in present and pos[dep.source] > pos[tool]:
                violations.append((dep.source, tool))
    return violations


def _sourced_params(tool: str, tool_deps: Mapping[str, Sequence[Dep]]) -> dict[str, list[str]]:
    """``param -> [producing tool_ids]`` for ``tool``'s PARAMETER_* deps that name a param (ADR 0013).

    These are the arguments a real agent would obtain from an upstream tool's output. Param-less
    PARAMETER_* deps impose ordering only (handled by :func:`_order_violations`) and are excluded here.
    """
    out: dict[str, list[str]] = {}
    for dep in tool_deps.get(tool, ()):
        if dep.relation in ORDERING_RELATIONS and dep.param is not None:
            out.setdefault(dep.param, []).append(dep.source)
    return out


def run(
    plan: ExecPlan,
    tool_deps: Mapping[str, Sequence[Dep]],
    required_tools: Sequence[str],
    *,
    routing_ms: int = 0,
    contract_ms: int = 0,
    arg_overrides: Mapping[str, Mapping] | None = None,
) -> ExecResult:
    """Execute ``plan`` on the deterministic mock backend and return an ``ExecResult`` (ADR 0015).

    Args:
        plan: the contract layer's ``ExecPlan``. ``bound_tools`` are called **in the presented order**.
        tool_deps: injected ground-truth dependency map (same shape the contract layer consumes);
            only ``PARAMETER_*`` entries affect ordering (ADR 0013).
        required_tools: the gold tool set. Completion requires every one to have been invoked.
        routing_ms, contract_ms: real wall-clock spent in the upstream layers, measured by the caller
            (ADR 0017). Default 0 when the caller does not measure them. The runner measures its own
            execution loop; ``LatencyMs`` reconciles exactly (``total == routing + contract + execution``).
        arg_overrides: deterministic fault-injection hook (ADR 0017) — ``{tool_id: {param: value}}``
            merged over the synthesized args for that tool, e.g. to force a schema-invalid argument.

    Returns:
        ExecResult with the ordered ``call_trace``, the structural ``completed`` verdict, reconciled
        ``latency_ms``, and ``tools_used``.
    """
    overrides = arg_overrides or {}
    call_order = [ts.tool_id for ts in plan.bound_tools]
    present = set(call_order)
    pos = {tid: i for i, tid in enumerate(call_order)}

    # Reuse the loader's ordering helper (ADR 0012) as an acyclicity guard on the selected sub-graph:
    # a PARAMETER_* cycle here is a data/contract bug, and topo_order raises on it before we execute.
    topo_order(present, tool_deps)

    # --- Execute each tool in the presented order, timing the loop as `execution` (ADR 0017). ---
    call_trace: list[ToolCall] = []
    exec_start = time.perf_counter_ns()
    for i, ts in enumerate(plan.bound_tools):
        required = set(ts.schema_.get("required", []) or [])
        args = synthesize_args(ts.schema_)

        # Drop any PARAMETER_*-sourced required arg whose producer is absent or has not run yet
        # (structural availability, not value-threading): that arg is "unsourced" and fails the call.
        unsourced: list[str] = []
        for param, producers in _sourced_params(ts.tool_id, tool_deps).items():
            if param not in required:
                continue  # optional sourced args are not synthesized; ordering still checked below
            available = any(p in present and pos[p] < i for p in producers)
            if not available:
                args.pop(param, None)
                unsourced.append(param)

        if ts.tool_id in overrides:
            args = {**args, **overrides[ts.tool_id]}  # deterministic point-injection (ADR 0017)
        unsourced = sorted(p for p in unsourced if p not in args)  # an override may re-supply it

        call_start = time.perf_counter_ns()
        errors = sorted(Draft202012Validator(ts.schema_).iter_errors(args), key=str)
        t_ms = (time.perf_counter_ns() - call_start) // 1_000_000

        ok = not unsourced and not errors
        if unsourced:
            error = f"required arg(s) {unsourced} not sourced: producer absent or not yet run"
        elif errors:
            error = errors[0].message
        else:
            error = None
        call_trace.append(
            ToolCall(
                tool_id=ts.tool_id, args=args, ok=ok, error=error, t_ms=t_ms, synthetic=True
            )
        )
    execution_ms = (time.perf_counter_ns() - exec_start) // 1_000_000

    # De-duplicated invocation order (a bound tool appears once, but stay defensive).
    tools_used = list(dict.fromkeys(call_order))

    # --- Structural completion verdict (ADR 0004). ---
    all_required_invoked = set(required_tools) <= set(tools_used)
    order_ok = not _order_violations(call_order, tool_deps)
    all_calls_ok = all(c.ok for c in call_trace)
    completed = all_required_invoked and order_ok and all_calls_ok

    latency_ms = LatencyMs(
        routing=routing_ms,
        contract=contract_ms,
        execution=execution_ms,
        total=routing_ms + contract_ms + execution_ms,
    )

    return ExecResult(
        query_id=plan.query_id,
        trace_id=plan.trace_id,
        call_trace=call_trace,
        completed=completed,
        latency_ms=latency_ms,
        tools_used=tools_used,
    )
