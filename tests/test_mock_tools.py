"""Deterministic mock runner (ADR 0015/0016/0017) — unit tests on hand-built fixtures.

Real-data end-to-end lives in ``test_integration.py``; here we isolate the runner's structural
verdict, argument synthesis, deterministic failure injection, and measured latency with small
fixtures (matching the codebase's unit=fixtures / integration=real-data split).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mcp_router_eval.contract_layer.invariants import Dep
from mcp_router_eval.contracts import (
    EdgeType,
    ExecPlan,
    GateDecision,
    InvariantReport,
    ToolSpec,
)
from mcp_router_eval.executor.mock_tools import run, synthesize_args

# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
_OBJ = "object"


def _spec(tool_id: str, properties: dict, required: list[str]) -> ToolSpec:
    return ToolSpec(
        tool_id=tool_id,
        is_core=False,
        schema={"type": _OBJ, "properties": properties, "required": required},
        deps=[],
    )


def _plan(specs: list[ToolSpec]) -> ExecPlan:
    """An ExecPlan that presents ``specs`` in the given order (that order == the call order)."""
    return ExecPlan(
        query_id="q-test",
        query_text="unit fixture",
        bound_tools=specs,
        invariant_report=InvariantReport(closure_complete=True, dangling_params=[], violations=[]),
        gate_decision=GateDecision.PASS,
        trace_id="t-test",
    )


# Two-tool chain: B needs a value produced by A (PARAMETER_* dependency).
A = _spec("A", {"x": {"type": "string"}}, ["x"])
B = _spec("B", {"session_id": {"type": "string"}}, ["session_id"])
CHAIN_DEPS: dict[str, list[Dep]] = {
    "B": [Dep(source="A", param="session_id", relation=EdgeType.PARAM_DIRECT)],
    "A": [],
}
CHAIN_GOLD = ["A", "B"]


# --------------------------------------------------------------------------- #
# Scenario A — perfect run
# --------------------------------------------------------------------------- #
def test_scenario_A_perfect_completes():
    res = run(_plan([A, B]), CHAIN_DEPS, CHAIN_GOLD)
    assert res.completed is True
    assert all(c.ok for c in res.call_trace)
    assert set(res.tools_used) == set(CHAIN_GOLD)
    assert res.tools_used == ["A", "B"]  # deps-first order preserved
    assert all(c.synthetic for c in res.call_trace)  # ADR 0004


# --------------------------------------------------------------------------- #
# Failure injection (deterministic point-injection, ADR 0017)
# --------------------------------------------------------------------------- #
def test_scenario_C_reversed_order_fails():
    """Deps intact, but B (dependent) is called before A (its PARAMETER_* dep) → completed=False.

    The dependent's sourced arg (``session_id``, produced by A) is not yet available when B runs, so
    B's call fails — matching the scoring doc's Scenario C ("arg not yet produced"). A itself is fine.
    """
    res = run(_plan([B, A]), CHAIN_DEPS, CHAIN_GOLD)
    assert res.completed is False
    by_id = {c.tool_id: c for c in res.call_trace}
    assert by_id["B"].ok is False and "session_id" in by_id["B"].error
    assert by_id["A"].ok is True  # A has no unmet dependency


def test_arg_type_break_fails_that_call_and_completion():
    """Point-inject a schema-invalid arg → that ToolCall.ok=False → completed=False (ADR 0017)."""
    n_tool = _spec("N", {"n": {"type": "integer"}}, ["n"])
    res = run(
        _plan([n_tool]),
        {"N": []},
        ["N"],
        arg_overrides={"N": {"n": "not-an-int"}},
    )
    bad = res.call_trace[0]
    assert bad.ok is False
    assert bad.error is not None
    assert res.completed is False


def test_missing_required_tool_fails_completion():
    """A required tool never invoked (not in the plan) → completed=False (verdict criterion 1)."""
    res = run(_plan([A]), CHAIN_DEPS, CHAIN_GOLD)  # gold needs A and B; only A bound
    assert res.completed is False
    assert set(res.tools_used) == {"A"}


# --------------------------------------------------------------------------- #
# Argument synthesis honors enum / default (ADR 0016)
# --------------------------------------------------------------------------- #
def test_enum_param_gets_first_value():
    schema = {
        "type": _OBJ,
        "properties": {"category": {"type": "string", "enum": ["movie", "tv_show", "game"]}},
        "required": ["category"],
    }
    assert synthesize_args(schema) == {"category": "movie"}


def test_default_param_is_used_and_wins_over_enum():
    schema = {
        "type": _OBJ,
        "properties": {
            "language": {"type": "string", "enum": ["english", "spanish"], "default": "english"}
        },
        "required": ["language"],
    }
    assert synthesize_args(schema) == {"language": "english"}


def test_required_only_and_canonical_type_dummies():
    schema = {
        "type": _OBJ,
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "f": {"type": "number"},
            "b": {"type": "boolean"},
            "arr": {"type": "array"},
            "obj": {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]},
            "optional": {"type": "string"},  # not required → must be omitted
        },
        "required": ["s", "i", "f", "b", "arr", "obj"],
    }
    assert synthesize_args(schema) == {
        "s": "synthetic",
        "i": 0,
        "f": 0.0,
        "b": False,
        "arr": [],
        "obj": {"k": "synthetic"},  # object recurses minimally over its required props
    }


# --------------------------------------------------------------------------- #
# Determinism (ADR 0015) — identical ExecResult except latency numbers
# --------------------------------------------------------------------------- #
def _fingerprint(res):
    return (
        [(c.tool_id, c.args, c.ok, c.error, c.synthetic) for c in res.call_trace],
        res.completed,
        res.tools_used,
    )


def test_same_plan_twice_is_identical_modulo_latency():
    r1 = run(_plan([A, B]), CHAIN_DEPS, CHAIN_GOLD)
    r2 = run(_plan([A, B]), CHAIN_DEPS, CHAIN_GOLD)
    assert _fingerprint(r1) == _fingerprint(r2)  # trace/args/completed/tools_used identical
    # Latency numbers are measured wall-clock and may differ; structure still reconciles (below).


# --------------------------------------------------------------------------- #
# Latency (ADR 0017) — measured, reconciles; no absolute-value assertions
# --------------------------------------------------------------------------- #
def test_latency_reconciles_and_is_structured():
    res = run(_plan([A, B]), CHAIN_DEPS, CHAIN_GOLD, routing_ms=5, contract_ms=3)
    lat = res.latency_ms
    # All three layers present + non-negative (LatencyMs enforces ge=0; assert structure explicitly).
    assert lat.routing >= 0 and lat.contract >= 0 and lat.execution >= 0
    assert lat.routing == 5 and lat.contract == 3  # caller-measured values preserved
    assert lat.total == lat.routing + lat.contract + lat.execution  # exact reconcile
    assert all(c.t_ms >= 0 for c in res.call_trace)
    # NB: no assertion on absolute latency — environment-dependent (ADR 0017).


# --------------------------------------------------------------------------- #
# q240 real-data end-to-end (Audible spine)
# --------------------------------------------------------------------------- #
_HAS_DATA = (Path("data/processed") / "tools.jsonl").exists()


@pytest.mark.skipif(not _HAS_DATA, reason="processed data absent; run preprocess")
def test_q240_audible_spine_completes():
    from mcp_router_eval.data.loader import load

    ds = load()
    q = ds.query_by_id("q240")
    order = ds.execution_order(q.required_tools)  # topo-sorted (deps first)
    plan = _plan([ds.tools[t] for t in order])

    res = run(plan, ds.tool_deps, q.required_tools)
    assert res.completed is True
    assert all(c.ok for c in res.call_trace)
    assert set(res.tools_used) == set(q.required_tools)

    # enum+default honored on real data: download_audible_book.language defaults to "english".
    dl = next(c for c in res.call_trace if c.tool_id == "download_audible_book")
    assert dl.args == {"session_id": "synthetic", "book_name": "synthetic", "language": "english"}
