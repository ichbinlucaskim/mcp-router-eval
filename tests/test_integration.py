"""Assembly proof — loader → invariants → attribution compose on REAL data (no fixtures).

First end-to-end wiring of the data pipeline into the contract layer, on the real q240 Audible chain.
"""
from pathlib import Path

import pytest

from mcp_router_eval.contract_layer.attribution import attribute
from mcp_router_eval.contract_layer.invariants import check_invariants
from mcp_router_eval.contracts import Blame, ExecResult, LatencyMs, Outcome, RouteResult, ToolCall
from mcp_router_eval.data.loader import Dataset, load

pytestmark = pytest.mark.skipif(
    not (Path("data/processed") / "tools.jsonl").exists(),
    reason="processed data absent; run `python -m mcp_router_eval.data.preprocess`",
)


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return load()


def _route(q, selected):
    return RouteResult(
        query_id=q.query_id, query_text=q.query_text, selected_tools=list(selected),
        confidence=0.8, homophily_local=0.1, router_name="test",
    )


def _failed_result(q):
    return ExecResult(
        query_id=q.query_id, trace_id="t-" + q.query_id,
        call_trace=[ToolCall(tool_id=q.main, args={}, ok=False, error="unsourced arg", t_ms=1)],
        completed=False, latency_ms=LatencyMs(routing=1, contract=1, execution=1, total=3),
        tools_used=[],
    )


def test_full_gold_closure_is_complete(ds):
    """Selecting the entire gold set → closure complete, nothing dangling (real data)."""
    q = ds.query_by_id("q240")
    rep = check_invariants(_route(q, q.required_tools), ds.tool_deps)
    assert rep.closure_complete is True and rep.dangling_params == []


def test_dropping_dependency_dangles_param_on_real_data(ds):
    """Drop the low-homophily param-source validate_email → dangling param (Scenario B, real)."""
    q = ds.query_by_id("q240")
    selected = [t for t in q.required_tools if t != "validate_email"]
    rep = check_invariants(_route(q, selected), ds.tool_deps)
    assert rep.closure_complete is False
    assert rep.dangling_params == ["audible_account_login.email"]


def test_scenario_B_contract_blame_on_real_data(ds):
    """loader → invariants → attribution ⇒ blame=CONTRACT (Scenario B).

    required_tools here is the SELECTED primary set (validate_email treated as a *dependency* the
    contract's closure expansion should have added, not a required primary the router had to surface).
    That isolates the CONTRACT case — matching the completion-scoring doc's primary-vs-dependency
    distinction. (Passing the full gold set instead yields ROUTING; see the next test.)
    """
    q = ds.query_by_id("q240")
    selected = [t for t in q.required_tools if t != "validate_email"]
    route = _route(q, selected)
    rep = check_invariants(route, ds.tool_deps)  # dangling audible_account_login.email
    att = attribute(route, _failed_result(q), rep, required_tools=selected)
    assert att.outcome is Outcome.FAILURE
    assert att.blame is Blame.CONTRACT
    assert "audible_account_login.email" in att.evidence


def test_same_drop_is_ROUTING_when_gold_is_the_required_set(ds):
    """Honesty check: with required_tools = full gold, the SAME drop is ROUTING (upstream-wins).

    validate_email is itself a gold tool, so omitting it is a routing miss that dominates the
    downstream contract symptom — documents the primary-vs-dependency nuance on real data.
    """
    q = ds.query_by_id("q240")
    selected = [t for t in q.required_tools if t != "validate_email"]
    route = _route(q, selected)
    rep = check_invariants(route, ds.tool_deps)
    att = attribute(route, _failed_result(q), rep, required_tools=q.required_tools)
    assert att.blame is Blame.ROUTING
    assert "validate_email" in att.evidence


def test_success_path_blame_none(ds):
    """Full gold selected + completed=True → SUCCESS / NONE (real data)."""
    q = ds.query_by_id("q240")
    route = _route(q, q.required_tools)
    rep = check_invariants(route, ds.tool_deps)
    result = ExecResult(
        query_id=q.query_id, trace_id="t", call_trace=[], completed=True,
        latency_ms=LatencyMs(routing=1, contract=1, execution=1, total=3), tools_used=[],
    )
    att = attribute(route, result, rep, required_tools=q.required_tools)
    assert att.outcome is Outcome.SUCCESS and att.blame is Blame.NONE
