"""T1.3 — deterministic attribution rule (upstream-wins priority).

Grounded partly in the real q240 Audible chain from docs/completion-scoring-examples.md.
"""
import ast
from pathlib import Path

import pytest

from mcp_router_eval.contract_layer.attribution import attribute
from mcp_router_eval.contracts import (
    Blame,
    ExecResult,
    InvariantReport,
    LatencyMs,
    Outcome,
    RouteResult,
    ToolCall,
)

QID = "q240"
GOLD = ["download_audible_book", "audible_account_login", "validate_email"]


def _route(selected: list[str]) -> RouteResult:
    return RouteResult(
        query_id=QID,
        query_text="download 'The Great Gatsby' from Audible",
        selected_tools=selected,
        confidence=0.8,
        homophily_local=0.1,
        router_name="rgcn",
    )


def _result(completed: bool, call_trace: list[ToolCall] | None = None, qid: str = QID) -> ExecResult:
    return ExecResult(
        query_id=qid,
        trace_id="trace-" + qid,
        call_trace=call_trace or [],
        completed=completed,
        latency_ms=LatencyMs(routing=1, contract=1, execution=1, total=3),
        tools_used=[],
    )


_CLOSURE_OK = InvariantReport(closure_complete=True, dangling_params=[], violations=[])
_CLOSURE_DANGLING = InvariantReport(
    closure_complete=False,
    dangling_params=["audible_account_login.email"],
    violations=[
        "dangling param audible_account_login.email",
        "missing dependency validate_email required by audible_account_login",
    ],
)


# --------------------------------------------------------------------------- #
# Scenario A — success
# --------------------------------------------------------------------------- #
def test_success_is_blame_none():
    att = attribute(_route(GOLD), _result(True), _CLOSURE_OK, GOLD)
    assert att.outcome is Outcome.SUCCESS and att.blame is Blame.NONE
    assert att.query_id == QID


# --------------------------------------------------------------------------- #
# Scenario B — CONTRACT (tools all selected, but dangling param)
# --------------------------------------------------------------------------- #
def test_contract_blame_on_dangling_param():
    # validate_email dropped from selection, but it's NOT a required gold tool omission here:
    # required set is the two the router did select + closure is what broke. To isolate CONTRACT,
    # required_tools ⊆ selected, yet the invariant report shows the dangling param.
    selected = ["download_audible_book", "audible_account_login"]
    att = attribute(_route(selected), _result(False), _CLOSURE_DANGLING, required_tools=selected)
    assert att.outcome is Outcome.FAILURE and att.blame is Blame.CONTRACT
    assert "audible_account_login.email" in att.evidence


# --------------------------------------------------------------------------- #
# Scenario C — EXECUTION (tools present, closure intact, a call failed / out of order)
# --------------------------------------------------------------------------- #
def test_execution_blame_on_failed_call():
    trace = [
        ToolCall(tool_id="download_audible_book", args={}, ok=False,
                 error="arg 'session_id' not yet produced", t_ms=2),
        ToolCall(tool_id="audible_account_login", args={}, ok=True, error=None, t_ms=5),
    ]
    att = attribute(_route(GOLD), _result(False, trace), _CLOSURE_OK, GOLD)
    assert att.blame is Blame.EXECUTION
    assert "download_audible_book" in att.evidence and "session_id" in att.evidence


def test_execution_blame_when_no_failed_call_but_incomplete():
    # completed=False, closure fine, required present, yet no ok=False call -> generic EXECUTION.
    att = attribute(_route(GOLD), _result(False, []), _CLOSURE_OK, GOLD)
    assert att.blame is Blame.EXECUTION and "no ROUTING/CONTRACT cause" in att.evidence


# --------------------------------------------------------------------------- #
# ROUTING — required tool absent, even if downstream also broken
# --------------------------------------------------------------------------- #
def test_routing_blame_when_required_tool_missing():
    selected = ["download_audible_book", "audible_account_login"]  # validate_email (gold) missing
    att = attribute(_route(selected), _result(False), _CLOSURE_DANGLING, required_tools=GOLD)
    assert att.blame is Blame.ROUTING
    assert "validate_email" in att.evidence


# --------------------------------------------------------------------------- #
# PRIORITY / determinism — ROUTING + CONTRACT + EXECUTION all fire -> ROUTING wins
# --------------------------------------------------------------------------- #
def test_upstream_wins_when_all_signals_present():
    selected = ["download_audible_book"]  # both audible_account_login AND validate_email missing (ROUTING)
    trace = [ToolCall(tool_id="download_audible_book", args={}, ok=False, error="boom", t_ms=1)]  # EXECUTION signal
    result = _result(False, trace)
    att = attribute(_route(selected), result, _CLOSURE_DANGLING, required_tools=GOLD)  # CONTRACT signal too
    assert att.blame is Blame.ROUTING  # upstream wins
    # determinism: identical inputs -> identical Attribution
    att2 = attribute(_route(selected), result, _CLOSURE_DANGLING, required_tools=GOLD)
    assert att == att2


def test_contract_beats_execution_when_no_routing():
    # required ⊆ selected (no ROUTING), but closure broken AND a call failed -> CONTRACT wins.
    trace = [ToolCall(tool_id="x", args={}, ok=False, error="boom", t_ms=1)]
    att = attribute(_route(GOLD), _result(False, trace), _CLOSURE_DANGLING, required_tools=GOLD)
    assert att.blame is Blame.CONTRACT


# --------------------------------------------------------------------------- #
# query_id mismatch -> raises
# --------------------------------------------------------------------------- #
def test_query_id_mismatch_raises():
    with pytest.raises(ValueError):
        attribute(_route(GOLD), _result(True, qid="q999"), _CLOSURE_OK, GOLD)


# --------------------------------------------------------------------------- #
# No-import guard (AST) — attribution.py must not import invariants/loader/graph_build
# --------------------------------------------------------------------------- #
def test_attribution_does_not_import_invariants_or_loader():
    src = Path("src/mcp_router_eval/contract_layer/attribution.py").read_text()
    tree = ast.parse(src)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
    banned = ("invariants", "loader", "graph_build")
    offenders = [m for m in modules if any(b in m for b in banned)]
    assert offenders == [], f"attribution.py must not import {banned}; found {offenders}"
