"""BM25 router + shared closure stage (ADR 0018 + 2026-07-05 amendment).

Covers: the pure-ranking Router interface, the shared closure stage (identical for every router —
ablation-A hygiene), normalized confidence, the homophily sentinel, determinism, and the full real-data
path BM25 → closure → RouteResult → invariants → mock executor → attribution (incl. a dependency-drop
that yields blame=CONTRACT *through the real router*).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mcp_router_eval.contract_layer.attribution import attribute
from mcp_router_eval.contract_layer.invariants import Dep, check_invariants
from mcp_router_eval.contracts import (
    Blame,
    EdgeType,
    ExecPlan,
    GateDecision,
    Outcome,
    RouteResult,
)
from mcp_router_eval.data.loader import load, topo_order
from mcp_router_eval.eval.harness import variant_a_required_set
from mcp_router_eval.executor.mock_tools import run as mock_run
from mcp_router_eval.routers.base import HOMOPHILY_NA, RankResult, Router, normalize_confidence
from mcp_router_eval.routers.baselines import BM25Router
from mcp_router_eval.routers.closure import assemble_route_result, expand_closure

_HAS_DATA = (Path("data/processed") / "tools.jsonl").exists()
pytestmark = pytest.mark.skipif(_HAS_DATA is False, reason="processed data absent; run preprocess")


@pytest.fixture(scope="module")
def ds():
    return load()


@pytest.fixture(scope="module")
def bm25(ds):
    return BM25Router(ds)


# --------------------------------------------------------------------------- #
# Interface — BM25 is a Router doing pure ranking; assembly yields a valid RouteResult
# --------------------------------------------------------------------------- #
def test_bm25_is_a_router(bm25):
    assert isinstance(bm25, Router)
    assert bm25.name == "bm25"


def test_rank_returns_full_ranking_and_topk(bm25, ds):
    q = ds.query_by_id("q240")
    rr = bm25.rank(q.query_text, q.query_id)
    assert isinstance(rr, RankResult)
    assert len(rr.ranked_tools) == len(ds.tools)          # full ranking
    assert [t.rank for t in rr.ranked_tools] == list(range(len(ds.tools)))  # 0..N-1
    assert len(rr.top_k) == 10 and rr.top_k[0] == rr.ranked_tools[0].tool_id


def test_assembly_produces_valid_route_result(bm25, ds):
    q = ds.query_by_id("q240")
    route = assemble_route_result(bm25.rank(q.query_text, q.query_id), ds.tool_deps)
    assert isinstance(route, RouteResult)  # pydantic extra="forbid" would have raised on drift
    assert route.router_name == "bm25"
    assert route.selected_tools and route.query_id == "q240"


# --------------------------------------------------------------------------- #
# Real-data ranking sanity — the gold main tool ranks at the very top
# --------------------------------------------------------------------------- #
def test_gold_main_tool_ranks_highly(bm25, ds):
    q = ds.query_by_id("q240")
    rr = bm25.rank(q.query_text, q.query_id)
    rank_of_main = next(t.rank for t in rr.ranked_tools if t.tool_id == q.main)
    assert rank_of_main == 0                    # download_audible_book is #1 for the Audible query
    assert q.main in rr.top_k


# --------------------------------------------------------------------------- #
# Shared closure stage — adds PARAMETER_* deps only (TOOL_* excluded), deterministic
# --------------------------------------------------------------------------- #
def test_closure_adds_param_deps_excludes_tool_edges():
    """Synthetic: C→B (param), B→A (param), C→T (TOOL_*). Closure of {C} = {A,B,C}; T excluded."""
    tool_deps = {
        "C": [
            Dep(source="B", param="x", relation=EdgeType.PARAM_DIRECT),
            Dep(source="T", param="z", relation=EdgeType.TOOL_DIRECT),  # must NOT expand
        ],
        "B": [Dep(source="A", param="y", relation=EdgeType.PARAM_INDIRECT)],
        "A": [],
    }
    selected, edges = expand_closure(["C"], tool_deps)
    assert selected == ["A", "B", "C"]          # topo order, deps first
    assert "T" not in selected                  # TOOL_* dependency not pulled in (ADR 0013)
    assert {(e.src, e.dst, e.type.value) for e in edges} == {
        ("C", "B", "param_direct"),
        ("B", "A", "param_indirect"),
    }


def test_closure_on_real_audible_spine(ds):
    selected, edges = expand_closure(["download_audible_book"], ds.tool_deps)
    assert set(selected) == {
        "download_audible_book",
        "audible_account_login",
        "validate_email",
        "get_system_language",
    }
    # TOOL_*-attached gold tools (wifi/battery) are never pulled in by closure.
    assert "get_wifi_status" not in selected and "get_battery_status" not in selected
    assert ("download_audible_book", "audible_account_login") in {(e.src, e.dst) for e in edges}


def test_closure_identical_regardless_of_router(ds):
    """Ablation-A hygiene: same top_k → identical closure, whichever 'router' produced it."""
    top_k = ["download_audible_book", "read_audible_book"]

    def route_from(router_name: str) -> RouteResult:
        rr = RankResult(
            query_id="q240", query_text="x", ranked_tools=[], top_k=top_k,
            confidence=0.5, router_name=router_name,
        )
        return assemble_route_result(rr, ds.tool_deps)

    a, b = route_from("bm25"), route_from("some_other_router")
    assert a.selected_tools == b.selected_tools
    assert [(e.src, e.dst, e.type.value) for e in a.closure_edges] == \
           [(e.src, e.dst, e.type.value) for e in b.closure_edges]


# --------------------------------------------------------------------------- #
# Confidence normalization + homophily sentinel (ADR 0018)
# --------------------------------------------------------------------------- #
def test_confidence_in_unit_range(bm25, ds):
    q = ds.query_by_id("q240")
    rr = bm25.rank(q.query_text, q.query_id)
    assert 0.0 <= rr.confidence <= 1.0


def test_confidence_degenerate_rule():
    assert normalize_confidence([]) == 1.0            # empty window → constant
    assert normalize_confidence([5.0, 5.0, 5.0]) == 1.0  # M == m → constant (no div-by-zero)
    assert normalize_confidence([0.0, 1.0]) == 0.5    # (0,1) min-max mean


def test_homophily_sentinel_for_bm25(bm25, ds):
    q = ds.query_by_id("q240")
    route = assemble_route_result(bm25.rank(q.query_text, q.query_id), ds.tool_deps)
    assert route.homophily_local == HOMOPHILY_NA == 0.0  # GNN-only signal; n/a for lexical BM25


# --------------------------------------------------------------------------- #
# Determinism — same query → identical RouteResult
# --------------------------------------------------------------------------- #
def test_determinism(bm25, ds):
    q = ds.query_by_id("q240")
    r1 = assemble_route_result(bm25.rank(q.query_text, q.query_id), ds.tool_deps)
    r2 = assemble_route_result(bm25.rank(q.query_text, q.query_id), ds.tool_deps)
    assert r1.model_dump() == r2.model_dump()


# --------------------------------------------------------------------------- #
# Full path on real data — BM25 → closure → RouteResult → invariants → executor → attribution
# --------------------------------------------------------------------------- #
def _plan(ds, route, rep):
    order = topo_order(route.selected_tools, ds.tool_deps)
    return ExecPlan(
        query_id=route.query_id, query_text=route.query_text,
        bound_tools=[ds.tools[t] for t in order], invariant_report=rep,
        gate_decision=GateDecision.PASS, trace_id="t-" + route.query_id,
    )


def test_full_path_success_through_bm25(ds, bm25):
    # De-circularized (ADR-0030, checkup step 5): completion is scored against the **variant-A
    # required-set** (the required-arg PARAMETER_* spine the harness uses), NOT route.selected_tools —
    # which trivially contains its own closure and passes tautologically. This now verifies that BM25
    # actually RECOVERED the required-arg tools (download_audible_book → audible_account_login →
    # validate_email) for q240.
    q = ds.query_by_id("q240")
    route = assemble_route_result(bm25.rank(q.query_text, q.query_id), ds.tool_deps)
    rep = check_invariants(route, ds.tool_deps)
    assert rep.closure_complete is True             # the shared stage completes the closure
    required = variant_a_required_set(q.main, ds.tool_deps)
    assert required <= set(route.selected_tools)    # REAL check: router surfaced the required-arg spine
    res = mock_run(_plan(ds, route, rep), ds.tool_deps, list(required))
    att = attribute(route, res, rep, required_tools=list(required))
    assert res.completed is True
    assert att.outcome is Outcome.SUCCESS and att.blame is Blame.NONE


def test_full_path_contract_blame_when_dependency_dropped(ds, bm25):
    """Drop a param-source from the real router's selection → blame=CONTRACT (Scenario B)."""
    q = ds.query_by_id("q240")
    route = assemble_route_result(bm25.rank(q.query_text, q.query_id), ds.tool_deps)
    # validate_email is a REQUIRED-arg source (it sits in the variant-A required-set), so dropping it is
    # a genuine required-dependency miss — the substance of the CONTRACT case, not label noise.
    assert "validate_email" in variant_a_required_set(q.main, ds.tool_deps)
    dropped = [t for t in route.selected_tools if t != "validate_email"]
    route_c = route.model_copy(update={"selected_tools": dropped})
    rep = check_invariants(route_c, ds.tool_deps)
    assert rep.closure_complete is False
    assert "audible_account_login.email" in rep.dangling_params  # validate_email sourced it
    res = mock_run(_plan(ds, route_c, rep), ds.tool_deps, dropped)
    att = attribute(route_c, res, rep, required_tools=dropped)
    assert res.completed is False
    assert att.outcome is Outcome.FAILURE and att.blame is Blame.CONTRACT
