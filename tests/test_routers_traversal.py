"""Traversal router — Graph RAG-Tool Fusion, standard/no-rerank (ADR 0021 + 2026-07-05 amendment).

Block-interleaving order, DFS depth limit, TOOL_* exclusion, and the §7.1 thesis premise (traversal
recovers low-homophily dependencies dense retrieval misses). Synthetic fixtures test the algorithm in
isolation (via a fake initial ranker); real-data tests use the actual hybrid + BGE.
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
    ToolScore,
)
from mcp_router_eval.data.loader import load, topo_order
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.executor.mock_tools import run as mock_run
from mcp_router_eval.routers.base import HOMOPHILY_NA, RankResult, Router
from mcp_router_eval.routers.baselines import (
    BM25Router,
    HybridRAGRouter,
    NaiveRAGRouter,
    TraversalRouter,
    _dfs_dependencies,
)
from mcp_router_eval.routers.closure import assemble_route_result


# --------------------------------------------------------------------------- #
# Synthetic fixtures — exact block-interleaving / DFS behavior in isolation
# --------------------------------------------------------------------------- #
class _FakeInitial:
    """Stand-in for HybridRAGRouter: returns a fixed initial vector ranking."""

    def __init__(self, order: list[str]):
        self._order = order

    def rank(self, query_text: str, query_id: str) -> RankResult:
        ranked = [ToolScore(tool_id=t, score=float(len(self._order) - i), rank=i)
                  for i, t in enumerate(self._order)]
        return RankResult(query_id=query_id, query_text=query_text, ranked_tools=ranked,
                          top_k=self._order, confidence=0.5, router_name="fake")


def _pd(source: str, param: str = "p", relation: EdgeType = EdgeType.PARAM_DIRECT) -> Dep:
    return Dep(source=source, param=param, relation=relation)


# v1 → a → b (2-hop param chain); v2 → c (direct); plus a TOOL_* edge that must NOT be traversed.
SYNTH_DEPS = {
    "v1": [_pd("a"), Dep(source="t_noise", param=None, relation=EdgeType.TOOL_DIRECT)],
    "a": [_pd("b")],
    "b": [],
    "v2": [_pd("c")],
    "c": [],
}


def test_block_interleaving_order():
    """[v1, deps(v1), v2, deps(v2)] — each tool immediately followed by its deps; vector order kept."""
    trav = TraversalRouter(_FakeInitial(["v1", "v2"]), SYNTH_DEPS, k=2, d_limit=3, final_top_k=10)
    order = [t.tool_id for t in trav.rank("q", "q1").ranked_tools]
    assert order == ["v1", "a", "b", "v2", "c"]  # v1 then its 2-hop chain, then v2 then its dep


def test_dfs_depth_limit():
    assert _dfs_dependencies("v1", SYNTH_DEPS, 1) == ["a"]          # direct only
    assert _dfs_dependencies("v1", SYNTH_DEPS, 2) == ["a", "b"]     # 2-hop reached
    # d_limit truncates: with limit 1 the 2-hop 'b' is excluded from the interleaving
    trav1 = TraversalRouter(_FakeInitial(["v1"]), SYNTH_DEPS, k=1, d_limit=1, final_top_k=10)
    assert [t.tool_id for t in trav1.rank("q", "q1").ranked_tools] == ["v1", "a"]


def test_tool_star_excluded_from_traversal():
    """PARAMETER_*-centric (ADR 0013): a TOOL_* neighbor is never traversed."""
    assert "t_noise" not in _dfs_dependencies("v1", SYNTH_DEPS, 3)


def test_dedup_and_truncation():
    """First occurrence wins; final list truncated to final_top_k."""
    # v2 also depends on 'a' (already surfaced under v1) → not repeated.
    deps = {**SYNTH_DEPS, "v2": [_pd("a"), _pd("c")]}
    trav = TraversalRouter(_FakeInitial(["v1", "v2"]), deps, k=2, d_limit=3, final_top_k=4)
    order = [t.tool_id for t in trav.rank("q", "q1").ranked_tools]
    assert order == ["v1", "a", "b", "v2"]  # 'a' not repeated under v2; truncated to 4


def test_is_router_and_no_rerank_determinism():
    trav = TraversalRouter(_FakeInitial(["v1", "v2"]), SYNTH_DEPS)
    assert isinstance(trav, Router) and trav.name == "traversal"
    r1 = trav.rank("q", "q1")
    r2 = trav.rank("q", "q1")
    assert [t.tool_id for t in r1.ranked_tools] == [t.tool_id for t in r2.ranked_tools]


# --------------------------------------------------------------------------- #
# Real-data tests
# --------------------------------------------------------------------------- #
_HAS_DATA = (Path("data/processed") / "tools.jsonl").exists()
realdata = pytest.mark.skipif(_HAS_DATA is False, reason="processed data absent; run preprocess")


@pytest.fixture(scope="module")
def st_model():
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(LocalEmbedder.MODEL_ID)
    m.eval()
    return m


@pytest.fixture(scope="module")
def ds():
    return load()


@pytest.fixture(scope="module")
def routers(ds, st_model, tmp_path_factory):
    emb = LocalEmbedder(cache_dir=tmp_path_factory.mktemp("emb"), model=st_model)
    hybrid = HybridRAGRouter(BM25Router(ds), NaiveRAGRouter(ds, emb))
    naive = NaiveRAGRouter(ds, emb)
    traversal = TraversalRouter(hybrid, ds.tool_deps)
    return {"naive": naive, "traversal": traversal}


@realdata
def test_valid_route_result_and_params(ds, routers):
    q = ds.query_by_id("q240")
    trav = routers["traversal"]
    assert trav.params == {"k": 3, "d_limit": 3, "final_top_k": 10}  # recorded per run (amendment)
    route = assemble_route_result(trav.rank(q.query_text, q.query_id), ds.tool_deps)
    assert isinstance(route, RouteResult) and route.router_name == "traversal"


@realdata
def test_traversal_recovers_low_homophily_dependency(ds, routers):
    """§7.1 premise: traversal surfaces validate_email where dense retrieval buries it."""
    q = ds.query_by_id("q240")
    trav_order = [t.tool_id for t in routers["traversal"].rank(q.query_text, q.query_id).ranked_tools]
    naive_ranked = routers["naive"].rank(q.query_text, q.query_id).ranked_tools
    naive_rank_of_ve = next(t.rank for t in naive_ranked if t.tool_id == "validate_email")
    assert "validate_email" in trav_order          # traversal catches the low-homophily dep
    assert naive_rank_of_ve > 50                    # dense retrieval ranks it far down (≈384)


@realdata
def test_still_passes_shared_closure(ds, routers):
    q = ds.query_by_id("q240")
    route = assemble_route_result(routers["traversal"].rank(q.query_text, q.query_id), ds.tool_deps)
    rep = check_invariants(route, ds.tool_deps)
    assert rep.closure_complete is True             # shared closure guarantees completeness


@realdata
def test_confidence_homophily_determinism(ds, routers):
    q = ds.query_by_id("q240")
    trav = routers["traversal"]
    rr = trav.rank(q.query_text, q.query_id)
    assert 0.0 <= rr.confidence <= 1.0
    r1 = assemble_route_result(rr, ds.tool_deps)
    r2 = assemble_route_result(trav.rank(q.query_text, q.query_id), ds.tool_deps)
    assert r1.homophily_local == HOMOPHILY_NA == 0.0
    assert r1.model_dump() == r2.model_dump()


def _plan(ds, route, rep):
    order = topo_order(route.selected_tools, ds.tool_deps)
    return ExecPlan(
        query_id=route.query_id, query_text=route.query_text,
        bound_tools=[ds.tools[t] for t in order], invariant_report=rep,
        gate_decision=GateDecision.PASS, trace_id="t-" + route.query_id,
    )


@realdata
def test_full_path_success_and_contract(ds, routers):
    q = ds.query_by_id("q240")
    route = assemble_route_result(routers["traversal"].rank(q.query_text, q.query_id), ds.tool_deps)
    rep = check_invariants(route, ds.tool_deps)
    res = mock_run(_plan(ds, route, rep), ds.tool_deps, route.selected_tools)
    att = attribute(route, res, rep, required_tools=route.selected_tools)
    assert res.completed is True and att.outcome is Outcome.SUCCESS and att.blame is Blame.NONE

    # Dependency-drop → CONTRACT through the real traversal router (Scenario B).
    assert "validate_email" in route.selected_tools
    dropped = [t for t in route.selected_tools if t != "validate_email"]
    route_c = route.model_copy(update={"selected_tools": dropped})
    rep_c = check_invariants(route_c, ds.tool_deps)
    res_c = mock_run(_plan(ds, route_c, rep_c), ds.tool_deps, dropped)
    att_c = attribute(route_c, res_c, rep_c, required_tools=dropped)
    assert res_c.completed is False and att_c.blame is Blame.CONTRACT
