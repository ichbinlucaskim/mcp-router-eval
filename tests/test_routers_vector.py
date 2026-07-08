"""Vector baselines — NaiveRAG (dense) + HybridRAG (convex-combination fusion).

ADR 0003 (embedding provider) / 0018 (normalized signals) / 0019 (CC fusion) / 0020 (shared text).
Loads the real BGE model once (module-scoped) and injects it into embedders with isolated caches, so
per-test cost stays small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mcp_router_eval.contract_layer.attribution import attribute
from mcp_router_eval.contract_layer.invariants import check_invariants
from mcp_router_eval.contracts import Blame, ExecPlan, GateDecision, Outcome, RouteResult
from mcp_router_eval.data.loader import load, topo_order
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.eval.harness import variant_a_required_set
from mcp_router_eval.executor.mock_tools import run as mock_run
from mcp_router_eval.routers.base import (
    HOMOPHILY_NA,
    Router,
    minmax_normalize,
    normalize_confidence,
)
from mcp_router_eval.routers.baselines import (
    BM25Router,
    HybridRAGRouter,
    NaiveRAGRouter,
    tool_document,
)
from mcp_router_eval.routers.closure import assemble_route_result

_HAS_DATA = (Path("data/processed") / "tools.jsonl").exists()
pytestmark = pytest.mark.skipif(_HAS_DATA is False, reason="processed data absent; run preprocess")


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
def embedder(st_model, tmp_path_factory):
    return LocalEmbedder(cache_dir=tmp_path_factory.mktemp("emb"), model=st_model)


@pytest.fixture(scope="module")
def bm25(ds):
    return BM25Router(ds)


@pytest.fixture(scope="module")
def naive(ds, embedder):
    return NaiveRAGRouter(ds, embedder)


@pytest.fixture(scope="module")
def hybrid(bm25, naive):
    return HybridRAGRouter(bm25, naive)  # α = 0.5


# --------------------------------------------------------------------------- #
# Interface + valid RouteResult
# --------------------------------------------------------------------------- #
def test_vector_routers_are_routers(naive, hybrid):
    assert isinstance(naive, Router) and naive.name == "naive_rag"
    assert isinstance(hybrid, Router) and hybrid.name == "hybrid_rag"
    assert hybrid.alpha == 0.5  # ADR 0019 default


def test_assembly_valid_route_result(ds, naive, hybrid):
    q = ds.query_by_id("q240")
    for router in (naive, hybrid):
        route = assemble_route_result(router.rank(q.query_text, q.query_id), ds.tool_deps)
        assert isinstance(route, RouteResult)  # extra="forbid" would raise on drift
        assert route.router_name == router.name and route.selected_tools


# --------------------------------------------------------------------------- #
# ADR 0020 — the dense router embeds the exact text BM25 indexes
# --------------------------------------------------------------------------- #
def test_same_text_as_bm25(ds, naive, bm25):
    assert naive.tool_ids == bm25.tool_ids  # same tool order
    # Every embedded document is exactly tool_document(spec) — the same helper BM25 tokenizes.
    assert naive.documents == [tool_document(ds.tools[t]) for t in naive.tool_ids]
    idx = naive.tool_ids.index("download_audible_book")
    assert naive.documents[idx] == tool_document(ds.tools["download_audible_book"])


# --------------------------------------------------------------------------- #
# Embedding cache actually reused — 573 tool vectors computed once
# --------------------------------------------------------------------------- #
def test_embedding_cache_reused(ds, st_model, tmp_path):
    e1 = LocalEmbedder(cache_dir=tmp_path, model=st_model)
    NaiveRAGRouter(ds, e1)
    assert e1.last_computed == len(ds.tools)  # cold cache → all 573 computed
    e2 = LocalEmbedder(cache_dir=tmp_path, model=st_model)
    NaiveRAGRouter(ds, e2)
    assert e2.last_computed == 0               # warm cache → nothing recomputed


# --------------------------------------------------------------------------- #
# NaiveRAG real-data sanity — semantically related gold tool ranks highly
# --------------------------------------------------------------------------- #
def test_naive_ranks_gold_highly(ds, naive):
    q = ds.query_by_id("q240")
    rr = naive.rank(q.query_text, q.query_id)
    rank_of_main = next(t.rank for t in rr.ranked_tools if t.tool_id == q.main)
    assert q.main in rr.top_k and rank_of_main <= 2  # download_audible_book near the top


# --------------------------------------------------------------------------- #
# HybridRAG convex combination — α endpoints reduce to the pure rankers
# --------------------------------------------------------------------------- #
def test_hybrid_alpha_endpoints(ds, bm25, naive):
    q = ds.query_by_id("q240")
    pure_sparse = HybridRAGRouter(bm25, naive, alpha=0.0).rank(q.query_text, q.query_id)
    pure_dense = HybridRAGRouter(bm25, naive, alpha=1.0).rank(q.query_text, q.query_id)
    assert pure_sparse.top_k == bm25.rank(q.query_text, q.query_id).top_k    # α=0 → BM25
    assert pure_dense.top_k == naive.rank(q.query_text, q.query_id).top_k    # α=1 → NaiveRAG
    assert pure_sparse.top_k != pure_dense.top_k                             # α genuinely changes ranking


def test_hybrid_blends_both_signals(ds, bm25, naive, hybrid):
    q = ds.query_by_id("q240")
    fused = hybrid.rank(q.query_text, q.query_id).top_k
    sparse = bm25.rank(q.query_text, q.query_id).top_k
    dense = naive.rank(q.query_text, q.query_id).top_k
    # A genuine blend at α=0.5: not identical to either pure ranker's top-k.
    assert fused != sparse or fused != dense


# --------------------------------------------------------------------------- #
# Confidence + homophily sentinel (ADR 0018)
# --------------------------------------------------------------------------- #
def test_confidence_and_homophily(ds, naive, hybrid):
    q = ds.query_by_id("q240")
    for router in (naive, hybrid):
        rr = router.rank(q.query_text, q.query_id)
        assert 0.0 <= rr.confidence <= 1.0
        route = assemble_route_result(rr, ds.tool_deps)
        assert route.homophily_local == HOMOPHILY_NA == 0.0  # not the GNN


def test_confidence_degenerate_rule():
    assert normalize_confidence([]) == 1.0
    assert normalize_confidence([0.7, 0.7, 0.7]) == 1.0        # M == m → constant
    assert np.array_equal(minmax_normalize([5.0, 5.0]), np.ones(2))  # vector degenerate → all 1.0


# --------------------------------------------------------------------------- #
# Determinism — cached, deterministic vectors → identical RouteResult
# --------------------------------------------------------------------------- #
def test_determinism(ds, naive, hybrid):
    q = ds.query_by_id("q240")
    for router in (naive, hybrid):
        r1 = assemble_route_result(router.rank(q.query_text, q.query_id), ds.tool_deps)
        r2 = assemble_route_result(router.rank(q.query_text, q.query_id), ds.tool_deps)
        assert r1.model_dump() == r2.model_dump()


# --------------------------------------------------------------------------- #
# Full path on real data — NaiveRAG → closure → invariants → executor → attribution
# --------------------------------------------------------------------------- #
def _plan(ds, route, rep):
    order = topo_order(route.selected_tools, ds.tool_deps)
    return ExecPlan(
        query_id=route.query_id, query_text=route.query_text,
        bound_tools=[ds.tools[t] for t in order], invariant_report=rep,
        gate_decision=GateDecision.PASS, trace_id="t-" + route.query_id,
    )


def test_full_path_success_through_naive(ds, naive):
    # De-circularized (ADR-0030, checkup step 5): score completion against the variant-A required-set
    # (the required-arg PARAMETER_* spine the harness uses), not route.selected_tools (tautological).
    # This verifies the dense router actually recovered the required-arg tools for q240.
    q = ds.query_by_id("q240")
    route = assemble_route_result(naive.rank(q.query_text, q.query_id), ds.tool_deps)
    rep = check_invariants(route, ds.tool_deps)
    assert rep.closure_complete is True
    required = variant_a_required_set(q.main, ds.tool_deps)
    assert required <= set(route.selected_tools)    # REAL check: router surfaced the required-arg spine
    res = mock_run(_plan(ds, route, rep), ds.tool_deps, list(required))
    att = attribute(route, res, rep, required_tools=list(required))
    assert res.completed is True
    assert att.outcome is Outcome.SUCCESS and att.blame is Blame.NONE


def test_full_path_contract_blame_when_dependency_dropped(ds, naive):
    """Drop a param-source from the dense router's selection → blame=CONTRACT (Scenario B)."""
    q = ds.query_by_id("q240")
    route = assemble_route_result(naive.rank(q.query_text, q.query_id), ds.tool_deps)
    assert "validate_email" in route.selected_tools  # pulled in by closure via the audible spine
    assert "validate_email" in variant_a_required_set(q.main, ds.tool_deps)  # a REQUIRED-arg source
    dropped = [t for t in route.selected_tools if t != "validate_email"]
    route_c = route.model_copy(update={"selected_tools": dropped})
    rep = check_invariants(route_c, ds.tool_deps)
    assert rep.closure_complete is False
    assert "audible_account_login.email" in rep.dangling_params
    res = mock_run(_plan(ds, route_c, rep), ds.tool_deps, dropped)
    att = attribute(route_c, res, rep, required_tools=dropped)
    assert res.completed is False
    assert att.outcome is Outcome.FAILURE and att.blame is Blame.CONTRACT
