"""GNN stage 3 — GNNRouter integration (ADR 0018 / 0022 / 0027). Completes the GNN implementation.

A genuinely trained GNN (short real-data run → checkpoint → load) is wrapped as a Router and exercised
exactly like the baselines: pure ranking → shared closure → invariants → executor → attribution. The
real BGE model is loaded once (module-scoped).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mcp_router_eval.contract_layer.attribution import attribute
from mcp_router_eval.contract_layer.invariants import check_invariants
from mcp_router_eval.contracts import Blame, ExecPlan, GateDecision, Outcome, RouteResult
from mcp_router_eval.data.graph_build import build_graph
from mcp_router_eval.data.loader import load, topo_order
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.executor.mock_tools import run as mock_run
from mcp_router_eval.routers.base import HOMOPHILY_NA, Router
from mcp_router_eval.routers.gnn import GNNRouter
from mcp_router_eval.routers.gnn_train import GNNTrainConfig, GNNTrainer

pytestmark = pytest.mark.skipif(
    not (Path("data/processed") / "tools.jsonl").exists(), reason="processed data absent; run preprocess"
)


@pytest.fixture(scope="module")
def st_model():
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(LocalEmbedder.MODEL_ID)
    m.eval()
    return m


@pytest.fixture(scope="module")
def env(st_model, tmp_path_factory):
    ds = load()
    graph = build_graph(ds)
    embedder = LocalEmbedder(cache_dir=tmp_path_factory.mktemp("emb"), model=st_model)
    ckpt_dir = tmp_path_factory.mktemp("ckpts")
    # train a SHORT real checkpoint per backbone so the router wraps a genuinely trained GNN
    checkpoints = {}
    for bb in ("rgcn", "gat", "sage"):
        trainer = GNNTrainer(ds, graph, embedder, GNNTrainConfig(backbone=bb, epochs=5, seed=0))
        trainer.train()
        checkpoints[bb] = trainer.save_checkpoint(ckpt_dir / f"{bb}.pt")
    return {"ds": ds, "graph": graph, "embedder": embedder, "checkpoints": checkpoints}


def _router(env, backbone="rgcn"):
    return GNNRouter.from_checkpoint(
        env["checkpoints"][backbone], env["ds"], env["graph"], env["embedder"]
    )


# --------------------------------------------------------------------------- #
# Checkpoint load + contract
# --------------------------------------------------------------------------- #
def test_loads_checkpoint_and_is_router(env):
    r = _router(env)
    assert isinstance(r, Router) and r.name == "gnn_rgcn"


def test_rank_produces_valid_route_result(env):
    r = _router(env)
    ds = env["ds"]
    q = ds.query_by_id("q240")
    rr = r.rank(q.query_text, q.query_id)
    assert len(rr.ranked_tools) == len(ds.tools)
    assert 0.0 <= rr.confidence <= 1.0                     # ADR 0018
    route = r.route(q.query_text, q.query_id)
    assert isinstance(route, RouteResult) and route.router_name == "gnn_rgcn"


# --------------------------------------------------------------------------- #
# Late cosine — matmul, no query-node fusion MLP (ADR 0022 amendment)
# --------------------------------------------------------------------------- #
def test_scoring_is_late_cosine_matmul(env):
    r = _router(env)
    q = env["ds"].query_by_id("q240")
    with torch.no_grad():
        node = r._node_embeddings()
        qv = r._scorer.query_embedding(
            torch.as_tensor(r._embedder.encode([q.query_text])[0], dtype=torch.float)
        )
        manual = (node @ qv).numpy()                       # explicit cosine matmul
    scored = {ts.tool_id: ts.score for ts in r.rank(q.query_text, q.query_id).ranked_tools}
    assert np.allclose([scored[t] for t in r._tool_ids], manual, atol=1e-5)
    # two-tower: projections are per-tower; no Linear ingests a fused (query+node) vector
    fused = r._embedder.dim + r._scorer.encoder.out_dim
    assert all(m.in_features != fused for m in r._scorer.modules() if isinstance(m, nn.Linear))


# --------------------------------------------------------------------------- #
# homophily_local (ADR 0027, GNN-only) — real for dep-tools, sentinel for dep-free
# --------------------------------------------------------------------------- #
def test_homophily_real_for_dependency_tool(env):
    r = _router(env)
    # audible_account_login has PARAMETER_* deps (validate_email, get_system_language) → a real value
    h = r.tool_homophily("audible_account_login")
    assert h != HOMOPHILY_NA and -1.0001 <= h <= 1.0001
    # validate_email is a core tool with NO PARAMETER_* deps → the sentinel (not a fabricated non-zero)
    assert r.tool_homophily("validate_email") == HOMOPHILY_NA
    assert r.tool_homophily("get_wifi_status") == HOMOPHILY_NA
    # the RouteResult scalar over a dep-containing set is a real (non-sentinel) mean
    agg = r.homophily_local(["audible_account_login", "download_audible_book"])
    assert agg != HOMOPHILY_NA and -1.0001 <= agg <= 1.0001
    # over a dependency-free set → sentinel
    assert r.homophily_local(["get_wifi_status", "get_battery_status"]) == HOMOPHILY_NA


# --------------------------------------------------------------------------- #
# Ranking sanity (not an eval metric) + all three backbones + determinism
# --------------------------------------------------------------------------- #
def test_ranking_sanity_deep_dependency_query(env):
    r = _router(env)
    q = env["ds"].query_by_id("q240")
    ranked = r.rank(q.query_text, q.query_id).ranked_tools
    best_gold = min(next(t.rank for t in ranked if t.tool_id == g) for g in q.required_tools)
    assert best_gold < len(env["ds"].tools)               # a gold tool is ranked (sanity, not a metric)


@pytest.mark.parametrize("backbone", ["rgcn", "gat", "sage"])
def test_all_backbones_load_and_rank(env, backbone):
    r = _router(env, backbone)
    q = env["ds"].query_by_id("q240")
    route = r.route(q.query_text, q.query_id)
    assert route.router_name == f"gnn_{backbone}" and route.selected_tools


def test_determinism_same_checkpoint(env):
    q = env["ds"].query_by_id("q240")
    a = _router(env).rank(q.query_text, q.query_id).ranked_tools
    b = _router(env).rank(q.query_text, q.query_id).ranked_tools
    assert [(t.tool_id, t.score) for t in a] == [(t.tool_id, t.score) for t in b]


# --------------------------------------------------------------------------- #
# Full-pipeline integration — GNNRouter → closure → invariants → executor → attribution
# --------------------------------------------------------------------------- #
def test_full_pipeline_integration(env):
    ds = env["ds"]
    r = _router(env)
    q = ds.query_by_id("q240")
    route = r.route(q.query_text, q.query_id)              # pure ranking + shared closure (ADR 0021)
    rep = check_invariants(route, ds.tool_deps)
    assert rep.closure_complete is True                   # shared closure guarantees completeness
    order = topo_order(route.selected_tools, ds.tool_deps)
    plan = ExecPlan(
        query_id=q.query_id, query_text=q.query_text,
        bound_tools=[ds.tools[t] for t in order], invariant_report=rep,
        gate_decision=GateDecision.PASS, trace_id="t-gnn",
    )
    res = mock_run(plan, ds.tool_deps, route.selected_tools)
    att = attribute(route, res, rep, required_tools=route.selected_tools)
    assert res.completed is True
    assert att.outcome is Outcome.SUCCESS and att.blame is Blame.NONE
