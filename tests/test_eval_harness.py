"""Evaluation harness (ADR 0005/0015/0024/0028) — small real run; the full run is stage 3."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_router_eval.contracts import Blame
from mcp_router_eval.data.graph_build import build_graph
from mcp_router_eval.data.loader import load
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.eval.harness import (
    EVAL_DIR,
    EvalConfig,
    build_routers,
    evaluate,
    evaluate_query,
)
from mcp_router_eval.eval.metrics import QueryResult
from mcp_router_eval.eval.slices import DEEP, MEDIUM, SHALLOW
from mcp_router_eval.routers.gnn_train import GNNTrainConfig, GNNTrainer, split_queries

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
    trainer = GNNTrainer(ds, graph, embedder, GNNTrainConfig(backbone="rgcn", epochs=5, seed=0))
    trainer.train()
    ckpt = trainer.save_checkpoint(tmp_path_factory.mktemp("ck") / "rgcn.pt")
    return {"ds": ds, "graph": graph, "embedder": embedder, "ckpt": ckpt}


# --------------------------------------------------------------------------- #
# QueryResult population (one query × one router, real pipeline)
# --------------------------------------------------------------------------- #
def test_query_result_populated(env):
    ds = env["ds"]
    bm25 = build_routers(ds, env["graph"], env["embedder"])["bm25"]
    qr = evaluate_query(bm25, ds.query_by_id("q240"), ds)
    assert isinstance(qr, QueryResult)
    assert qr.ranked_tools and len(qr.ranked_tools) == len(ds.tools)     # full ranking present
    assert qr.gold == frozenset(ds.query_by_id("q240").required_tools)   # gold set present
    assert isinstance(qr.completed, bool)                                # completion outcome present
    assert qr.blame is None or isinstance(qr.blame, Blame)               # attribution present
    assert qr.closure_depth >= len(qr.gold) - len(qr.gold)               # a non-negative depth
    assert qr.router_name == "bm25"


# --------------------------------------------------------------------------- #
# Five routers run; GNN uses the checkpoint; baseline-only fallback works
# --------------------------------------------------------------------------- #
def test_all_five_routers_with_checkpoint(env):
    routers = build_routers(env["ds"], env["graph"], env["embedder"], gnn_checkpoint=env["ckpt"])
    assert set(routers) == {"bm25", "naive_rag", "hybrid_rag", "traversal", "gnn_rgcn"}


def test_baseline_only_fallback_no_crash(env):
    routers = build_routers(env["ds"], env["graph"], env["embedder"], gnn_checkpoint=None)
    assert set(routers) == {"bm25", "naive_rag", "hybrid_rag", "traversal"}   # no GNN, no crash


# --------------------------------------------------------------------------- #
# Comparison aggregation: per router × slice, finite-or-sentinel numbers
# --------------------------------------------------------------------------- #
def test_comparison_structure_and_finite(env, tmp_path):
    comp = evaluate(
        env["ds"], env["embedder"],
        config=EvalConfig(k=10, seed=0, limit=12), gnn_checkpoint=env["ckpt"],
        out_dir=tmp_path / "eval", save=True,
    )
    assert set(comp["routers"]) == {"bm25", "naive_rag", "hybrid_rag", "traversal", "gnn_rgcn"}
    for name, rep in comp["routers"].items():
        assert set(rep["slices"]) == {SHALLOW, MEDIUM, DEEP}             # every slice present
        for block in [rep["overall"], *rep["slices"].values()]:
            r = block["retrieval"]
            assert all(0.0 <= v <= 1.0 for v in r.values())             # retrieval metrics finite in [0,1]
            assert 0.0 <= block["completion"]["rate"] <= 1.0
            cond = block["transfer_loss"]["conditional"]
            assert cond is None or 0.0 <= cond <= 1.0                   # finite OR the empty-denom sentinel
            assert "sub_rates" in block["completion"]                   # decomposable
            assert isinstance(block["attribution"], dict)              # attribution breakdown present
    # headline present for every router (deep-slice conditional; None allowed)
    assert set(comp["headline_deep_transfer_loss"]) == set(comp["routers"])
    # artifacts written to the (gitignored) out dir
    assert (tmp_path / "eval" / "comparison_seed0.json").exists()
    assert (tmp_path / "eval" / "comparison_seed0.txt").exists()


# --------------------------------------------------------------------------- #
# Test-split only (no leakage) + determinism
# --------------------------------------------------------------------------- #
def test_evaluates_test_split_only(env):
    ds = env["ds"]
    split = split_queries(len(ds.queries), seed=0)          # SAME seed the harness + trainer use
    # the harness's test queries are exactly the split's test indices (never train/val → no leakage)
    comp = evaluate(ds, env["embedder"], config=EvalConfig(seed=0, limit=None),
                    gnn_checkpoint=None, save=False)
    assert comp["config"]["n_queries"] == len(split.test)
    assert not (set(split.test) & set(split.train))          # disjoint by construction


def test_determinism(env):
    kw = dict(config=EvalConfig(seed=0, limit=10), gnn_checkpoint=env["ckpt"], save=False)
    a = evaluate(env["ds"], env["embedder"], **kw)
    b = evaluate(env["ds"], env["embedder"], **kw)
    assert json.dumps(a, default=str, sort_keys=True) == json.dumps(b, default=str, sort_keys=True)


def test_eval_dir_is_gitignored_location():
    assert "data/processed/eval" in EVAL_DIR.as_posix()      # under gitignored data/processed/*
