"""GNN stage 2 — vectorized training loop (ADR 0022 / 0023 / 0024 / 0026).

Query-level split + train-only fitting, precomputed false-negative mask, one GNN forward per step,
matmul scoring, batch-embedded queries, masked InfoNCE that descends, determinism, checkpointing.
No GNNRouter (stage 3). The real BGE model is loaded once (module-scoped).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mcp_router_eval.data.graph_build import build_graph
from mcp_router_eval.data.loader import load
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.routers.gnn_train import (
    CHECKPOINT_DIR,
    GNNTrainConfig,
    GNNTrainer,
    build_masks,
    masked_infonce,
    split_queries,
)

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
def ctx(st_model, tmp_path_factory):
    ds = load()
    graph = build_graph(ds)
    embedder = LocalEmbedder(cache_dir=tmp_path_factory.mktemp("emb"), model=st_model)
    return {"ds": ds, "graph": graph, "embedder": embedder}


def _trainer(ctx, **cfg):
    return GNNTrainer(ctx["ds"], ctx["graph"], ctx["embedder"], GNNTrainConfig(**cfg))


# --------------------------------------------------------------------------- #
# Split (ADR 0024) — query-level, no overlap, train-only fitting
# --------------------------------------------------------------------------- #
def test_split_query_level_no_overlap(ctx):
    n = len(ctx["ds"].queries)
    sp = split_queries(n, seed=0)
    assert len(sp.train) + len(sp.val) + len(sp.test) == n         # partition, no query dropped
    tr, va, te = set(sp.train), set(sp.val), set(sp.test)
    assert not (tr & va) and not (va & te) and not (tr & te)       # no query in two splits
    assert split_queries(n, seed=0).train == sp.train             # deterministic


def test_stats_fit_train_only(ctx):
    t = _trainer(ctx, epochs=1)
    # trainer holds train + val tensors for tuning; the TEST split is never materialized or touched
    assert t._q_train.shape[0] == len(t.split.train)
    assert t._q_val.shape[0] == len(t.split.val)
    assert not hasattr(t, "_q_test") and not hasattr(t, "_gold_test")
    # no test query leaks into the training set
    assert not (set(t.split.test) & set(t.split.train))


# --------------------------------------------------------------------------- #
# False-negative mask (ADR 0023) — excludes gold's PARAMETER_* deps
# --------------------------------------------------------------------------- #
def test_mask_excludes_gold_param_deps(ctx):
    ds, graph = ctx["ds"], ctx["graph"]
    ti = dict(graph.id_to_index)
    q = ds.query_by_id("q240")
    gold, dep = build_masks([q], ti, ds.tool_deps)
    assert bool(gold[0, ti["download_audible_book"]])              # a gold tool is a positive
    # validate_email / audible_account_login are PARAMETER_* deps of gold → excluded from negatives
    assert bool(dep[0, ti["validate_email"]])
    assert bool(dep[0, ti["audible_account_login"]])
    # a TOOL_*-only / unrelated tool is not marked as a dependency false-negative
    assert not bool(dep[0, ti["get_wifi_status"]])


# --------------------------------------------------------------------------- #
# Vectorization asserts (profiling R1 / R3 / scoring)
# --------------------------------------------------------------------------- #
def test_one_forward_per_step_not_per_query(ctx):
    t = _trainer(ctx, epochs=1, batch_size=None)  # full-batch: 1 train step + 1 val eval
    t.train()
    # one node forward per step (train) + one for val — NOT one per query.
    assert t.node_forward_count == 2
    assert t.node_forward_count < len(t.split.train)   # far fewer than #queries


def test_scoring_is_matmul(ctx):
    t = _trainer(ctx, epochs=1)
    t.scorer.eval()  # disable dropout so the two forwards are comparable
    qb = t._q_train[:5]
    with torch.no_grad():
        scores = t._score_batch(qb)
    assert scores.shape == (5, len(ctx["ds"].tools))   # [B, N] cosine matrix
    # equals the explicit normalized query @ node.T matmul (no per-(query,node) loop)
    with torch.no_grad():
        node = t.scorer.node_embeddings(t._x, t._edge_index, t._edge_type)
        expected = t.scorer.query_embedding(qb) @ node.T
    assert torch.allclose(scores, expected, atol=1e-6)
    assert float(scores.min()) >= -1.001 and float(scores.max()) <= 1.001


def test_queries_batch_embedded_once(st_model, tmp_path):
    ds = load()
    graph = build_graph(ds)
    embedder = LocalEmbedder(cache_dir=tmp_path, model=st_model)  # fresh cache
    t = GNNTrainer(ds, graph, embedder, GNNTrainConfig(epochs=2))
    assert t._n_query_embed_calls == 2          # one batched call for train, one for val — not per step
    before = t._n_query_embed_calls
    t.train()
    assert t._n_query_embed_calls == before     # training loop never re-embeds
    # the train-split query texts are now cached (a re-encode computes nothing)
    embedder.encode([ds.queries[i].query_text for i in t.split.train[:5]])
    assert embedder.last_computed == 0


# --------------------------------------------------------------------------- #
# Masked InfoNCE (ADR 0026)
# --------------------------------------------------------------------------- #
def test_masked_infonce_excludes_false_negatives():
    # 1 query, 4 tools: tool0 = gold, tool1 = a gold dependency (false negative), tools 2/3 = negatives.
    scores = torch.tensor([[5.0, 4.0, 0.0, 0.0]])
    gold = torch.tensor([[True, False, False, False]])
    dep = torch.tensor([[False, True, False, False]])           # tool1 masked out of negatives
    no_dep = torch.zeros_like(dep)
    loss_masked = masked_infonce(scores, gold, dep, tau=1.0)
    loss_unmasked = masked_infonce(scores, gold, no_dep, tau=1.0)
    assert loss_masked >= 0.0
    # excluding the high-scoring false negative (tool1) lowers the loss vs counting it as a negative
    assert float(loss_masked) < float(loss_unmasked)


# --------------------------------------------------------------------------- #
# Loss descends — all three backbones train
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backbone", ["rgcn", "gat", "sage"])
def test_loss_descends(ctx, backbone):
    t = _trainer(ctx, backbone=backbone, epochs=10, seed=0)
    h = t.train()
    assert h["train"][-1] < h["train"][0]       # InfoNCE loss decreases


# --------------------------------------------------------------------------- #
# Determinism — same seed → identical trajectory
# --------------------------------------------------------------------------- #
def test_determinism_same_seed(ctx):
    h1 = _trainer(ctx, backbone="gat", epochs=5, seed=7).train()
    h2 = _trainer(ctx, backbone="gat", epochs=5, seed=7).train()
    assert h1["train"] == h2["train"] and h1["val"] == h2["val"]


# --------------------------------------------------------------------------- #
# Checkpoint round-trip + gitignored location
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip(ctx, tmp_path):
    t = _trainer(ctx, backbone="sage", epochs=3, seed=0)
    t.train()
    path = t.save_checkpoint(tmp_path / "sage.pt")
    qb = t._q_train[:4]
    t.scorer.eval()
    with torch.no_grad():
        before = t._score_batch(qb)

    t2 = _trainer(ctx, backbone="sage", epochs=3, seed=999)  # different init
    t2.load_checkpoint(path)
    t2.scorer.eval()
    with torch.no_grad():
        after = t2._score_batch(qb)
    assert torch.allclose(before, after, atol=1e-6)          # loaded model reproduces scores

    # default checkpoint dir is under the gitignored data/processed/
    assert "data/processed/gnn_checkpoints" in CHECKPOINT_DIR.as_posix()
