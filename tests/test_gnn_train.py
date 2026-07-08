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
    train_log_q,
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
# logQ popularity correction (ADR 0031 amendment) — TRAIN-only −α·log Q, training-time only
# --------------------------------------------------------------------------- #
def test_train_log_q_add1_smoothed_and_monotonic():
    # 2 train queries, 3 tools: counts = [2, 1, 0]; add-1 → q = (3,2,1)/(3+3) = (0.5, 0.333, 0.167).
    gold = torch.tensor([[True, True, False], [True, False, False]])
    lq = train_log_q(gold)
    expected = torch.log(torch.tensor([3.0, 2.0, 1.0]) / 6.0)
    assert torch.allclose(lq, expected, atol=1e-6)
    assert lq[0] > lq[1] > lq[2]                    # more frequent ⇒ higher (less negative) log Q
    assert torch.isfinite(lq[2])                    # never-gold tool is finite (add-1, no log 0)


def test_log_q_is_train_only(ctx):
    t = _trainer(ctx, epochs=1, seed=0)
    # log Q is derived ONLY from the train-split gold mask (ADR 0024 — no val/test fitting)
    assert torch.equal(t._log_q, train_log_q(t._gold_train))
    # spot check that validation gold does NOT enter: recomputing WITH val rows changes the vector
    assert not torch.equal(t._log_q, train_log_q(torch.cat([t._gold_train, t._gold_val])))


def test_alpha_zero_recovers_baseline_logits():
    # alpha=0 (or log_q=None) leaves the InfoNCE loss EXACTLY as the pre-correction baseline.
    # Non-dominating scores (tau=1.0) so the loss is meaningfully nonzero and alpha visibly moves it.
    scores = torch.tensor([[1.0, 0.9, 0.5, 0.4]])
    gold = torch.tensor([[True, False, False, False]])
    dep = torch.tensor([[False, True, False, False]])          # tool1 masked out of the negatives
    log_q = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    base = masked_infonce(scores, gold, dep, tau=1.0)
    assert torch.equal(base, masked_infonce(scores, gold, dep, tau=1.0, log_q=log_q, alpha=0.0))
    assert torch.equal(base, masked_infonce(scores, gold, dep, tau=1.0, log_q=None, alpha=1.0))
    # a nonzero alpha DOES change the loss (the correction is actually applied)
    assert not torch.equal(base, masked_infonce(scores, gold, dep, tau=1.0, log_q=log_q, alpha=1.0))


def test_trainer_alpha_zero_matches_default(ctx):
    # alpha defaults to 0.0; an explicit alpha=0 trains an identical trajectory (behavior-preserving).
    h_default = _trainer(ctx, backbone="rgcn", epochs=3, seed=0).train()
    h_alpha0 = _trainer(ctx, backbone="rgcn", epochs=3, seed=0, alpha=0.0).train()
    assert h_default["train"] == h_alpha0["train"] and h_default["val"] == h_alpha0["val"]


def test_alpha_downweights_high_freq_relative_to_rare(ctx):
    # ADR-0031: α>0 lowers a high-frequency tool's TRAINING logit RELATIVE to a rare tool (the absolute
    # level cancels in InfoNCE; only the relative frequency signal matters). Checked on the score matrix.
    t = _trainer(ctx, epochs=1, seed=0)
    ti = dict(ctx["graph"].id_to_index)
    hi, lo = ti["get_wifi_status"], ti["download_audible_book"]   # frequent vs rare (in train gold)
    assert t._log_q[hi] > t._log_q[lo]
    t.scorer.eval()
    with torch.no_grad():
        row = t._score_batch(t._q_train[:1])[0] / t.config.tau    # training logits for one query
    gap_alpha0 = (row[hi] - row[lo]).item()
    corrected = row - 1.0 * t._log_q                              # alpha=1
    gap_alpha1 = (corrected[hi] - corrected[lo]).item()
    assert gap_alpha1 < gap_alpha0                                # high-freq tool drops relative to rare


def test_logq_correction_keeps_one_forward_per_step(ctx):
    # The correction is one broadcasted subtraction, not a loop: it adds NO extra GNN forward.
    t = _trainer(ctx, epochs=1, batch_size=None, alpha=1.0)
    t.train()
    assert t.node_forward_count == 2                             # one train + one val forward, as before


def test_determinism_with_alpha(ctx):
    h1 = _trainer(ctx, backbone="rgcn", epochs=4, seed=3, alpha=1.0).train()
    h2 = _trainer(ctx, backbone="rgcn", epochs=4, seed=3, alpha=1.0).train()
    assert h1["train"] == h2["train"] and h1["val"] == h2["val"]  # same seed+alpha → identical trajectory


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
