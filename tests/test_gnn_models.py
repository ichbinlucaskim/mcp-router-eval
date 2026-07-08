"""GNN stage 1 — model definitions + graph feature filling (forward pass only).

ADR 0010 (backbones) / 0020 (shared text) / 0022 amendment (late cosine, no fusion) / 0025 (2 layers,
searchable hidden/heads/dropout). No training loop, no GNNRouter — those are stages 2 and 3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mcp_router_eval.data.graph_build import build_graph
from mcp_router_eval.data.loader import load
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.routers.baselines import tool_document
from mcp_router_eval.routers.gnn_models import (
    GATEncoder,
    GNNEncoder,
    GNNScorer,
    RGCNEncoder,
    SAGEEncoder,
    node_feature_matrix,
)

pytestmark = pytest.mark.skipif(
    not (Path("data/processed") / "tools.jsonl").exists(), reason="processed data absent; run preprocess"
)

_EMB_DIM = 384
_N = 573


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
    x = node_feature_matrix(graph, ds, embedder)
    return {
        "ds": ds, "graph": graph, "embedder": embedder, "x": x,
        "edge_index": graph.data.edge_index, "edge_type": graph.data.edge_type,
    }


def _encoders(in_dim):
    return {"rgcn": RGCNEncoder(in_dim), "gat": GATEncoder(in_dim), "sage": SAGEEncoder(in_dim)}


# --------------------------------------------------------------------------- #
# Node features — placeholder x[:, 1:] resolved (ADR 0003/0020)
# --------------------------------------------------------------------------- #
def test_node_features_filled(ctx):
    x, graph, ds, embedder = ctx["x"], ctx["graph"], ctx["ds"], ctx["embedder"]
    assert x.shape == (_N, 1 + _EMB_DIM)                       # [573, 385]
    # col 0 is still is_core (0/1), matching the graph's structural column
    assert torch.equal(x[:, 0], graph.data.x[:, 0])
    assert set(x[:, 0].tolist()) <= {0.0, 1.0}
    # cols 1: are exactly the BGE embeddings of tool_document(), in node order
    docs = [tool_document(ds.tools[t]) for t in graph.node_ids]
    expected = torch.as_tensor(embedder.encode(docs), dtype=torch.float)
    assert torch.allclose(x[:, 1:], expected)


# --------------------------------------------------------------------------- #
# Backbones — 2 layers, forward on the real graph → refined embeddings [573, H]
# --------------------------------------------------------------------------- #
def test_all_backbones_forward_shape(ctx):
    x, ei, et = ctx["x"], ctx["edge_index"], ctx["edge_type"]
    for name, enc in _encoders(x.shape[1]).items():
        assert enc.num_layers == 2                             # fixed (ADR 0025)
        enc.eval()
        with torch.no_grad():
            h = enc(x, ei, et)
        assert h.shape == (_N, 64), name                       # refined embeddings [573, hidden]


def test_rgcn_uses_four_relations(ctx):
    x = ctx["x"]
    enc = RGCNEncoder(x.shape[1])
    assert enc.uses_edge_type is True
    assert enc.num_relations == 4 and enc.conv1.num_relations == 4   # ADR 0006/0013
    assert sorted(set(ctx["edge_type"].tolist())) == [0, 1, 2, 3]


def test_gat_default_heads_two(ctx):
    enc = GATEncoder(ctx["x"].shape[1])
    assert enc.heads == 2 and enc.uses_edge_type is False           # ADR 0025


def test_uniform_interface_across_backbones(ctx):
    """Same call signature and same output shape for all three (edge_type optional for GAT/SAGE)."""
    x, ei, et = ctx["x"], ctx["edge_index"], ctx["edge_type"]
    shapes = set()
    for enc in _encoders(x.shape[1]).values():
        enc.eval()
        with torch.no_grad():
            shapes.add(tuple(enc(x, ei, et).shape))             # same signature works for all
    assert shapes == {(_N, 64)}


# --------------------------------------------------------------------------- #
# Initial residual (ADR-0025 amendment) — GCNII-style; α_res=0 is the exact baseline
# --------------------------------------------------------------------------- #
def _syn(in_dim=1 + _EMB_DIM, n=24, e=80):
    torch.manual_seed(0)
    return torch.randn(n, in_dim), torch.randint(0, n, (2, e)), torch.randint(0, 4, (e,))


_BACKBONES_ET = [(RGCNEncoder, True), (GATEncoder, False), (SAGEEncoder, False)]


def test_alpha_res_zero_is_exact_baseline():
    # α_res=0 builds NO projection module → the forward is byte-identical to the pre-amendment encoder.
    x, ei, et = _syn()
    for cls, needs_et in _BACKBONES_ET:
        torch.manual_seed(0); base = cls(x.shape[1], alpha_res=0.0).eval()
        torch.manual_seed(0); resid = cls(x.shape[1], alpha_res=0.5).eval()
        assert base.h0_proj is None                                  # no residual params at α_res=0
        assert not any("h0_proj" in n for n, _ in base.named_parameters())
        # conv weights match (h0_proj built AFTER the convs) → the residual is the ONLY difference
        base_params = {n: p for n, p in base.named_parameters()}
        for n, p in resid.named_parameters():
            if "h0_proj" not in n:
                assert torch.equal(base_params[n], p), n
        with torch.no_grad():
            out_base = base(x, ei, et if needs_et else None)
            out_resid = resid(x, ei, et if needs_et else None)
        assert not torch.allclose(out_base, out_resid)               # α_res=0.5 actually changes the output


def test_alpha_res_one_replaces_mp_with_shared_projection():
    # α_res=1 → the output layer's MP is fully replaced by the shared h0 projection (feature preservation).
    x, ei, et = _syn()
    for cls, needs_et in _BACKBONES_ET:
        torch.manual_seed(0); enc = cls(x.shape[1], alpha_res=1.0).eval()
        assert enc.h0_proj is not None
        assert enc.h0_proj.in_features == x.shape[1] and enc.h0_proj.out_features == enc.hidden  # one shared proj → hidden
        with torch.no_grad():
            out = enc(x, ei, et if needs_et else None)
        assert torch.allclose(out, enc.h0_proj(x), atol=1e-6)        # output == projected raw features


def test_initial_residual_shape_and_determinism(ctx):
    x, ei, et = ctx["x"], ctx["edge_index"], ctx["edge_type"]
    for cls, needs_et in _BACKBONES_ET:
        torch.manual_seed(1); a = cls(x.shape[1], alpha_res=0.5).eval()
        torch.manual_seed(1); b = cls(x.shape[1], alpha_res=0.5).eval()
        with torch.no_grad():
            ha = a(x, ei, et if needs_et else None)
            hb = b(x, ei, et if needs_et else None)
        assert ha.shape == (_N, 64)                                  # output shape unchanged by the residual
        assert torch.equal(ha, hb)                                   # seed + α_res → identical (deterministic)


def test_scorer_over_residual_encoder_keeps_unit_cosine(ctx):
    # Downstream late-cosine + per-tower L2 (ADR 0022) unchanged when the encoder has an initial residual.
    x, ei, et, embedder = ctx["x"], ctx["edge_index"], ctx["edge_type"], ctx["embedder"]
    torch.manual_seed(0)
    scorer = GNNScorer(RGCNEncoder(x.shape[1], alpha_res=0.5), query_dim=_EMB_DIM).eval()
    q = torch.as_tensor(embedder.encode(["download an audiobook"])[0], dtype=torch.float)
    with torch.no_grad():
        node = scorer.node_embeddings(x, ei, et)
        scores = scorer.score(q, x, ei, et)
    assert torch.allclose(node.norm(dim=-1), torch.ones(_N), atol=1e-5)   # L2 unchanged
    assert float(scores.min()) >= -1.001 and float(scores.max()) <= 1.001  # cosine range


# --------------------------------------------------------------------------- #
# Late-cosine scoring (ADR 0022 amendment) — per-tower L2, NO fusion MLP
# --------------------------------------------------------------------------- #
def test_late_cosine_scores(ctx):
    x, ei, et, embedder = ctx["x"], ctx["edge_index"], ctx["edge_type"], ctx["embedder"]
    torch.manual_seed(0)
    scorer = GNNScorer(RGCNEncoder(x.shape[1]), query_dim=_EMB_DIM)
    scorer.eval()
    q = torch.as_tensor(embedder.encode(["download an audiobook"])[0], dtype=torch.float)
    with torch.no_grad():
        scores = scorer.score(q, x, ei, et)
        node = scorer.node_embeddings(x, ei, et)
        qn = scorer.query_embedding(q)
    assert scores.shape == (_N,)
    assert float(scores.min()) >= -1.001 and float(scores.max()) <= 1.001   # cosine range
    # per-tower L2 normalization: unit norms on both towers
    assert torch.allclose(node.norm(dim=-1), torch.ones(_N), atol=1e-5)
    assert torch.allclose(qn.norm(), torch.tensor(1.0), atol=1e-5)


def test_no_query_node_fusion_module(ctx):
    """Two-tower structure: projections are per-tower; no Linear ingests query+node jointly."""
    scorer = GNNScorer(RGCNEncoder(ctx["x"].shape[1]), query_dim=_EMB_DIM, proj_dim=128)
    assert scorer.has_projection
    # Each projection touches ONE tower only: query_proj sees query_dim, node_proj sees node_dim.
    assert scorer.query_proj.in_features == _EMB_DIM                 # 384, NOT 384 + node_dim
    assert scorer.node_proj.in_features == scorer.encoder.out_dim    # 64
    # No linear layer anywhere ingests a fused (query+node) vector.
    fused_dim = _EMB_DIM + scorer.encoder.out_dim
    assert all(m.in_features != fused_dim for m in scorer.modules() if isinstance(m, nn.Linear))
    # The only learned parts are the encoder + the two per-tower projections (no fusion net).
    linears = [m for m in scorer.modules() if isinstance(m, nn.Linear)]
    assert {scorer.query_proj, scorer.node_proj} <= set(linears)


def test_optional_projection_can_be_disabled(ctx):
    """proj_dim=None works when the encoder already maps to the query dim (projection is optional)."""
    x, ei, et = ctx["x"], ctx["edge_index"], ctx["edge_type"]
    scorer = GNNScorer(RGCNEncoder(x.shape[1], out_dim=_EMB_DIM), query_dim=_EMB_DIM, proj_dim=None)
    assert scorer.has_projection is False
    scorer.eval()
    with torch.no_grad():
        scores = scorer.score(torch.ones(_EMB_DIM), x, ei, et)
    assert scores.shape == (_N,)


# --------------------------------------------------------------------------- #
# Determinism — same seed + same input → identical scores
# --------------------------------------------------------------------------- #
def test_determinism_same_seed(ctx):
    x, ei, et = ctx["x"], ctx["edge_index"], ctx["edge_type"]
    q = torch.ones(_EMB_DIM)
    outs = []
    for _ in range(2):
        torch.manual_seed(1234)
        scorer = GNNScorer(GATEncoder(x.shape[1]), query_dim=_EMB_DIM)
        scorer.eval()
        with torch.no_grad():
            outs.append(scorer.score(q, x, ei, et))
    assert torch.equal(outs[0], outs[1])


# --------------------------------------------------------------------------- #
# Stage-2 readiness — constructor search args honored
# --------------------------------------------------------------------------- #
def test_search_args_change_config(ctx):
    x, ei, et = ctx["x"], ctx["edge_index"], ctx["edge_type"]
    gat = GATEncoder(x.shape[1], hidden=32, heads=4, dropout=0.3)
    assert (gat.hidden, gat.heads, gat.dropout) == (32, 4, 0.3)
    assert gat.conv1.heads == 4 and gat.conv1.out_channels == 32     # heads/hidden propagate to convs
    gat.eval()
    with torch.no_grad():
        assert gat(x, ei, et).shape == (_N, 32)                      # out_dim tracks hidden
    rgcn = RGCNEncoder(x.shape[1], hidden=128, dropout=0.0)
    assert (rgcn.hidden, rgcn.dropout) == (128, 0.0)
    rgcn.eval()
    with torch.no_grad():
        assert rgcn(x, ei, et).shape == (_N, 128)


def test_base_is_abstract():
    with pytest.raises(NotImplementedError):
        GNNEncoder(10)  # base can't build convs itself
