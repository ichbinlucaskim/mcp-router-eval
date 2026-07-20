"""Embedding provider (ADR 0003) — interface conformance, determinism, versioned cache, graph slot.

These tests load the real BGE model (``BAAI/bge-small-en-v1.5``); the weights download to the HF cache
on first run and load locally thereafter (huggingface.co reachability was verified before implementing).
The model is loaded once (module-scoped) and injected into each embedder so per-test cost stays small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mcp_router_eval.embedding.base import Embedder
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.embedding.openai_embed import OpenAIEmbedder

_HAS_DATA = (Path("data/processed") / "tools.jsonl").exists()
_TEXTS = ["download an audiobook", "validate an email address", "toggle wifi status"]


@pytest.fixture(scope="module")
def st_model():
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(LocalEmbedder.MODEL_ID)
    m.eval()
    return m


@pytest.fixture
def embedder(tmp_path, st_model) -> LocalEmbedder:
    """A LocalEmbedder with an isolated (empty) cache dir, reusing the shared model."""
    return LocalEmbedder(cache_dir=tmp_path, model=st_model)


# --------------------------------------------------------------------------- #
# Interface conformance
# --------------------------------------------------------------------------- #
def test_local_is_an_embedder(embedder):
    assert isinstance(embedder, Embedder)


def test_encode_shape_and_dtype(embedder):
    out = embedder.encode(_TEXTS)
    assert out.shape == (len(_TEXTS), 384)
    assert out.dtype == np.float32
    assert embedder.dim == 384


def test_empty_input_returns_zero_rows(embedder):
    out = embedder.encode([])
    assert out.shape == (0, 384)


# --------------------------------------------------------------------------- #
# Determinism — same text → identical vector (independent of caching)
# --------------------------------------------------------------------------- #
def test_determinism_independent_compute(tmp_path, st_model):
    a = LocalEmbedder(cache_dir=tmp_path / "a", model=st_model)
    b = LocalEmbedder(cache_dir=tmp_path / "b", model=st_model)  # separate empty cache → both recompute
    va, vb = a.encode(_TEXTS), b.encode(_TEXTS)
    assert a.last_computed == len(_TEXTS) and b.last_computed == len(_TEXTS)  # neither hit cache
    assert np.array_equal(va, vb)


# --------------------------------------------------------------------------- #
# Cache round-trip — second call loads from cache, no recompute
# --------------------------------------------------------------------------- #
def test_cache_round_trip(embedder, tmp_path):
    first = embedder.encode(_TEXTS)
    assert embedder.last_computed == len(_TEXTS)  # cold cache → all computed
    # cache files landed on disk under the per-version dir
    cache_files = list((tmp_path).rglob("*.npy"))
    assert len(cache_files) == len(_TEXTS)

    second = embedder.encode(_TEXTS)
    assert embedder.last_computed == 0  # warm cache → NOTHING recomputed
    assert np.array_equal(first, second)


def test_cache_recomputes_only_missing(embedder):
    embedder.encode(_TEXTS)
    embedder.encode([*_TEXTS, "a brand new sentence"])
    assert embedder.last_computed == 1  # only the new text is computed


# --------------------------------------------------------------------------- #
# Version tagging — model version recorded in cache metadata, retrievable
# --------------------------------------------------------------------------- #
def test_version_recorded_in_cache(embedder, tmp_path):
    embedder.encode(_TEXTS[:1])
    meta = embedder.cache_metadata()
    assert meta["version"] == embedder.version == LocalEmbedder.MODEL_ID
    assert meta["dim"] == 384
    # metadata physically on disk under the version-slugged sub-directory
    assert (tmp_path / "BAAI_bge-small-en-v1.5" / "meta.json").exists()


def test_revision_changes_version_and_cache_namespace(tmp_path, st_model):
    e = LocalEmbedder(cache_dir=tmp_path, model=st_model, revision="abc123")
    assert e.version == f"{LocalEmbedder.MODEL_ID}@abc123"
    e.encode(_TEXTS[:1])
    assert (tmp_path / "BAAI_bge-small-en-v1.5_abc123").exists()  # segregated from the unpinned dir


# --------------------------------------------------------------------------- #
# Small real run — embed real tool descriptions from data/processed/tools.jsonl
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_DATA, reason="processed data absent; run preprocess")
def test_real_tool_descriptions(embedder):
    from mcp_router_eval.data.loader import load

    ds = load()
    texts = []
    for tid in list(ds.tools)[:5]:
        props = ds.tools[tid].schema_.get("properties", {})
        descs = "; ".join(p.get("description", "") for p in props.values() if p.get("description"))
        texts.append(f"{tid}: {descs}")
    out = embedder.encode(texts)
    assert out.shape == (5, 384)
    assert out.dtype == np.float32


# --------------------------------------------------------------------------- #
# ToolGraph slot compatibility — produced dim fits the x[:, 1:] embedding placeholder
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_DATA, reason="processed data absent; run preprocess")
def test_fits_graph_node_feature_slot(embedder):
    from mcp_router_eval.data.graph_build import build_graph
    from mcp_router_eval.data.loader import load

    graph = build_graph(load())
    k = 4
    node_texts = list(graph.node_ids[:k])
    emb = embedder.encode(node_texts)  # (k, 384)

    x = graph.data.x.numpy()  # (N, 1): column 0 is is_core
    assert x.shape[1] == 1  # only the structural column so far (embeddings appended later)
    # Interface alignment ONLY — we do not fill the graph here (that is the router step).
    combined = np.hstack([x[:k], emb])
    assert combined.shape == (k, 1 + 384)  # embedding dim slots cleanly into x[:, 1:]


# --------------------------------------------------------------------------- #
# OpenAI provider stays an interface-satisfying stub (ADR 0003) — no paid API
# --------------------------------------------------------------------------- #
def test_openai_embedder_is_stub():
    assert issubclass(OpenAIEmbedder, Embedder)  # satisfies the interface slot
    with pytest.raises(NotImplementedError):
        OpenAIEmbedder()
