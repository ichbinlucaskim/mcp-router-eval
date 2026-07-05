"""LocalEmbedder — default provider wrapping a local BGE sentence-transformers model (ADR 0003).

Default for every relative comparison in the project: no API key, no cost. Uses
``BAAI/bge-small-en-v1.5`` (384-dim). The weights download to the Hugging Face cache on first use and
load locally thereafter. Implements the :class:`~mcp_router_eval.embedding.base.Embedder` interface, so
the versioned on-disk cache is inherited unchanged.

Determinism (ADR 0003): the model is put in ``eval`` mode and encoded under ``torch.no_grad()`` with no
dropout/sampling, so the same text always yields the identical vector (CPU float32). Embeddings are
**not** L2-normalized here — that is a downstream (router/gate) choice, kept out of the raw provider.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mcp_router_eval.embedding.base import DEFAULT_CACHE_DIR, Embedder

__all__ = ["LocalEmbedder"]


class LocalEmbedder(Embedder):
    """BGE sentence-transformers embedder (``BAAI/bge-small-en-v1.5``, 384-dim)."""

    MODEL_ID = "BAAI/bge-small-en-v1.5"
    DIM = 384

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        *,
        model_id: str = MODEL_ID,
        revision: str | None = None,
        device: str | None = None,
        model: object | None = None,
    ) -> None:
        """
        Args:
            cache_dir: embedding cache root (per-version sub-dirs; ADR 0003).
            model_id: sentence-transformers model id.
            revision: optional pinned revision; recorded in :attr:`version` when given.
            device: torch device string (``None`` lets sentence-transformers pick).
            model: an already-loaded ``SentenceTransformer`` to reuse instead of loading (test seam;
                avoids reloading the model across many small embedders).
        """
        super().__init__(cache_dir=cache_dir)
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._model = model  # lazily loaded on first compute if not injected
        # version = model id, plus the revision when pinned, so a revision change segregates the cache.
        self._version = model_id if revision is None else f"{model_id}@{revision}"

    @property
    def dim(self) -> int:
        return self.DIM

    @property
    def version(self) -> str:
        return self._version

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model_id, revision=self._revision, device=self._device)
            model.eval()  # deterministic: no dropout
            self._model = model
        return self._model

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model = self._load()
        with torch.no_grad():
            vecs = model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        return np.asarray(vecs, dtype=np.float32)
