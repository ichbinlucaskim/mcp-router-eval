"""Embedder interface — encode(texts) → vectors, with an on-disk cache keyed by model version (ADR 0003).

One shared embedding space feeds the vector baselines (§5.1), the GNN node features (§3.1, appended as
``x[:, 1:]`` — see ``data/graph_build.py``), and the gate's homophily signal. Because those uses are
**relative** comparisons, the specific vendor model does not matter — only that every vector in a run
comes from the *same* model. The :attr:`version` tag records **which** model produced a vector; it keys
the cache and is stored in cache metadata so results stay traceable and vectors from different models
are never silently mixed.

Layering
--------
This module freezes the interface and owns the **caching + version** machinery so every provider gets
it for free. A concrete provider implements only three things:

* :attr:`dim` — the output dimensionality,
* :attr:`version` — the model-id (+ revision) string,
* :meth:`_embed` — compute vectors for a batch of texts (no caching; the base wraps it).

:meth:`encode` is the public, cache-aware entry point. ``LocalEmbedder`` (BGE) is the default provider;
a future ``OpenAIEmbedder`` (ada-002) satisfies the *same* interface (rule-of-three: the interface is
shaped to accommodate that second case, but only the local provider is built here).

Cache-key convention
--------------------
``cache_key(version, text) = sha256(version + "\\x00" + text)`` (hex). Including the version in the key
— **and** segregating vectors into a per-version sub-directory — means a model change lands in a fresh
namespace and can never collide with or overwrite another model's vectors.
"""
from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = ["Embedder", "DEFAULT_CACHE_DIR"]

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent
#: Default cache root. Under ``data/processed/`` so it is regenerable and already gitignored
#: (``data/processed/*``, ADR 0011/0014). Each provider version gets its own sub-directory.
DEFAULT_CACHE_DIR = _PKG_ROOT / "data" / "processed" / "embeddings"


def _version_slug(version: str) -> str:
    """Filesystem-safe sub-directory name for a model version (e.g. ``BAAI/bge-small-en-v1.5``)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", version)


class Embedder(ABC):
    """Provider-agnostic text embedder with a versioned on-disk cache (ADR 0003).

    Subclasses implement :attr:`dim`, :attr:`version`, and :meth:`_embed`. Callers use :meth:`encode`.
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self._cache_root = Path(cache_dir)
        #: Number of texts actually recomputed on the most recent :meth:`encode` call (0 ⇒ all cache
        #: hits). A simple, inspectable marker so tests can assert cache behavior without a spy.
        self.last_computed: int = 0

    # --- provider contract -------------------------------------------------- #
    @property
    @abstractmethod
    def dim(self) -> int:
        """Output dimensionality; every returned vector has this length."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Model identity (model-id, plus revision if available). Keys the cache; recorded in metadata."""

    @abstractmethod
    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Compute embeddings for ``texts`` → ``(len(texts), dim)`` float32. No caching (the base wraps)."""

    # --- cache-key convention ---------------------------------------------- #
    @staticmethod
    def cache_key(version: str, text: str) -> str:
        """Deterministic cache key: ``sha256(version + "\\x00" + text)`` (hex)."""
        h = hashlib.sha256()
        h.update(version.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    # --- cache plumbing ----------------------------------------------------- #
    def _cache_dir(self) -> Path:
        """Per-version cache directory, created on demand with a ``meta.json`` version record."""
        d = self._cache_root / _version_slug(self.version)
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
        meta = d / "meta.json"
        if not meta.exists():
            meta.write_text(
                json.dumps({"version": self.version, "dim": self.dim}, indent=2, sort_keys=True)
            )
        return d

    def cache_metadata(self) -> dict:
        """Return the persisted ``{version, dim}`` record for this embedder's cache (creating it if new)."""
        return json.loads((self._cache_dir() / "meta.json").read_text())

    # --- public API --------------------------------------------------------- #
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` → ``(len(texts), dim)`` float32, deterministically, using the cache.

        Cache hits are loaded from disk; only cache-missing texts are recomputed (in one batch) and
        then written back. Same texts + same :attr:`version` → identical vectors, and
        :attr:`last_computed` reports how many were (re)computed (0 ⇒ fully served from cache).
        """
        texts = list(texts)
        if not texts:
            self.last_computed = 0
            return np.empty((0, self.dim), dtype=np.float32)

        cache_dir = self._cache_dir()
        results: list[np.ndarray | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, text in enumerate(texts):
            path = cache_dir / f"{self.cache_key(self.version, text)}.npy"
            if path.exists():
                results[i] = np.load(path)
            else:
                missing_idx.append(i)
                missing_texts.append(text)

        self.last_computed = len(missing_texts)
        if missing_texts:
            computed = np.asarray(self._embed(missing_texts), dtype=np.float32)
            if computed.shape != (len(missing_texts), self.dim):
                raise ValueError(
                    f"{type(self).__name__}._embed returned {computed.shape}, "
                    f"expected {(len(missing_texts), self.dim)} (model/dim mismatch)"
                )
            for j, i in enumerate(missing_idx):
                vec = computed[j]
                results[i] = vec
                np.save(cache_dir / f"{self.cache_key(self.version, missing_texts[j])}.npy", vec)

        return np.stack(results).astype(np.float32)  # type: ignore[arg-type]
