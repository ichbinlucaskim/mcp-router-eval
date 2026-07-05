"""OpenAIEmbedder — optional provider using text-embedding-ada-002 (ADR 0003). STUB (not implemented).

Only needed to reproduce the reference paper's published dense-baseline numbers; requires
``OPENAI_API_KEY`` / ``AZURE_*`` credentials. It is **optional** (ADR 0003): the default and only
implemented provider is :class:`~mcp_router_eval.embedding.local.LocalEmbedder`.

This class exists to hold the interface slot (rule-of-three: the :class:`Embedder` interface is shaped
so ada-002 slots in unchanged). It is intentionally **not** implemented and calls **no** paid API —
constructing it raises ``NotImplementedError``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mcp_router_eval.embedding.base import Embedder

__all__ = ["OpenAIEmbedder"]


class OpenAIEmbedder(Embedder):
    """text-embedding-ada-002 provider (1536-dim). Not implemented — see module docstring (ADR 0003)."""

    MODEL_ID = "text-embedding-ada-002"
    DIM = 1536

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "embedding.openai_embed: OpenAIEmbedder is optional (ADR 0003) and not implemented; "
            "use LocalEmbedder. ada-002 is only for reproducing the paper's published numbers."
        )

    @property
    def dim(self) -> int:
        return self.DIM

    @property
    def version(self) -> str:
        return self.MODEL_ID

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("OpenAIEmbedder is not implemented (ADR 0003).")
