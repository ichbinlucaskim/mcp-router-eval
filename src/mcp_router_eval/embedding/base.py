"""Embedder interface: encode(texts) -> vectors, with caching and a version tag (ADR 0003).

One shared embedding space is used for baselines, GNN node features, and gate similarity so
comparisons stay relative. The version tag keys the on-disk cache and guards stale reuse.

STUB.
"""

raise NotImplementedError("embedding.base: interface not implemented yet")
