# 0003 — Embedding behind a provider interface; LocalEmbedder(BGE) default, ada-002 optional

## Status

Accepted

## Context

Node features (§3.1), dense baselines (§5.1), and the gate's homophily signal all need text
embeddings. From `docs/build-readiness-report.md`:

- Claude/Anthropic provides **no embedding API**, so embeddings come from a separate provider.
- The **relative** comparisons at the heart of this project (router vs. router, sliced by query
  type) only require **one shared embedding space**, not a specific vendor model.
- The reference paper's **published** dense-baseline numbers were produced with Azure OpenAI
  **text-embedding-ada-002**; matching them within ±2% requires that exact model (a paid API).

## Decision

All embedding access goes through a single `Embedder` interface (`encode`, on-disk cache, version
tag). `LocalEmbedder` (a local BGE sentence-transformers model) is the default for all relative
comparisons. `OpenAIEmbedder` (ada-002) is an optional path used only to reproduce published
numbers. The version tag keys the cache to prevent stale reuse across providers.

## Consequences

- No vendor lock-in and no API cost for the core experiments; ada-002 keys are needed only for
  optional reproduction.
- Baseline numbers produced with `LocalEmbedder` will **not** match the paper's table; that is
  acceptable for relative claims and flagged wherever absolute numbers are reported.
- `configs/default.yaml` sets `embedding.provider: local_bge`.

## Alternatives considered

- **ada-002 as the default** — rejected: adds cost and an API dependency to every run for no benefit
  to relative comparisons.
- **Hard-code a single embedder** — rejected: prevents swapping in ada-002 for reproduction.
