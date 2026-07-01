# 0014 — Store processed artifacts as JSONL (record collections) + JSON (single metadata), not parquet

## Status

Accepted

## Context

ADR 0011 introduced a preprocessing stage that writes normalized artifacts to `data/processed/`.
This ADR fixes the on-disk **format** before that code is written (ADR 0009).

The processed data is **tiny and nested**: 573 tools (~1.5 MB total across raw files), each with a
`parameters[]` list and a `depends_on[]` list, plus 1,569 queries. It also must stay
**human-readable** so normalization can be debugged by eye (did `bool → boolean` actually happen? did
the malformed dep rows get fixed?).

Format guidance verified 2026-07-01:

- *JSONL vs JSON vs Parquet — When to Use Which* (jsonlkit.com) — "Parquet wins for analytical filters
  because it can skip most of the file, while **JSONL wins for streaming, debuggability, and
  append-only workloads**." <https://jsonlkit.com/jsonl-vs-json>
- *Parquet vs CSV vs JSON* (drivedatascience.com) — "for datasets **under a few megabytes**, the
  overhead of Parquet's metadata and structure may outweigh its benefits."
  <https://www.drivedatascience.com/parquet-csv-json-file-format-comparison/>

At ~1.5 MB with no analytical column-pruning workload, Parquet's columnar/compression benefits do not
apply, and it would add a `pyarrow` dependency and hurt readability.

## Decision

Processed artifacts use the **standard library** only:

- **`tools.jsonl`** — one normalized tool per line.
- **`queries.jsonl`** — one normalized query instance per line.
- **`metadata.json`** — a single JSON object: the normalization report, graph-level stats, and
  validation-hook results.

Record collections → JSONL; single metadata object → JSON. **No parquet at this scale; no new
dependency.**

## Consequences

- Artifacts are inspectable/diffable in any text editor — normalization is auditable by eye.
- Zero extra dependency; stdlib `json` reads/writes everything.
- `data/processed/` stays **gitignored** (regenerable from raw + code; ADR 0011).
- **Revisit only** if the data grows by orders of magnitude, or when bulk **embedding vectors** need
  storage — that is a separate, later decision (vectors are numeric/large and may warrant `.npy`/
  parquet), explicitly out of scope here.

## Alternatives considered

- **Parquet** — rejected: metadata/columnar overhead with no analytical benefit at ~1.5 MB, poor
  human-readability, and needs `pyarrow`.
- **One big JSON array** — rejected: JSONL is more append/stream/diff-friendly for record collections
  (a one-line change is a one-line diff, not a re-serialization of the whole array).
