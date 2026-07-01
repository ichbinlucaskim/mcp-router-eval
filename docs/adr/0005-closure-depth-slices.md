# 0005 — Query slice = closure-depth buckets (shallow 2-3 vs deep >=6)

## Status

Accepted

## Context

The proposal's headline result format slices every metric by **single-tool** vs.
**dependency/composite** query, because the GNN's expected advantage lives in the composite slice
(§4). But inspection (`docs/build-readiness-report.md` §2) found:

- ToolLinkOS has **zero single-tool queries**: every one of the 1,569 instances has **≥2** gold
  functions (minimum `golden_function_names` size is 2; mean ≈ 6).

The single-tool bucket would therefore be empty.

## Decision

Replace single-vs-composite with **closure-depth buckets** derived from gold-set size:
**shallow = 2–3 gold tools**, **deep = ≥6 gold tools**. Depth still isolates the composite regime
where dependency-aware message passing should help.

## Consequences

- `eval/slices.py` and `configs/default.yaml` encode these thresholds.
- T0.3's query-taxonomy done-when is redefined around depth buckets, not single-vs-composite.
- Claims about "where the GNN wins" are stated in terms of closure depth.

## Alternatives considered

- **Synthesize single-tool queries** from dependency-free tools — deferred: possible but adds a
  synthetic slice not present in the benchmark.
- **Keep single-vs-composite** — rejected: the single-tool slice has no data.
