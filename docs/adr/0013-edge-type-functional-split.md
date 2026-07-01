# 0013 — Separate edges by function: PARAMETER_* = execution precondition (ordering); TOOL_* = conceptual association (router representation)

## Status

Accepted

## Context

The cycle analysis in `docs/feasibility-completion.md` revealed that the two dependency families in
ToolLinkOS differ in **meaning** and in **graph properties**, and that conflating them caused a real
blocker:

- **`PARAMETER_*_DEPENDS_ON`** (`DIRECTLY` + `INDIRECTLY`, 644 edges) = "tool A needs a parameter
  value produced by tool B." This is a genuine **execution precondition**. The `PARAMETER_*`
  sub-graph is **acyclic across all 1,569 gold sets**.
- **`TOOL_*_DEPENDS_ON`** (`DIRECTLY` + `INDIRECTLY`, 850 edges) = conceptual/association relations
  (e.g. `get_`/`set_` status pairs, data-derivation mutuals). These are **not** run-order
  constraints, and the `TOOL_*` sub-graph is **cyclic** (426 nodes; the source of the
  1,374/1,569 gold-set cycles).

Treating all four relation types as one ordering graph made topo-sort ill-defined (ADR 0012
amendment). This ADR promotes the fix from a patch to a stated structural principle.

## Decision

The pipeline treats dependency edges as **two functional roles**:

- **Ordering role → `PARAMETER_*` sub-graph** (acyclic). Drives execution order and the
  structural-proxy completion checks (ADR 0004, ADR 0012).
- **Representation role → all 4 typed relations** remain available to the router/GNN as typed edges
  (`num_relations = 4` stays, ADR 0006).

`contracts.py` `Edge` keeps **all 4** relation types; downstream **ordering** consumes only the
`PARAMETER_*` subset.

## Consequences

- (a) Completion/ordering logic **filters to `PARAMETER_*`**; `TOOL_*` edges are ignored for run order.
- (b) The edge-type ablation (proposal ablation B) becomes **sharper**: it can now test
  `PARAMETER` vs `TOOL` as **functionally distinct families**, not just four undifferentiated
  relations — a more interpretable result about *which* dependency structure the GNN exploits.
- (c) The router still sees **all 4 types**, so no retrieval signal is lost.
- (d) ADR 0011's validation hook asserts acyclicity on the ordering sub-graph only.

## Alternatives considered

- **Break cycles heuristically in the full graph** (e.g. drop one edge per cycle) — rejected:
  arbitrary, and it hides the real semantic distinction between the two edge families.
- **Drop `TOOL_*` edges entirely** — rejected: they are real structure that may help the router's
  retrieval, and the thesis is about exploiting dependency structure; discarding half of it is
  premature.
