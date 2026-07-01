# 0012 — Derive execution order by dependency topo-sort; golden_function_names order is not runnable order

## Status

Accepted

## Amendment — 2026-07-01 (which sub-graph the topo-sort runs on)

The original Decision said "topological sort of the dependency graph" without specifying edge types.
Firsthand cycle analysis (`docs/feasibility-completion.md`) then found the **full 4-type graph is
cyclic** — 485/573 nodes in cycles; **1,374/1,569 gold sets** contain a cycle — caused by
`TOOL_INDIRECTLY` `get_`/`set_` pairs (e.g. `get_location_service_status ⇄ set_location_service_status`).
A topo-sort over all edge types is therefore ill-defined.

**Amended decision:** execution order = topological sort of the **`PARAMETER_*` sub-graph only**
(`PARAMETER_DIRECTLY_DEPENDS_ON` + `PARAMETER_INDIRECTLY_DEPENDS_ON`). **`TOOL_*` edges are EXCLUDED
from ordering.** Rationale: the `PARAMETER_*` sub-graph is **acyclic across all 1,569 gold sets**, and
`PARAMETER_*_DEPENDS_ON` means "needs an argument value produced by another tool" — the true run
precondition — whereas `TOOL_*` encodes conceptual association, not a run-order constraint. See
ADR 0013 for the general functional split. The original Decision below stands except that "dependency
graph" is now scoped to the `PARAMETER_*` sub-graph.

## Context

Two data-verified facts from firsthand inspection (`docs/data-inspection-toollinkos.md`):

- **`golden_function_names` is main-first, not execution order.** `main_golden_function_name` sits at
  index 0 in **all 1,569** instances (last in 0). Dependencies that must run *before* the main tool
  therefore appear *after* it in the list — the list order is **not** a runnable sequence.
- **Core tools are not leaves.** 30 of 50 core tools have their own `depends_on`; only 20 are
  dependency-free.

ADR 0004 defines completion as a structural proxy that includes a **dependency-order** check, so the
notion of "correct order" must be defined against real dependency structure, not list position.

## Decision

- The completion dependency-order check (ADR 0004) derives run order via **topological sort of the
  dependency graph**, and **never** from `golden_function_names` list order.
- **No code anywhere assumes `core ⇒ leaf`.** Core tools are treated as ordinary nodes that may have
  dependencies; `is_core` is a label only.

## Consequences

- ADR 0004's structural-proxy completion depends on a correct topological sort of the (cleaned,
  ADR 0011) dependency graph; the two ADRs are coupled and should move together.
- A cycle in the dependency graph would make topo-sort ill-defined — the preprocessing validation
  hook (ADR 0011) should surface any cycle loudly rather than let it corrupt the order check.
- `golden_function_names` remains the correct **set** for retrieval metrics and the Attribution
  required-tool set (ADR 0004 §3.4); only its **order** is discarded.

## Alternatives considered

- **Trust `golden_function_names` order as the execution sequence** — rejected: it is main-first, so
  it would run the main tool before its dependencies.
- **Special-case core tools as always-satisfied leaves** — rejected: 30/50 have real dependencies,
  so this would skip required predecessors.
