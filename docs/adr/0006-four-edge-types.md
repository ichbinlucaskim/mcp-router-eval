# 0006 — 4 real dependency edge types; "core" is a node feature, not an edge

## Status

Accepted

## Context

The proposal (§3.1, §5.2) models dependency edges with three types — `param_source`, `precond`,
`core` — and sets the R-GCN accordingly. Inspection of the actual `depends_on` data
(`docs/build-readiness-report.md` §2, §4) found:

- Four real `dependence_type` values: `PARAMETER_DIRECTLY_DEPENDS_ON`,
  `PARAMETER_INDIRECTLY_DEPENDS_ON`, `TOOL_DIRECTLY_DEPENDS_ON`, `TOOL_INDIRECTLY_DEPENDS_ON`.
- **"Core" is not an edge type** — it is a node attribute (`func_type ∈ {regular, core}`); core
  tools are leaf utilities with empty `depends_on`.
- Two malformed `PARAMETER_DEPENDS_ON` rows exist in the data.

## Decision

Model dependency edges with the **4 real types** (param × {direct, indirect}, tool ×
{direct, indirect}) → R-GCN `num_relations = 4`. Encode "core" as an `is_core` **node feature**, not
an edge. Clean/drop the 2 malformed `PARAMETER_DEPENDS_ON` rows at load time.

## Consequences

- `RouteResult.Edge.type` enum becomes the 4 real types (contract change tracked for T1.1).
- Edge-type ablation B drops these 4 types (param vs tool, direct vs indirect), a cleaner design
  than the proposal's original scheme.
- `graph_build.py` sets `num_relations=4` and adds `is_core` to node features.

## Alternatives considered

- **Collapse to 2 types** (param_source / precond) with a `direct` flag — viable fallback; kept
  in reserve if 4-way typing proves too sparse to learn.
- **Keep the proposal's 3-type "core edge" scheme** — rejected: does not exist in the data.
