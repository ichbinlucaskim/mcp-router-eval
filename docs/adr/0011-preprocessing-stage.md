# 0011 — Normalize raw data in a dedicated preprocessing stage; loaders read processed, not raw

## Status

Accepted

## Amendment — 2026-07-01 (acyclicity applies to the ordering sub-graph only)

The original validation hook says to "surface any cycle loudly." Cycle analysis
(`docs/feasibility-completion.md`) shows the **full 4-type graph is expected to be cyclic** (485/573
nodes; 1,374/1,569 gold sets) because of `TOOL_INDIRECTLY` `get_`/`set_` pairs — and that is
**acceptable**, because the full graph is the *router's* representation, not the ordering source.

**Amended hook spec:** the acyclicity assertion applies to the **ordering sub-graph (`PARAMETER_*`)
only** — the hook must assert the `PARAMETER_*` sub-graph is acyclic and **fail loudly** if not. It
must **not** assert acyclicity of the full 4-type graph (that would fail on valid data). This
distinction (ordering vs representation) is formalized in ADR 0013. All other validation assertions
below are unchanged.

## Context

Firsthand inspection (`docs/data-inspection-toollinkos.md`) found the raw ToolLinkOS data is dirty:

- **Type aliasing** — `bool`/`boolean` (26+16) and `int`/`integer` (189+15) both occur.
- **21 non-scalar params** typed `dict`/`list`/`array` — the schema builder can't assume flat scalars.
- **Side keys** — `enum` on 127 params, `default` on 54, both outside the `type` field.
- **2 malformed dependency rows** — `PARAMETER_DEPENDS_ON` (should be `PARAMETER_DIRECTLY_DEPENDS_ON`).
- **2 malformed param objects** — one missing `type`, one missing `required`.

The widely adopted **Cookiecutter Data Science** structure separates `data/raw` (the original,
**immutable** dump — "never modified; any cleaning creates a new dataset") from `data/processed`
(the final canonical sets consumed by modeling code). Verified 2026-07-01:

- Cookiecutter Data Science — <https://cookiecutter-data-science.drivendata.org/> (raw is immutable;
  transformations write to interim/processed).
- "Best Practices for Organizing and Coding Data Science Projects" (The Deep Hub, Medium) —
  raw input, temporary work, and final outputs kept in separate directories.

Recorded before `contracts.py` per ADR 0009.

## Decision

Introduce a dedicated **preprocessing stage** that reads `data/raw/` → normalizes → writes
`data/processed/`. Loaders and **all** pipeline code consume `data/processed/` only; `data/raw/`
stays read-only/immutable. Normalization rules, specified explicitly and version-controlled in code:

- **Type vocabulary → canonical set:** unify `bool → boolean` and `int → integer` (canonical
  spelling = the **long** form, matching JSON Schema's own `"integer"`/`"boolean"`).
- **Map every param into a JSON-Schema representation**, scalar and non-scalar (`dict → object`,
  `list`/`array → array`).
- **Fold side keys** `enum` and `default` into the built schema.
- **Normalize the 2 malformed `PARAMETER_DEPENDS_ON` rows → `param_direct`.**
- **Tolerate the 2 params missing `type`/`required`** (default `type` conservatively; treat missing
  `required` as `false`).
- **Assign `is_core` from `func_type`** — with **no** "core ⇒ leaf" assumption (ADR 0012).

Add a **post-normalization VALIDATION hook** that **fails loudly** and asserts: only the canonical
type vocabulary remains, **0** dangling dependency targets, exactly **573** tools, and referential
integrity of all golden names in `instances.json`.

## Consequences

- `contracts.py`'s `ToolSpec` describes the **normalized** shape, not the raw shape.
- Processed artifacts are fully regenerable from raw + code, so they stay **gitignored**
  (`data/processed/` holds no source of truth).
- Dirtiness is handled once, in one auditable place, rather than smeared across the pipeline.
- The validation hook turns silent data drift into an immediate, loud failure.

## Alternatives considered

- **Normalize inside the loader at load time** — rejected: hides dirtiness behind every read,
  non-reproducible, and re-done on every run.
- **Normalize inside contracts** — rejected: mixes schema validation with data cleaning, two
  concerns that must stay separate.
