# ToolLinkOS — ground-truth data inspection

Firsthand inspection of the real dataset bytes (not second-hand from the paper/report). This is the
**ground-truth reference** for downstream schema and preprocessing decisions. Regenerate the data
with `scripts/fetch_data.py`; provenance is in `data/raw/SOURCE.md`.

- **Inspected:** 2026-07-01
- **Source commit:** `b630b98656e25c3b83a71ea0406572add38ae46d` (github.com/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS, MIT)
- **Files + SHA-256:**
  - `regular_tools.json` — 808,301 B — `dc3a944e1198aeaa83bc351f7444935f5a6e3e48059fbadbfc07e92be490f687`
  - `core_tools.json` — 27,886 B — `fe9ca1f02acffc6ae3120ecda203137c1c0c9714b79e9f0d031c7245b98e4be9`
  - `instances.json` — 713,178 B — `37e4d14268d3d2584cfd966ebb81151f7d12c514a5506f23ab85088e11185af9`

## Tools — 573 total (523 regular + 50 core)

- **Object shape is uniform:** every tool has exactly `{name, description, parameters, depends_on, func_type}`.
- **Identity is clean:** 573 names, all distinct, **zero** regular∩core collisions → tool identity = `name` holds.

### Parameter `type` vocabulary (dirty — needs normalization)

| type | count | | type | count |
| --- | ---: | --- | --- | ---: |
| `string` | 868 | | `dict` | 9 |
| `int` | 189 | | `list` | 7 |
| `float` | 71 | | `array` | 5 |
| `boolean` | 26 | | *(missing `type` key)* | 1 |
| `bool` | 16 | | | |
| `integer` | 15 | | | |

- **Aliasing:** both `boolean`/`bool` and `int`/`integer` occur — must be unified.
- **Non-scalar types:** 21 params are `dict`/`list`/`array` — the JSON-Schema builder cannot assume flat scalars.
- **Side keys:** `enum` appears on **127** params (type stays `string`), `default` on **54**.
- **Malformed param objects (2):** one lacks a `type` key, one lacks `required`.

### Core tools — **NOT leaves** (correction to second-hand claim)

- **30 of 50 core tools have non-empty `depends_on`** (e.g. `get_current_location`, `set_wifi_status`); only 20 are dependency-free.
- The earlier "core ⇒ empty `depends_on` / leaf utility" claim is **false**. `is_core` is a valid node label but implies **nothing** about having dependencies.
- All 523 regular tools have dependencies.

## Dependencies — 1,496 edges

| `dependence_type` | count |
| --- | ---: |
| `TOOL_DIRECTLY_DEPENDS_ON` | 676 |
| `PARAMETER_DIRECTLY_DEPENDS_ON` | 404 |
| `PARAMETER_INDIRECTLY_DEPENDS_ON` | 239 |
| `TOOL_INDIRECTLY_DEPENDS_ON` | 175 |
| `PARAMETER_DEPENDS_ON` *(malformed)* | 2 |

> **These are RAW counts** (`data/raw/` @ `b630b98`, pre-normalization). Preprocessing normalizes the 2
> malformed rows into `PARAMETER_DIRECTLY_DEPENDS_ON`, so the processed per-type totals differ (+1 each
> for the affected types); see `docs/feasibility-completion.md` for the raw↔processed reconciliation.

- **4 canonical types + exactly 2 malformed rows.** Both malformed rows are `PARAMETER_DEPENDS_ON`
  on `join_doctor_virtual_consultation` and `cancel_doctors_appointment`, both targeting
  `get_doctor_appointments` with a `parameter_name` → clearly intended as `PARAMETER_DIRECTLY_DEPENDS_ON`.
- **Every** dep entry has `{dependence_type, name, parameter_name, reason}` — `TOOL_*` edges carry a `parameter_name` too.
- **Dangling targets: 0** — every `depends_on.name` resolves to a real tool.

## Queries — 1,569 instances

- Uniform keys `{user_query, main_golden_function_name, golden_function_names}`.
- **`golden_function_names`:** min **2**, mean **6.02**, max **12**. **Zero** single-golden instances → confirms "no single-tool queries."
- **Set-like but main-first-ordered:** no duplicates in any list, but `main_golden_function_name`
  is at index 0 in **all 1,569** (last in 0). ⚠ The order is **main-then-deps, not execution order** —
  dependencies that must run before `main` appear *after* it. See ADR 0012.
- **Referential integrity: perfect** — every golden + main name resolves to a real tool; `main ∈ golden_function_names` in all 1,569.

## Implications (tracked in ADRs)

- Dirty type vocab + non-scalar params + malformed rows → dedicated preprocessing stage (**ADR 0011**).
- Golden order ≠ run order; core ≠ leaf → topo-sort for execution order, no core-is-leaf assumptions (**ADR 0012**).
