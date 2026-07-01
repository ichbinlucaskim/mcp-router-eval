# 0008 — Graph identity + metadata conventions: tool identity = name; query_id = q{index}

## Status

Accepted

## Context

The contracts (§3) key on `tool_id` and `query_id`, but the actual data
(`docs/build-readiness-report.md` §2, §4) has neither:

- Tools are identified only by their **`name`** string; there is no separate `tool_id` field.
- `instances.json` entries have **no id** at all.

Stable keys are required so `RouteResult`, `ExecPlan`, `ExecResult`, and `Attribution` can be joined
deterministically.

## Decision

Adopt **tool identity = the tool `name` string** (used directly as `tool_id`), and assign a
**synthetic `query_id = q{index}`** at load time (index into `instances.json`), frozen for the life
of the project. Both conventions live in `data/loader.py`.

## Consequences

- All four contracts use these keys; join/attribution logic is deterministic and reproducible.
- Reordering or filtering `instances.json` would change `q{index}` ids, so the load order is frozen.
- `depends_on` references (by name) map directly onto tool identity with no extra id table.

## Alternatives considered

- **Hash-based query ids** — rejected: opaque and harder to trace back to a source instance.
- **Introduce a separate integer tool_id table** — rejected: unnecessary indirection when `name` is
  already unique.
