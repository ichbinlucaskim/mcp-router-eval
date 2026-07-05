# 0016 — Mock executor synthesizes arguments minimally (required, type-valid) honoring enum/default, deterministically

## Status

Accepted

## Context

The deterministic mock runner (ADR 0015) consumes an `ExecPlan` and must emit `ToolCall`s with concrete
arguments — but the instances carry **no gold arguments**: the ground-truth inspection
(`docs/data-inspection-toollinkos.md`) found queries and dependency structure, not per-call argument
values. So the mock runner must **synthesize** the arguments itself.

What it needs to synthesize is narrow. Completion is a **structural proxy** (ADR 0004): the verdict is
*type-valid args against the built JSON Schema*, not *semantically meaningful values* — the tools are
fictional and execute nothing, so a "correct" string has no meaning to satisfy. In agent-eval practice
a mock stands in for the **tool implementation**, while the agent's own reasoning and
**parameter-extraction from natural language** is a *separate* concern that belongs to the LLM, not the
mock: testing with mocks focuses on "the agent's reasoning and tool selection logic, and the LLM's
ability to extract parameters... while the actual tool implementation (that's mocked out)... [is] not
tested" ([LangWatch Scenario — Mocking External APIs in Agent
Tests](https://langwatch.ai/scenario/testing-guides/mocks/)). This maps cleanly onto our two layers
(ADR 0015): **mock = schema-based argument synthesis; SDK replay = reasoning/extraction demonstration.**

Generating type-valid instances *from a JSON Schema* is a standard, well-supported operation, and its
knobs give us exactly the behavior we want — `requiredOnly` (generate only required properties),
`useDefaultValue` (emit the schema `default` when present), `fixedProbabilities` + `seed` (deterministic
output) ([json-schema-faker](https://github.com/json-schema-faker/json-schema-faker)). The keywords we
honor — `type`, `enum`, `default` — are the structural-validation vocabulary of the JSON Schema
Validation spec: an instance validates against `enum` only if it equals one of the listed values, and
`default` supplies a value for a missing one ([JSON Schema Validation: A Vocabulary for Structural
Validation of JSON](https://datatracker.ietf.org/doc/html/draft-bhutton-json-schema-validation-00)).

This matters for our data specifically. The 573 processed tools carry **127 `enum`-constrained** and
**54 `default`** parameters (verified over `data/processed/tools.jsonl`); the property types are
overwhelmingly `string` (869), `integer` (204), `number` (71), `boolean` (42), `array` (12), `object`
(9). Ignoring `enum`/`default` would produce args that fail the very schema built in preprocess.

## Decision

The mock runner synthesizes each required argument as follows:

1. **Required fields only, type-valid.** Synthesize only the schema's `required` properties, each a
   fixed canonical dummy for its `type`:
   - `string` → `"synthetic"` (fixed dummy string)
   - `integer` → `0`
   - `number` → `0.0`
   - `boolean` → `false`
   - `array` → `[]` (minimal valid instance)
   - `object` → `{}` recursively populated with *its* required properties by the same rules
2. **Honor `enum` and `default`.** If a property has `enum`, pick the **first** enum value (ignoring the
   type dummy). If it has `default`, use the default. (`default` and `enum` take precedence over the
   type dummy; where both are present, prefer `default` — it is by construction a valid choice.) This is
   what makes the 127 enum / 54 default params actually schema-satisfying.
3. **Deterministic.** The same `ExecPlan` yields **identical** args — fixed per-type dummies and
   fixed selection rules (first enum value), i.e. no randomness (equivalently, a fixed seed with
   `fixedProbabilities`). This is what preserves ablation A's reproducibility (router isolated,
   executor held constant — ADR 0015).
4. **Provenance.** Every synthesized argument is marked `synthetic=True` on the `ToolCall` (ADR 0004),
   so downstream consumers can tell structural stand-ins from real values.
5. **Out of scope — data-flow chaining.** Threading a dependency tool's *output* into a downstream
   tool's *argument* (semantic data-flow) is **not** done here. That is agent-reasoning territory,
   exercised only via the SDK replay adapter (ADR 0015), never by the structural mock.

## Consequences

- Argument validity is **deterministic and reproducible**: identical `ExecPlan` → identical `ToolCall`
  args → identical `ExecResult`.
- Honoring `enum`/`default` means synthesized calls **genuinely satisfy** the per-tool JSON Schema built
  in preprocess (ADR 0014), including the 127 enum-constrained params that bare type dummies would break.
- **Semantic completion stays unmeasured** — an honest, documented limitation of the structural proxy
  (ADR 0004), not a defect of the mock. Semantic/data-flow behavior is only ever demonstrated on the
  replay layer.
- The `synthetic=True` flag keeps the structural origin of every value auditable in the trace.

## Alternatives considered

- **Rich semantic synthesis / data-flow chaining** — rejected: buys nothing on fictional, non-executing
  tools, adds real complexity, and parameter-reasoning belongs to the replay layer, not the structural
  mock.
- **Bare type dummies, ignoring `enum`** — rejected: would emit schema-*invalid* args on the 127
  enum-constrained params (a dummy `"synthetic"` string is not a member of a closed enum), corrupting
  the structural verdict the mock exists to compute.

## Sources

- json-schema-faker — `requiredOnly` / `useDefaultValue` / `fixedProbabilities` / `seed`
  (schema→valid data; determinism; minimal + honor defaults):
  <https://github.com/json-schema-faker/json-schema-faker>
- LangWatch Scenario — mock replaces the tool implementation; parameter-extraction is the LLM's job:
  <https://langwatch.ai/scenario/testing-guides/mocks/>
- JSON Schema Validation: A Vocabulary for Structural Validation of JSON — `type` / `enum` / `default`
  semantics: <https://datatracker.ietf.org/doc/html/draft-bhutton-json-schema-validation-00>
