# 0004 — Completion is a structural proxy, not semantic success

## Status

Accepted

## Context

Layer 3 reports `completion_rate`, and RQ1 depends on comparing it against retrieval recall. But
(`docs/build-readiness-report.md` §2–3):

- ToolLinkOS tools are **fictional and do not execute**.
- `instances.json` provides gold tool **names** only — **no gold arguments and no gold answer** — so
  there is nothing to check semantic task success against.

## Decision

`completed` is defined as a **structural proxy**: a query is complete iff the agent invokes exactly
the required tool set (`golden_function_names`), respects dependency order (a tool is called only
after its `depends_on` are satisfied), and every call is type-valid against the tool's built
JSONSchema. This is implemented by the generic mock executor (executor/mock_tools.py).

## Consequences

- `transfer_loss = recall@k − completion_rate` measures structural, not semantic, completion; this
  is documented wherever the metric appears.
- No promise of real functional completion is made; keeps the executor within the Week 4–5 budget.
- The mock executor synthesizes type-valid args (marked synthetic in `ExecResult` evidence).

## Alternatives considered

- **Real functional execution** — rejected: impossible with fictional, non-runnable tools.
- **LLM-judged semantic completion** — rejected: no gold answer to judge against; adds cost and
  non-determinism to a verdict that must be reproducible.

## Amendment 2026-07-05 — completion required-set is the PARAMETER_* dependency spine

This refines (does **not** overwrite) the Decision above, reconciling it with the later,
cycle-finding-aware documents that operationalized completion.

### Decision

The completion **required-set** = the **`PARAMETER_*` dependency closure of the query's gold tools** (the
dependency *spine*), **not** the full `golden_function_names`. `TOOL_*`-associated gold tools
(system/connectivity tools that carry no required-argument dependency) are **excluded** from the
completion requirement. The ordering and type-validity parts of the original Decision are **unchanged** —
this amendment only fixes *which set of tools must be invoked*.

### Reconciliation (our own documents, quoted with file:line)

The original Decision says a query completes iff the agent *"invokes exactly the required tool set
(`golden_function_names`)"* (`docs/adr/0004-…:18-19`). Two **later** documents, both written to reconcile
with the ADR-0012 cycle finding, refine that intent to the `PARAMETER_*` spine:

- **ADR-0013** — *"**Ordering role → `PARAMETER_*` sub-graph** … Drives execution order and the
  **structural-proxy completion checks** (ADR 0004, ADR 0012)"* (`docs/adr/0013-…:28-29`) and
  *"Completion/ordering logic **filters to `PARAMETER_*`**; `TOOL_*` edges are ignored for run order"*
  (`docs/adr/0013-…:38`).
- **`completion-scoring-examples.md`** — *"`completed = true` iff: (1) the selected set contains **the
  full dependency closure of the gold tools** …"* (`:8-12`); the `get_wifi_status` / `get_battery_status`
  / `*_wifi_*` / `*_low_battery_*` / `get_system_language` gold tools *"are attached by `TOOL_*` edges
  only; they carry **no required-arg dependency** and are **excluded from ordering**"* (`:32-37`); and the
  worked q240 example completes on the **3-tool spine** —
  `gold_closure: [validate_email, audible_account_login, download_audible_book]` (`:53`),
  `router.selected_tools: [download_audible_book, audible_account_login, validate_email]` (`:56`) →
  `verdict.completed: true` (`:69`) — **not** the 9-name golden set.

> **Correction (superseded by [ADR-0030](0030-completion-required-set.md)).** The inherited reason
> *"`get_system_language` … attached by `TOOL_*` edges only"* (quoted above from
> `completion-scoring-examples.md:32-37`) is **factually wrong**. Measured on `data/processed/tools.jsonl`
> this session, `get_system_language` is a **`PARAMETER_*` source** (`param_indirect`) of
> `audible_account_login`'s **optional** `language` argument
> (`audible_account_login.required = [email, password]`; `get_system_language.deps = []`). The **3-tool
> q240 spine is still correct**, but *because* the `language` arg is **optional** — so under the
> required-argument rule (ADR-0030, variant A) its source is excluded from the completion required-set —
> **not** because the edge is `TOOL_*`. The `get_wifi_status` / `*_wifi_*` / `*_low_battery_*` /
> `get_battery_status` golds remain genuinely `TOOL_*`-only. This amendment's history is kept intact; only
> the edge-type *reason* for excluding `get_system_language` is corrected.

ADR-0013 and the scoring examples post-date ADR-0004 and explicitly reconcile with the cycle finding, so
they refine ADR-0004's intent. This amendment makes ADR-0004 consistent with them: the earlier phrasing
`golden_function_names` predates the `PARAMETER_*` / `TOOL_*` functional split (ADR 0013) and is superseded
here by the spine.

### Why this is not benchmark-gaming (stated explicitly)

The excluded `TOOL_*` golds are **query-irrelevant label associations with no execution precondition**.
Measured on our own data (the diagnosis over the 235 validation queries, seed 0): the mean gold set has
**5.8 tools, of which ~3.2 lie *outside* the `PARAMETER_*` spine** (TOOL_*-attached), and the
most-frequently-missed golds are `get_wifi_status` (missed in **184/235** queries) and `set_wifi_status`
(**184/235**) — generic connectivity tools attached to nearly every task regardless of what it is.
Requiring their recall measures **label noise, not structural task completion**, and matches the data's
known dirtiness (the recorded dependency-semantic gap of mean lexical-Jaccard **0.08**,
`docs/feasibility-completion.md`). Concretely, under our data BM25's completion is **0.098** against the
full golden set but **0.877** against the `PARAMETER_*` spine.

The spine is also the only **non-circular** choice: scoring against `route.selected_tools` (as some
earlier router integration tests did) always passes because the selection contains its own closure by
construction (circular, measures nothing); the full golden set is the opposite extreme (unreachable label
noise); the `PARAMETER_*` spine — anchored on the gold main tool + its required-argument dependencies,
independent of the router — is the substantive middle that reflects real structural completion.

### Transparency (anti-cherry-pick safeguard)

The evaluation reports **both**: the `PARAMETER_*`-spine `completion_rate` as the **primary**, north-star
metric, **and** the full-golden-set `completion_rate` as a **secondary** reported number — so the choice
of required-set is auditable, not hidden.

### Scope

This changes only **what the harness passes as `required_tools`** for completion scoring (the harness /
eval layer), to be implemented as the **next step**; the executor's completion mechanism, ordering, and
type-validity checks (ADR 0004/0012/0016) are unchanged.
