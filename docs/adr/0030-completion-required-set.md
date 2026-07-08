# 0030 — Completion required-set = required-argument PARAMETER_* closure (variant A), decoupled from the execution/ordering closure

## Status

Accepted. **Supersedes the ADR-0004 2026-07-05 amendment's A/B ambiguity** (that amendment picked "the
spine" but straddled two incompatible spine definitions and carried a data-factual error; both are
resolved here). ADR-0004's core decision — completion is a *structural proxy* — stands unchanged.

## Context

The full-project consistency checkup found the completion **required-set** — *which* tools must be
invoked for a query to count as structurally complete — wired **three different ways** at once:

1. **Execution/ordering closure = full `PARAMETER_*` (variant B).** `routers/closure.py:48` expands a
   router's top-k by pulling in **every** `PARAMETER_*` dependency source, with no check on whether the
   sourced argument is required:
   > `if dep.relation in ORDERING_RELATIONS and dep.source not in selected:` (`routers/closure.py:48`)
2. **Completion verdict = the full 9-tool golden set.** The harness scores completion against the whole
   `golden_function_names`:
   > `gold = frozenset(query.required_tools)` (`src/mcp_router_eval/eval/harness.py:92`), then
   > `mock_run(plan, tool_deps, list(gold))` / `attribute(..., required_tools=list(gold))`
   > (`harness.py:101-102`).
3. **ADR-0004 amendment intent = "the spine"** — but self-contradictory. It says the required-set is
   > "the **`PARAMETER_*` dependency closure** of the query's gold tools (the dependency *spine*), **not**
   > the full `golden_function_names`" (`docs/adr/0004-completion-structural-proxy.md:43`)
   which reads as the *full* `PARAMETER_*` closure (variant B), yet its worked example uses the **3-tool**
   set and excludes `get_system_language` on a **false** premise:
   > "the … `get_system_language` gold tools *are attached by `TOOL_*` edges only; they carry no
   > required-arg dependency and are excluded from ordering*" (`docs/adr/0004-…:62`), completing on the
   > "**3-tool spine** — `gold_closure: [validate_email, audible_account_login, download_audible_book]`"
   > (`docs/adr/0004-…:63-66`).

**The data contradicts the "`TOOL_*` edges only" claim.** Measured over `data/processed/tools.jsonl`
this session:

- `audible_account_login` has a **`param_indirect`** edge to `get_system_language`, sourcing the argument
  `language` — i.e. `get_system_language` is a `PARAMETER_*` node, **not** `TOOL_*`-only. (`get_system_language`
  itself has `deps=[]`.)
- That argument is **optional**: `audible_account_login.required = [email, password]`; `language` is a
  declared property but **not required**. (`download_audible_book.required = [book_name, language,
  session_id]` — its *own* `language` is required and has no producer, sourced by synthesis.)
- This is **systemic, not a q240 one-off**: `get_system_language` appears in the gold set of **414**
  queries and in the full-`PARAMETER_*` (variant-B) closure of **399** of them.

So the amendment reaches the *right 3-tool set* for q240 but via a *false reason* (calling
`get_system_language` `TOOL_*`-only). The correct reason is that `language` is **optional**, so under a
required-argument rule `get_system_language`'s source is excluded — which is exactly **variant A**.

The two candidate spine definitions, over q240 (`ORDERING_RELATIONS = {param_direct, param_indirect}`,
`contracts.py:70-72`; q240 param edges: `download_audible_book --param_direct--> audible_account_login`
(`session_id`); `audible_account_login --param_direct--> validate_email` (`email`),
`--param_indirect--> get_system_language` (`language`)):

- **Variant A — required-argument `PARAMETER_*` closure:** `[validate_email, audible_account_login,
  download_audible_book]` (**3 tools**). `get_system_language` excluded because `language` is optional.
- **Variant B — full `PARAMETER_*` closure:** `+ get_system_language` (**4 tools**). This is what
  `closure.py:48` currently computes.

Scoring completion against the full golden set (state #2) is the diagnosed bug: the excluded golds are
query-irrelevant `TOOL_*` label noise (per the amendment's own diagnosis, `get_wifi_status` missed in
**184/235** validation queries; mean gold **5.80** tools of which ~3 are `TOOL_*`-attached), so
`val_completion ≈ 0` for **every** router — measured `best_configs.json`: rgcn `val_completion=0.0`
(`val_map=0.444`), gat `0.047` (`0.431`), sage `0.004` (`0.389`); BM25 is **0.098** against the full
golden set vs **0.877** against the spine (`docs/adr/0004-…:83`). That collapse is **not GNN-specific**;
it is the required-set definition hitting all five routers.

## Decision

1. **Completion required-set = variant A — the required-argument `PARAMETER_*` closure.** A query's
   completion required-set is the transitive `PARAMETER_*` closure of its gold tools **restricted to
   edges that source a *required* argument of the dependent tool**. A `PARAMETER_*` source whose argument
   is *optional* on the consumer (e.g. `get_system_language` sourcing `audible_account_login.language`)
   is **not** in the completion required-set. Rationale (north star, ADR 0004): structural task
   completion means the tools **without which the task cannot structurally complete** — its
   required-argument dependencies. An optional argument has a default, so the task completes without its
   source; requiring it would measure label noise, not completion.

2. **Decouple execution/ordering from completion.** The execution/ordering closure **stays variant B**
   (full `PARAMETER_*`, ADR 0012/0013): `closure.py` may still pull an optional source into the selected
   set, and **if** an optional source is selected its `PARAMETER_*` **order must still be respected**. The
   completion required-set (variant A) is a **strict subset** of the ordering/execution closure
   (variant B). This is intended, not a contradiction: **"what must be present to complete" ≠ "what order
   things take if present"**. `required-set ⊊ ordering-set` is the designed relationship.

3. **Optional-param closure semantics (pins checkup #3 across `invariants.py` / `mock_tools.py` /
   `closure.py`).** An **optional** `PARAMETER_*`-sourced argument:
   - does **NOT** count toward completion or closure-completeness — its absence is **not** a dangling
     param and **not** a CONTRACT violation;
   - **is** honored for **ordering only** *if* its source is present.

   This aligns the three sites, which currently disagree: `closure.py:48` includes it (ordering — keep);
   `mock_tools.py:192` already ignores it for call success —
   > `if param not in required: continue  # optional sourced args are not synthesized` (`mock_tools.py:192-193`);
   but `invariants.py:91` flags **any** param'd dep as dangling regardless of `required` —
   > `if dep.param is not None:` … `dangling.add(token)` (`src/mcp_router_eval/contract_layer/invariants.py:91-93`).
   Under this ADR the completeness/attribution rule **must not require an optional source**; only
   required-argument sources make a closure incomplete. (Implementation of the `invariants.py` alignment
   is a later step — see Scope.)

## Consequences

- **Fixes the `val_completion ≈ 0` artifact.** Completion will be measured against tools the task truly
  needs, so the metric discriminates routers instead of judging all of them against unreachable label
  noise (BM25 0.098 → 0.877 on the spine, `docs/adr/0004-…:83`).
- **q240 completion required-set = `[validate_email, audible_account_login, download_audible_book]`
  (3 tools)** — now **consistent** with `docs/completion-scoring-examples.md`'s worked example, reached
  via the **required-argument rule**, *not* via the false "`TOOL_*`-only" claim.
- **Ordering stays correct.** Because expansion stays variant B, an optional source that is selected is
  still topologically ordered (ADR 0012); nothing about run order changes.
- **The GNN "collapse" was confounded with this bug.** The full-golden required-set drove every router's
  completion to ~0; the real GNN retrieval signal is `val_map` (rgcn **0.444** > gat **0.431** > sage
  **0.389**, `best_configs.json`). Any query-agnostic-collapse investigation must be re-run against the
  variant-A completion metric and read on retrieval, not on the pre-fix completion numbers.
- **`best_configs.json` is invalidated.** It was selected by `select_best` primary-sorting on the broken
  `val_completion` (`src/mcp_router_eval/eval/tuning.py:109`); the grid must be re-run after the harness
  change (a later step — see Scope). ADR 0029's "best by validation `completion_rate`" now means the
  variant-A rate.

## Transparency (carries forward the ADR-0004 amendment's promise)

The evaluation reports **both**: `completion_rate` against the variant-A required-set as the **PRIMARY**
(north-star) metric, **and** `completion_rate` against the full `golden_function_names` as a **SECONDARY**
reported number — so the choice of required-set is auditable, not hidden.

## Scope (what this ADR does and does not do)

- **Does:** fix the required-set definition, decouple it from the ordering closure, and pin the
  optional-param semantics. Records that the `get_system_language` edge-type error in
  `docs/completion-scoring-examples.md` (and the mirrored text at `docs/adr/0004-…:62`) must be corrected
  to the required-argument rule.
- **Does not (later steps):** edit `docs/completion-scoring-examples.md`; change the harness to pass the
  variant-A required-set (`harness.py`), or align `invariants.py:91`; de-circularize the router tests;
  re-run the grid search. No code changes in this ADR.

## Alternatives considered

- **Variant B — full `PARAMETER_*` closure incl. optional-argument sources** — rejected: it contradicts
  the amendment's own 3-tool q240 example and would require, for completion, tools the task structurally
  completes *without* (optional args have defaults). It also over-counts `get_system_language` into 399
  gold sets as "required."
- **Full golden set (status quo in `harness.py`)** — rejected: the diagnosed bug — unreachable `TOOL_*`
  label noise drives `completion ≈ 0` for every router (BM25 **0.098**, `get_wifi_status` missed
  **184/235**).
- **Required-set = ordering-set (coupled)** — rejected: conflates "needed to complete" with "ordered if
  present"; forcing them equal reintroduces either variant B's over-inclusion (if ordering wins) or
  breaks the ordering guarantee (if completion wins). Decoupling keeps both correct.
