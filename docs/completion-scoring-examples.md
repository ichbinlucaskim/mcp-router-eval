# Completion scoring — worked examples (structural proxy)

How **structural-proxy completion** (ADR 0004) is scored, using an execution order derived by
**topological sort of the PARAMETER dependency sub-graph** (ADR 0012, as refined by the cycle finding
in `docs/data-inspection-toollinkos.md` §Dependencies). These three scenarios are written to serve
**directly as test cases** for the completion + attribution logic.

> **Completion here is STRUCTURAL, not semantic.** We do **not** check that the audiobook actually
> downloaded or that the email is truly valid. `completed = true` iff: (1) the selected set contains
> the full dependency closure of the gold tools, (2) tool calls occur in a **topologically valid**
> order (every param-source runs before its consumer), and (3) every call's args are **type-valid**
> against the tool's normalized JSON-Schema. Nothing about real-world effect is asserted.

## Grounding (real data — instance `q240`)

- **Query:** *"I just finished a book and want to dive into 'The Great Gatsby.' Can you download it
  for me from Audible?"*
- **`main_golden_function_name`:** `download_audible_book`
- **`golden_function_names` (as stored, main-first):**
  `[download_audible_book, set_low_battery_mode_status, get_low_battery_mode_status, get_system_language, get_battery_status, set_wifi_status, get_wifi_status, audible_account_login, validate_email]`

The **param-source spine** (the part that carries required-argument dependencies) is a real 3-hop chain:

```
validate_email  ──produces validated 'email'──▶  audible_account_login  ──produces 'session_id'──▶  download_audible_book
```

Real edges (all `PARAMETER_DIRECTLY_DEPENDS_ON`, from the data):
- `audible_account_login`  depends on  `validate_email`   (param `email`)
- `download_audible_book`  depends on  `audible_account_login`  (param `session_id`)

`validate_email` is a `core` tool with **no** dependencies. The remaining gold tools are all correctly
dropped from the completion required-set, but for **two different reasons** (do not conflate them):

- `get_wifi_status`, `get_battery_status`, and the `*_low_battery_*` / `*_wifi_*` get/set pairs are
  attached by `TOOL_*` edges only; they carry **no required-arg dependency** and are **excluded from
  ordering** (the `TOOL_INDIRECTLY` get/set pairs are exactly what make the full-edge graph cyclic —
  see the feasibility report).
- `get_system_language` is **not** `TOOL_*`-only. It is a `PARAMETER_*` source (`param_indirect`) of
  `audible_account_login`'s argument `language`, but that argument is **optional**
  (`audible_account_login.required = [email, password]` — verified in `data/processed/tools.jsonl`). So
  under the **required-argument rule (ADR-0030, variant A)** its source is excluded from the completion
  **required-set** *because the sourced arg is optional* — **not** because the edge is `TOOL_*`. (If
  `get_system_language` is selected, its `PARAMETER_*` order is still honored — ordering closure, ADR
  0012/0013; it is simply not *required* to be present.)

Ordering uses the PARAMETER sub-graph, which is a DAG.

**Derived topo-sorted execution order (deps first, main last):**
```
validate_email  →  audible_account_login  →  download_audible_book
```
Note this is the **reverse** of the stored main-first order — the main tool runs **last**, after its
dependencies (ADR 0012).

---

## Scenario A — perfect → `completed=true`, `blame=none`

```yaml
query_id: q240
query_text: "... download 'The Great Gatsby' from Audible ..."
gold_closure: [validate_email, audible_account_login, download_audible_book]   # required-arg PARAMETER_* closure (ADR-0030 variant A); get_system_language excluded as an OPTIONAL-arg source, not as TOOL_*
exec_order:   [validate_email, audible_account_login, download_audible_book]   # topo (deps first)

router.selected_tools: [download_audible_book, audible_account_login, validate_email]

contract.invariant_report:
  closure_complete: true
  dangling_params: []
  violations: []
gate_decision: pass

executor.call_trace:                       # ok=true, args type-valid, topo order respected
  - {tool_id: validate_email,        args: {email: "<str>"},                   ok: true}
  - {tool_id: audible_account_login, args: {email: "<str>", password: "<str>"}, ok: true}
  - {tool_id: download_audible_book, args: {session_id: "<str>", book_name: "<str>", language: "<str>"}, ok: true}

verdict.completed: true

attribution:
  outcome: success
  blame: none
  evidence: "Closure complete; calls in topo-valid order; all args type-valid."
```

---

## Scenario B — low-homophily dependency MISSED → `completed=false`, `blame=CONTRACT` *(thesis-critical)*

The query is about *downloading a book*; `validate_email` is a generic email-validation utility that
is **semantically far** from the query and from the main tool — the kind of dependency dense vector
retrieval plausibly **misses** (see feasibility report: 809/1,496 dependency pairs have **zero**
lexical overlap; `validate_email → audible_account_login` shares only the token *email*). The router
retrieves the two "obvious" tools but drops the low-homophily param-source.

```yaml
query_id: q240-B
gold_closure: [validate_email, audible_account_login, download_audible_book]   # required-arg PARAMETER_* closure (ADR-0030)
exec_order:   [validate_email, audible_account_login, download_audible_book]

router.selected_tools: [download_audible_book, audible_account_login]   # validate_email MISSED

contract.invariant_report:
  closure_complete: false
  dangling_params: ["audible_account_login.email"]   # param 'email' has no sourcing tool in the set
  violations: ["missing param-source dependency: validate_email (sources audible_account_login.email)"]
gate_decision: pass        # (a completion-tuned gate MAY fallback here; shown as 'pass' to isolate the contract)

executor.call_trace:
  - {tool_id: audible_account_login, args: {email: "<UNSOURCED>", ...}, ok: false, error: "required arg 'email' not sourced"}

verdict.completed: false

attribution:
  outcome: failure
  blame: CONTRACT
  evidence: "audible_account_login.email is a dangling param: its param-source validate_email
             (PARAMETER_DIRECTLY_DEPENDS_ON, param=email) is a dependency of a SELECTED tool but was
             not pulled into the closure. Closure-incompleteness/dangling-param ⇒ CONTRACT."
```

> **Attribution nuance (make this rule explicit in the logic):** the missed tool is a *dependency of
> a selected tool*, so responsibility lies with **closure expansion** (CONTRACT), not with the router.
> Contrast with **ROUTING** blame, which applies only when a required **primary** tool never appears
> in the ranking at all (nothing selected points to it as a dependency). This primary-vs-dependency
> split is what makes the deterministic §3.4 rule unambiguous and must be encoded in the attribution
> implementation.

---

## Scenario C — correct set, ORDER violated → `completed=false`, `blame=EXECUTION`

The full closure is selected and the contract passes, but the executor calls the main tool **before**
its param-sources have run, so a required argument produced upstream is unavailable.

```yaml
query_id: q240-C
gold_closure: [validate_email, audible_account_login, download_audible_book]   # required-arg PARAMETER_* closure (ADR-0030)
exec_order:   [validate_email, audible_account_login, download_audible_book]   # expected

router.selected_tools: [download_audible_book, audible_account_login, validate_email]

contract.invariant_report:
  closure_complete: true
  dangling_params: []
  violations: []
gate_decision: pass

executor.call_trace:                       # ORDER VIOLATED: main called first
  - {tool_id: download_audible_book, args: {session_id: "<UNSOURCED>", ...}, ok: false, error: "arg 'session_id' not yet produced (audible_account_login has not run)"}
  - {tool_id: audible_account_login, args: {email: "<str>", ...}, ok: true}
  - {tool_id: validate_email,        args: {email: "<str>"},       ok: true}

verdict.completed: false

attribution:
  outcome: failure
  blame: EXECUTION
  evidence: "All required tools present & closure valid, but download_audible_book was invoked at
             position 1 before its param-source audible_account_login (session_id). A topo-order
             violation with tools present & valid ⇒ EXECUTION."
```

---

## Summary (test-case matrix)

| Scenario | closure_complete | order respected | args valid | completed | blame |
| --- | --- | --- | --- | --- | --- |
| A perfect | ✅ | ✅ | ✅ | **true** | none |
| B missed low-homophily dep | ❌ (dangling `email`) | — | — | **false** | CONTRACT |
| C order violated | ✅ | ❌ | ❌ (unsourced arg) | **false** | EXECUTION |

All three are backed by real tools/edges from ToolLinkOS `q240`; see `docs/feasibility-completion.md`
for the evidence and the cycle finding that fixes the ordering sub-graph.
