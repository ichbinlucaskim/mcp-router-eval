# Feasibility checkup — completion scoring against real ToolLinkOS data

Read-only verification of every claim the completion-scoring examples rest on
(`docs/completion-scoring-examples.md`). Data: `data/raw/` @ commit `b630b98`
(`docs/data-inspection-toollinkos.md`). Tags: **[CONFIRMED] / [ADJUSTED] / [BLOCKER]**.

## ⚠ HEADLINE — the full dependency graph is NOT a DAG **[BLOCKER → resolved by ADJUSTMENT]**

Building the directed graph *u → v* ("u depends on v") over **all** `depends_on` edges:

- **Full graph is cyclic:** 573 nodes, **485 in cycles**; **1,374 / 1,569 gold sets contain a cycle.**
- A topo-sort over all edge types is therefore **impossible** — this would break ADR 0012's execution
  order and the completion verdict if left unaddressed.

**Root cause — `TOOL_INDIRECTLY` get/set pairs.** Cycle count by edge-type subset:

| Edge subset | edges | cyclic nodes | DAG? | gold sets w/ cycle |
| --- | ---: | ---: | :---: | ---: |
| ALL 4 types | 1496 | 485 | ❌ | 1374 |
| `TOOL` (direct+indirect) | 851 | 426 | ❌ | 1374 |
| **`TOOL_DIRECTLY` only** | 676 | 0 | ✅ | 0 |
| **`PARAMETER` (direct+indirect)** | 645 | 0 | ✅ | 0 |
| **`PARAMETER_DIRECTLY` only** | 406 | 0 | ✅ | 0 |

> **Edge counts above are PROCESSED-data values** (`data/processed/tools.jsonl`, verified this session:
> `param_direct` 406, `param_indirect` 239, `PARAMETER` 645, `tool_direct` 676, `tool_indirect` 175,
> `TOOL` 851, all-4 **1496** — consistent with the "1,496 dependency pairs" in Check 2 below). The
> raw@`b630b98` figures this report was first written against are **1 lower per edge-type**
> (`PARAMETER` 644, `TOOL` 850, `param_direct` 405, `tool_direct` 675; all-4 1494); the +1-each drift is
> from malformed-row normalization during processing. Cycle / gold-set-cycle counts are unchanged.

- **21 mutual 2-cycles** (`u ⇄ v`); **12 are `get_`/`set_` pairs**, e.g.
  `get_location_service_status ⇄ set_location_service_status`,
  `get_airplane_mode_status ⇄ set_airplane_mode_status`,
  `get_cellular_service_status ⇄ set_cellular_service_status`. The rest are data-derivation mutuals
  (e.g. `get_gdp_per_capita_by_country ⇄ get_total_gdp_by_country`). These are **co-occurrence /
  soft relations, not execution preconditions.**

**[ADJUSTED] Resolution:** derive execution order from the **PARAMETER dependency sub-graph**, which
is a **DAG** (0 cycles, 0 gold-set cycles). `PARAMETER_*_DEPENDS_ON` = "tool A needs a parameter value
produced by tool B" — the *actual* run precondition. `TOOL_*_DEPENDS_ON` (especially `INDIRECTLY`)
carry no required-arg semantics and must be **excluded from ordering**. This narrows ADR 0012's
"topological sort of the dependency graph" to the PARAMETER sub-graph, and the ADR 0011 validation
hook must **assert acyclicity of that ordering sub-graph and reject any cycle loudly**.

## Check 1 — real dependency chain grounding the examples **[CONFIRMED]**

- **531 instances** contain a ≥3-node acyclic chain under `PARAMETER_DIRECTLY`.
- Chosen example (instance **q240**, query *"…download 'The Great Gatsby' from Audible"*):
  `validate_email → audible_account_login → download_audible_book`, all real
  `PARAMETER_DIRECTLY_DEPENDS_ON` edges (params `email`, `session_id`). Part 1 is built on this real
  chain. (The "restaurant/location" instances exist — 150 with ≥3 chains — but their gold sets are
  cyclic under TOOL edges, so the cleaner Audible chain was chosen.)

## Check 2 — low-homophily (dissimilar) param dependencies exist **[CONFIRMED, proxy only]**

Structural existence of "main tool → semantically distant dependency" pairs is clear:

- Over all 1,496 dependency pairs (lexical-Jaccard proxy on name+description, stopworded):
  **min 0.00, mean 0.08, and 809 pairs (54%) have ZERO lexical overlap.**
- Concrete zero-overlap examples:
  `download_audible_book → validate_email` (shares only *email*);
  `add_google_calendar_reminder → get_current_date`;
  `ali_express_checkout_cart → get_wifi_status`.
- These are exactly the dependencies dense retrieval would plausibly miss (the RQ2/gate thesis case).

> **[deferred]** This is a **crude lexical proxy**. A true embedding-cosine homophily measure needs
> the sentence-transformer/ada-002 model — **not installed this session** (per constraints). Deferred
> to the embedding week; only *structural existence* is confirmed here.

## Check 3 — cycle check — see HEADLINE above **[BLOCKER→ADJUSTED]**

Whole graph acyclic: **NO.** All 1,569 gold sets individually acyclic: **NO (1,374 cyclic).**
Under the PARAMETER sub-graph: **YES (all acyclic).**

## Check 4 — golden order is main-first; topo run order differs **[CONFIRMED]**

`main_golden_function_name` is at index 0 in **all 1,569** instances (last in 0). Example q240:

- **Stored (main-first):**
  `[download_audible_book, set_low_battery_mode_status, get_low_battery_mode_status, get_system_language, get_battery_status, set_wifi_status, get_wifi_status, audible_account_login, validate_email]`
- **Topo RUN order (PARAMETER DAG, deps first):**
  `[get_battery_status, get_low_battery_mode_status, get_system_language, get_wifi_status, set_low_battery_mode_status, set_wifi_status, validate_email, audible_account_login, download_audible_book]`
- **`download_audible_book` runs at position 9/9** — dead last, vs stored position 1. Confirms the
  stored order is **not** runnable order.

## Consequences to record (recommended follow-ups — not done this session)

1. **Update ADR 0012** to specify: execution order = topo-sort of the **PARAMETER** dependency
   sub-graph (DAG); `TOOL_*` edges are excluded from ordering (they are cyclic and non-precondition).
2. **Update ADR 0011** validation hook to assert the ordering sub-graph is acyclic and **fail loudly**
   on any cycle; the full 4-type graph is expected to be cyclic and that is fine for the *router*
   (message passing tolerates cycles) but not for the *ordering* used by completion.
3. Keep `TOOL_*` edges for the **router/graph** (they are real structure); restrict only the
   **ordering** logic to PARAMETER edges.
