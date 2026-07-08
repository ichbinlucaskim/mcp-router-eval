# 0028 — Evaluation metrics: standard retrieval + structural completion + north-star transfer loss, sliced by dependency depth with deterministic failure attribution

## Status

Accepted

## Context

The north star is **end-to-end structural completion** and the **retrieval→completion transfer loss**
on **deep-dependency** queries — *not* retrieval accuracy alone (ADR 0004/0005). Before building the
evaluation harness we fix what it measures. Two verified sources frame the design:

- **GRETEL** formalizes the **semantic-functional gap**: tool retrieval by semantic similarity "fails
  to capture functional viability" — a semantically relevant tool is often functionally inoperative.
  The gap is written exactly as **`P(functional | semantic) ≪ P(functional)`** ("the probability that a
  semantically relevant tool is functionally viable" is much lower than baseline), and GRETEL finds
  **85% of top-5 semantically retrieved candidates have functional flaws**, decomposed into
  **Parameter Mismatch (42%), Semantic Mismatch (25%), Execution Failure (18%)**
  ([Wu, Guo, Liang & Li, "GRETEL", arXiv:2510.17843](https://arxiv.org/abs/2510.17843)). This is exactly
  our thesis: retrieval success is *necessary but not sufficient* for completion.
- **MCP-Bench** provides a **multi-faceted, rule-based** evaluation of tool-using agents covering
  **tool-level schema understanding and usage, trajectory-level planning, and task completion**, scored
  from execution traces ([Wang, Chang, Patel et al., "MCP-Bench", arXiv:2508.20453](https://arxiv.org/abs/2508.20453)).
  We take its principle — completion should be a **rule-based, decomposable** verdict over tool use,
  not one opaque pass/fail.

**Honest limitation.** GRETEL and MCP-Bench are agent/tool benchmarks with *executing* tools; we reuse
their **concepts** — the semantic-functional gap formalization, and rule-based decomposable completion —
on the **ToolLinkOS dependency graph** with our **structural** completion proxy (ADR 0004). The concepts
transfer; the setup is ours. (Verification note: GRETEL's `P(functional|semantic)` formalization and its
three failure categories were confirmed verbatim this session; MCP-Bench's *abstract-level* multi-faceted
rule-based framing was confirmed, but the specific completion sub-dimension **names** below are **our**
decomposition mapped onto its framework, not verbatim from the paper.)

## Decision

Three metric groups, all computed on the **TEST split** (ADR 0024), each **sliced by closure-depth
buckets** (ADR 0005: shallow 2–3 vs deep ≥6):

**1. Standard retrieval (established practice — implemented, not cited).**
`mAP@k`, `recall@k`, `nDCG@k`. `k` configurable; **`k = 10`** as the default for comparison with the
ToolLinkOS paper. Ranking quality only.

**2. Structural completion (rule-based, decomposable — MCP-Bench principle).**
`completion_rate` per ADR 0004 (required-set = the variant-A **required-argument `PARAMETER_*` closure**,
[ADR 0030](0030-completion-required-set.md), *not* the full `golden_function_names`) = correct tool
**set** + dependency **order** + **type-valid** args. Also
report **component sub-rates** so completion is decomposable rather than a single opaque pass/fail —
aligned to MCP-Bench's rule-based tool-usage evaluation:
  - **name validity** — the invoked tools cover the completion **required-set** (the variant-A
    required-argument `PARAMETER_*` closure, ADR 0030). A gold tool attached only by `TOOL_*` edges — or
    via an *optional* `PARAMETER_*` argument — is neither **required** (it is outside the variant-A
    required-set) nor **spurious** (it may legitimately appear); name validity is judged solely against
    the variant-A required-set, not the full `golden_function_names`;
  - **schema / type adherence** — call args are type-valid against the built JSON Schema (ADR 0014);
  - **dependency compliance** — `PARAMETER_*` order respected + no unsourced dependency arg (ADR 0012/0016);
  - **runtime success** — the mock runner reports every call `ok` (ADR 0015).

**3. North-star transfer loss (GRETEL's gap; our formula).**
  - **Primary (conditional):** `transfer_loss = 1 − P(completion success | retrieval success)` — of the
    queries whose retrieval succeeded (gold recalled at `k`), the fraction that then **fail** structural
    completion. This directly operationalizes GRETEL's `P(functional | semantic)`: it isolates *what is
    lost in the retrieval→completion transfer*, conditioning on retrieval having succeeded.
  - **Secondary (descriptive):** the difference form `retrieval_metric − completion_metric` as an
    interpretable complement (a level gap). **Both are reported; the primary is the conditional.**

**Failure attribution.** Every failed query is attributed **deterministically** to
**ROUTING / CONTRACT / EXECUTION** by our contract layer (upstream-wins, ADR 0018 §3.4 / attribution.py).
This is our **deterministic, reproducible analogue** to GRETEL's LLM-decomposed failure categories —
judge-independent by construction. Report the attribution breakdown **per router** and **per depth slice**.

## Consequences

- All five routers **{BM25, NaiveRAG, HybridRAG, Traversal, GNN}** are compared on the same three groups,
  sliced by dependency depth, with **reproducible** failure attribution.
- The **deep-slice `transfer_loss`** is the headline number the thesis turns on (does the dependency-aware
  GNN reduce it where dense retrieval assumes homophily?).
- Completion is **decomposable** (MCP-Bench-style sub-rates), so a low score points at a cause (bad tool
  set vs bad order vs bad args) rather than an opaque fail.
- **`homophily_local`** (ADR 0027, GNN-only) can be **correlated with `transfer_loss`** to test the
  heterophily hypothesis — low local homophily should predict high transfer loss where the GNN helps.

## Alternatives considered

- **`transfer_loss` as a plain difference only** — rejected as primary: a difference measures a *level*
  gap, not the *conditional* transfer GRETEL formalizes; kept as the secondary descriptive form.
- **Retrieval-only evaluation (mAP alone)** — rejected: that is exactly what the north star argues
  against (retrieval success ≠ task completion, ADR 0004; GRETEL's 85%-flawed finding).
- **Single opaque completion pass/fail** — rejected: not decomposable; the MCP-Bench-style component
  sub-rates are chosen so a failure is diagnosable.

## Sources

- Wu, Guo, Liang & Li, "GRETEL: A Goal-driven Retrieval and Execution-based Trial Framework for LLM Tool
  Selection" (arXiv:2510.17843) — semantic-functional gap `P(functional|semantic) ≪ P(functional)`; 85%
  of top-5 semantic candidates functionally flawed; failure categories Parameter Mismatch / Semantic
  Mismatch / Execution Failure: <https://arxiv.org/abs/2510.17843>
- Wang, Chang, Patel et al., "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks
  via MCP Servers" (arXiv:2508.20453) — multi-faceted, rule-based tool-use evaluation (schema
  understanding/usage, planning, task completion): <https://arxiv.org/abs/2508.20453>

*(Only these two papers are cited; standard retrieval metrics — mAP / recall / nDCG — are established
practice and implemented without citation.)*

## Amendment 2026-07-05 — `retrieval_success` conditions on the variant-A required-set (align `transfer_loss` with ADR-0030)

This refines **only** the `retrieval_success` condition inside the north-star `transfer_loss` (metric
group 3). The three metric groups, the conditional-vs-difference forms, and the depth slicing are all
**unchanged**. No external papers are added; this is a keystone-consistency fix grounded in ADR-0030 and
our measured full-eval output.

### Context — the north-star headline came back empty (a target mismatch)

`transfer_loss` (conditional, primary) `= 1 − P(completion | retrieval success)`. Today
`retrieval_success` is `recall@10 ≥ 1.0` of the **full `golden_function_names`**
(`src/mcp_router_eval/eval/metrics.py:174-175`; threshold `EvalConfig.threshold = 1.0`,
`eval/harness.py:51`). But **completion** is scored on the **variant-A required-set** (ADR-0030) — so
`transfer_loss = 1 − P(completion_variantA | retrieval_of_FULL_gold)` mixes **two different targets**.

Measured on the full eval (`data/processed/eval/full_eval.json`, 236 test queries, 5 seeds):

- The **deep** slice (closure-depth ≥ 6, `eval/slices.py:30`) holds **n = 140** queries — the *largest*
  slice, so this is **not** sparsity.
- **Zero** deep queries reach `recall@10 = 1.0` of the full gold set (deep `recall@10` ≤ **0.44** across
  routers; e.g. NaiveRAG 0.219, traversal 0.443), so the conditional denominator is **empty** →
  `nan`/`None` (`transfer_loss_conditional`'s empty-denominator guard) for **every** router at the deep
  headline (plus all of BM25, whose recall is low at every slice).
- The tell: **NaiveRAG's deep completion is 1.000, yet its deep `transfer_loss` is `None`** — undefined
  *exactly where completion is perfectly defined*. The **difference** form is populated but **negative**
  for all routers (e.g. traversal overall −0.366) because it computes `recall@10(full gold) −
  completion(variant-A)` — the small spine completes more than the large full gold is recalled — the
  same mismatch surfacing differently.

This is the **same full-gold-vs-spine fix already made for completion (ADR-0030)**, not yet applied to
`transfer_loss`'s retrieval condition.

### Decision

Condition `retrieval_success` on recalling the **variant-A required-set** — the required-argument
`PARAMETER_*` spine (ADR-0030) — i.e. **`recall@k` of the required-set ≥ threshold**, the **same tools
completion needs**, not the full label-noisy gold. Then:

> **`transfer_loss = 1 − P(completion_variantA | retrieval of the variant-A required tools)`**

— *"of the queries whose **required** tools were retrieved, how many still fail completion."* Both the
**conditioning** and the **outcome** now use the variant-A target. The **difference** form is likewise
recomputed against the spine (recall of the required-set − completion), so it is target-consistent and no
longer sign-flipped.

### Why this is not metric-gaming (stated explicitly)

Identical justification to ADR-0030: the excluded full-gold tools are **query-irrelevant label noise**
(system tools appear in ~80% of gold sets — `get_wifi_status` gold in 887/1098 train queries; missed
184/235 in validation, `docs/adr/0004-…:78-79`). Requiring their *retrieval* measures label noise, not
whether the query's **actual** tools were found. Conditioning on the spine is the **target-consistent**
choice, and it is applied **uniformly to all seven routers** (baselines + GNN), so it favors no method.
For transparency the evaluation reports **both**: the **primary** spine-conditioned `transfer_loss` and,
where defined, the **secondary** full-gold-conditioned number — so the choice of retrieval target is
auditable, exactly as ADR-0030 does for completion.

### Expected effect (from the diagnosis — to verify on re-run)

- **NaiveRAG deep → `transfer_loss ≈ 0`** (it recalls the spine *and* completes: deep completion 1.000).
- **GNN deep → `transfer_loss ≈ 1.0`** (retrieval of the spine can succeed — deep `map@10 ≈ 0.42` — but
  completion is 0.000): retrieval succeeds, completion does not follow.

That contrast — high transfer loss where a dependency-aware method's retrieval does not convert to
completion — is exactly the north-star signal the metric was designed to capture. The headline
**populates** instead of being uniformly `n/a`.

### Threshold

Keep the **threshold semantics** (recall of the required-set ≥ threshold), noting the set is now the
**spine**, not the full gold. The crude alternatives are **rejected** because they keep the mismatch:

- *lower the threshold to ~0.5 against the full gold* — still conditions on label-noise tools;
- *fall back to the difference form for the deep headline* — the difference keeps the same full-gold-vs-
  spine mismatch (and stays negative).

Required-set **alignment** is the real fix, not a looser threshold.

### Scope

Changes **only** `retrieval_success`'s target (the set it measures `recall@k` against) and therefore how
`transfer_loss` (both conditional and difference) conditions — a metric/eval-layer change to be
implemented as the **next step**. Completion, ordering, type-validity, the depth slices, and the
retrieval metrics themselves (`mAP` / `recall` / `nDCG` reported against the full gold, established
practice) are **unchanged**.
