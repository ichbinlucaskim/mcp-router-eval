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
`completion_rate` per ADR 0004 = correct tool **set** + dependency **order** + **type-valid** args. Also
report **component sub-rates** so completion is decomposable rather than a single opaque pass/fail —
aligned to MCP-Bench's rule-based tool-usage evaluation:
  - **name validity** — the invoked tools are the required set (no missing/spurious tool);
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
