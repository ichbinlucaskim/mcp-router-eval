# Findings — the GNN collapse is a message-passing *structure* mismatch, not an architecture choice

**Status:** Negative result, documented. Establishes the xfail of
`tests/test_gnn_router.py::test_full_pipeline_integration` (ADR-0030 de-circularization) as an **honest,
config-invariant** outcome — not under-tuning. All numbers below are cited to measured data (this
session's read-only diagnosis/probes) or to an ADR / data file (`file:line`). The only external paper
cited here is **DPAA** (arXiv:2605.11145), verified this session; logQ and GCNII are referenced through
their ADRs.

---

## Thesis

**The GNN's failure is not an architecture-selection problem.** All three backbones (R-GCN, GAT,
GraphSAGE — ADR 0010) and the entire searched hyperparameter space —
`hidden × dropout × heads × τ × lr × weight_decay` (`eval/tuning.py:45-52`), plus the logQ strength
`α ∈ {0, 0.5, 1}` (ADR 0031; `eval/tuning.py` `ALPHAS`) and the initial-residual strength
`α_res ∈ {0, 0.1, 0.5, 0.8}` (ADR-0025 amendment) — collapse to the **same** variant-A completion
**0.000** on the validation split.

The one thing every one of those configs shares is the **message-passing paradigm**. That is the cause.
Message passing assumes **homophily** — connected nodes are similar, so mixing a node with its neighbors
sharpens its representation. On *this* graph that assumption is false, and learning collapses to a
frequency shortcut.

## Why this graph breaks message passing

Three measured characteristics of the ToolLinkOS dependency graph:

1. **Heterophilic dependencies.** A tool and its dependency are typically *semantically unrelated*:
   lexical-Jaccard over all 1,496 dependency pairs is **mean 0.08, with 809 pairs (54%) at zero overlap**
   (`docs/feasibility-completion.md:52-57`) — e.g. `download_audible_book`'s dependencies include
   `get_wifi_status` (a `TOOL_*` association with no semantic relation) and `validate_email` (shares only
   the token *email*). Mixing such neighbors **pollutes** a node's representation rather than sharpening
   it.
2. **Ubiquitous hubs.** `get_wifi_status` is a dependency-source of **371** tools (in-degree 371,
   measured) while the query-specific main tool `download_audible_book` has **in-degree 0** (measured).
   Aggregation lets the hub's signal dominate the embedding space (the DPAA mechanism, below).
3. **Frequency-biased labels.** The high-frequency system tools are gold in most queries:
   `get_wifi_status` is gold in **887 / 1,098** train queries (**80.8%**, measured; cf. the ADR-0004
   amendment's "missed in **184/235**" validation figure, `docs/adr/0004-…:78-79`). So the training
   objective is *rewarded* for ranking them high regardless of the query.

## Mechanism — two compounding forces

- **Force 1 — the loss rewards a frequency shortcut.** Masked InfoNCE (ADR 0026,
  `gnn_train.py:masked_infonce`) minimizes loss most cheaply by **always ranking the ~80%-gold system
  tools high**, rather than learning the per-query main tool. The model learns *"rank frequent tools,"*
  not *"understand the query."*
- **Force 2 — message passing amplifies it.** The high-in-degree hub (`get_wifi_status`, in-degree 371)
  spreads its signal across the graph during aggregation, pulling node embeddings into a
  popularity-dominated region and **reinforcing** the frequency shortcut — the message-passing
  popularity-amplification mechanism formalized by **DPAA** ("skewed distributions + repeated message
  passing amplify popular items into a popularity-dominated embedding region and inflate their scores",
  arXiv:2605.11145, verified this session).

## Controlled evidence — message passing is the meaningful differing variable

The crux. NaiveRAG and the GNN share the same BGE node features (ADR 0003/0020) and the same late-cosine
scoring (ADR 0022 amendment); the difference is message passing. To make that **clean** (a fairness audit
noted the *default* GNN also adds a learned two-tower projection), we ran an **isolation probe**: a GNN
with **no learned projection** (`proj_dim=None`, node tower in the raw **384-d BGE space**, cosine against
the raw BGE query) — so its **only** difference from NaiveRAG is message passing.

Measured on the validation split (seed 0, 235 queries; short-trained GNN):

| router | vs NaiveRAG | variant-A completion | main∈top-10 | main-tool median rank | `corr(gold_freq, rank)` | node pairwise-cos |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **NaiveRAG** (BGE + cosine) | — | **0.970** | **227/235** | **0** | **−0.03** | **0.501** |
| Traversal | no MP (dep-expansion) | 0.898 | — | 0 | +0.39 | — |
| BM25 | no MP | 0.762 | — | 1 | +0.19 | — |
| **GNN — MP only** (`proj=None`) | **+ message passing only** | **0.000** | **0/235** | ~272–296 | **−0.247** | **0.862** |
| GNN — MP + projection (default) | + MP + learned projection | 0.000 | 0/235 | ~272–296 | −0.244 | 0.883 |

- **The MP-only control collapses identically to the full GNN.** With *no* learned projection and a raw-BGE
  query — the only difference from NaiveRAG being message passing — completion is **0.000**, the main tool
  is **never** retrieved (**0/235**), and ranking is frequency-driven (`corr −0.247`). Adding the learned
  two-tower projection changes essentially nothing (node pairwise cosine **0.862 → 0.883**; every other
  number identical). So **the meaningful difference from NaiveRAG is message passing; the projection is a
  negligible compounding factor, not a co-driver.**
- **Message passing over-smooths the nodes *by itself*.** Node-embedding mean pairwise cosine:
  **NaiveRAG 0.501 → MP-only 0.862 → MP+projection 0.883** (1.0 = all identical). MP homogenizes the node
  embeddings with no projection involved, so the query cosine can barely discriminate tools — the concrete
  mechanism behind the frequency-ranking collapse (DPAA's amplification, cited above).
- **Uniform burial, not a partial effect.** The GNN gets the query's main tool into top-10 for **0/235**
  queries vs NaiveRAG's **227/235**; `corr(gold_freq, mean_rank) ≈ −0.24` (GNN) vs **−0.03** (NaiveRAG).
  The features are good; message passing degrades them.

**Over-determination (honest bound — no overclaim).** The collapse is *over-determined*: message passing
**alone** suffices (the isolation probe above), **and** the frequency-trained two-tower head **alone**
suffices (the `α_res=1` probe — ADR-0025 amendment — collapsed to 0.000 with message passing bypassed).
Foregrounding message passing is correct and data-supported — the claim is **"message passing by itself is
sufficient for the collapse,"** *not* "message passing is the only possible cause."

## Why the standard remedies did not help (we understand the failure)

- **logQ correction (ADR 0031)** targets a **negative-sampling** popularity bias. But our driver is
  **Force 1** — the loss rewarding *genuinely-frequent gold **positives*** — not a negative-sampling
  artifact. The diagnosis **refuted** the negative-sampling hypothesis directly: the ADR-0023
  false-negative filter removes **zero** additional negatives for the frequent tools (`neg_on == neg_off`;
  dep-but-not-gold exclusion = **0.0%**, measured). So logQ has little bias to correct, and raising `α`
  merely **suppressed true positives**, dropping `val_map` monotonically (**0.386 → 0.376 → 0.367** across
  `α ∈ {0, 0.5, 1}`, measured in the re-tune) while completion stayed **0.000**.
- **GCNII initial residual (ADR-0025 amendment)** preserves the raw features against message passing
  (targets **Force 2**). But even at `α_res = 0.8` — where the node tower is ≈ the projected raw features —
  variant-A completion stays **0.000** (measured probe: `α_res ∈ {0, 0.1, 0.5, 0.8}` → completion
  `{0.000, 0.000, 0.000, 0.000}`, main-tool median `{276, 284, 296, 294}`, `corr {−0.234, −0.234, −0.228,
  −0.231}`). **Fixing Force 2 alone doesn't help, because Force 1 (the loss) survives feature
  preservation** — the trainable two-tower projections + the frequency-dominated InfoNCE re-introduce the
  shortcut.

Together: no tested knob — across three backbones, the full architecture/optimizer grid, `α`, and
`α_res` — ever moved variant-A completion off **0.000**. The collapse is **config-invariant**.

## Evaluation fairness — is the collapse a real limitation, or our design's fault?

Audited directly (git-clean, measured) to pre-empt *"isn't this your evaluation's fault?"*. Verdict:
**the design is fair — the collapse is a genuine message-passing limitation, not an artifact.**

- **The hub is the DATA's, not our construction.** `get_wifi_status` in-degree matches the raw ToolLinkOS
  records **exactly**: raw **371** = processed **371** = graph **371**; total edges **1496** unchanged
  through the whole pipeline (`graph_build.py` maps each `depends_on` record 1:1 — no self-loops, no
  inflation).
- **The control is (near-)symmetric.** GNN and NaiveRAG use the same BGE features (+`is_core`, extra not
  missing), the same per-tower L2, and the same plain-cosine inference (dropout **off** at serving); the
  isolation probe above removes even the projection asymmetry, and MP-alone still collapses.
- **The GNN is query-conditioned BY CONSTRUCTION.** It has a `query_proj` and the InfoNCE loss scores
  `q @ node.T` per query (`gnn_train.py`), so it *can* rank per-query. The query-agnostic behavior —
  **identical top-10 across different queries**, top-1 constant — is **learned** (from the 0.862
  over-smoothing), not a structural inability to see the query.
- **The gate is uniform.** `evaluate_query` has **no** GNN-specific branch; the variant-A completion gate,
  depth slices, and attribution are byte-identical across all seven routers (the only router-type check,
  `_route`, merely picks `route()` vs `assemble_route_result()`, both → `RouteResult`).

**Bottom line:** a reviewer's *"isn't this your design's fault?"* is answered with data — **no**. The hub
is real, the control is symmetric (down to the MP-only isolation), the GNN can see the query by
construction, and the gate is uniform.

## Scope — honest bounds

This is **not** the claim that "message passing always fails." It is bounded to this **problem class**: a
**heterophilic, hub-dominated, frequency-biased-label** graph — which is exactly the character of this MCP
tool-routing benchmark (heterophily Jaccard 0.08; hub in-degree 371; labels 80.8% frequent). On such a
graph, message passing's homophily assumption breaks and learning collapses to a frequency shortcut. On
**homophilic** graphs GNNs remain effective; nothing here contradicts that. The finding is specific to
tool-dependency graphs with these measured properties, not a general verdict on GNNs.

## Practical takeaway

For tool routing with these data characteristics, a **learning-free dense-retrieval baseline beats a
dependency-aware GNN**: NaiveRAG **0.970** vs GNN **0.000** variant-A completion on identical features.
Paradoxically, the GNN's *ability to learn* is what lets it learn the bad shortcut, while NaiveRAG (no
learning) cannot and so avoids it. **Verify simple retrieval baselines before reaching for graph learning
on dependency graphs of this character.**

## Full-evaluation headline (test split, `data/processed/eval/full_eval.json`)

Test split **236 queries**, `k=10`, **5 seeds** (GNN mean ± std); baselines deterministic (ADR 0028/0029).
Deep slice = closure-depth ≥ 6, **n = 140**. Completion is variant-A (PRIMARY) with full-golden as
SECONDARY (ADR 0030); `transfer_loss` conditions retrieval-success on the variant-A spine (PRIMARY) with
full-gold as SECONDARY (ADR-0028 amendment).

| router | overall completion (variant-A / full-gold) | deep-slice `transfer_loss` (spine / full-gold) |
| --- | ---: | ---: |
| **NaiveRAG** | **0.979** / 0.114 | **0.000** / n/a |
| **HybridRAG** | **0.936** / 0.097 | **0.000** / n/a |
| **Traversal** | **0.877** / 0.093 | **0.000** / n/a |
| **BM25** | **0.725** / 0.064 | **0.077** / n/a |
| **GNN — R-GCN** | **0.000 ± 0.000** / 0.000 | **n/a** / n/a |
| **GNN — GAT** | **0.052 ± 0.022** / 0.030 | **n/a** / n/a |
| **GNN — SAGE** | **0.000 ± 0.000** / 0.000 | **n/a** / n/a |

*(SECONDARY full-gold deep `transfer_loss` is `n/a` for **every** router — recalling the full label-noisy
gold at `k=10` in the deep slice is unattainable; this is exactly the pathology the ADR-0028 amendment
moved the PRIMARY off of, by conditioning on the spine.)*

**Interpretation (honest).** The baselines transfer **near-perfectly**: they retrieve the query's required
tools, so retrieval converts to completion (deep spine-conditioned `transfer_loss` **0.000** for
NaiveRAG/HybridRAG/Traversal, **0.077** for BM25). The GNN's deep `transfer_loss` is **`n/a`** — but this is
**not a gap in the metric, it is the result**: `transfer_loss = 1 − P(completion | retrieved the spine)`,
and the GNN retrieves the spine for **0/235** queries (§ burial, over-smoothing 0.862), so the conditional
denominator is **empty**. The GNN doesn't *lose* the retrieval→completion transfer — **it never earns the
transfer, because it fails at the retrieval stage first** (overall completion **0.000 / 0.052 / 0.000**).
That is the north-star signal: on this heterophilic, hub-dominated graph, the dependency-aware GNN cannot
even surface the required tools, while learning-free dense retrieval completes 88–98% of queries.

**Homophily ↔ transfer_loss correlation — not computed (honest).** No `homophily_local`↔`transfer_loss`
correlation is present in `full_eval.json` (the metric blocks carry `retrieval` / `completion` /
`transfer_loss` / `attribution` only — 0 occurrences of "homophily"). It **cannot** be formed GNN-side
regardless: the GNN's `transfer_loss` is `n/a` (undefined under the 0/235 retrieval collapse), so there is
no per-query GNN transfer-loss series to correlate against `homophily_local`. We do **not** fabricate a
number. The heterophily driver is instead evidenced directly — measured dependency-pair Jaccard **0.08**
(`docs/feasibility-completion.md:52-57`) and the message-passing over-smoothing (node pairwise cosine
0.501 → 0.862) and burial (0/235) above — rather than via a correlation the run did not produce.

## Status / links

- Establishes the GNN completion test's **xfail** (`tests/test_gnn_router.py`, ADR-0030 de-circularization)
  as a documented, config-invariant negative result — **not** under-tuning. It flips to XPASS only if a
  future change (targeting **Force 1**, the loss/label-frequency driver, or the message-passing structure
  itself) actually lifts completion off 0.
- **The full TEST-split evaluation confirms it** (`full_eval.json`, 5 seeds): GNN overall completion
  **0.000 / 0.052 / 0.000** and deep `transfer_loss` **n/a** (from 0/235 spine retrieval) reproduce the
  config-invariant collapse established on validation — so the xfail stands as an honest, documented
  negative result on held-out test data, not an artifact of tuning.
- Related decisions and evidence: ADR 0010 (three backbones), ADR 0023 (false-negative filter — hypothesis
  refuted here), ADR-0025 amendment (initial residual — probed here), ADR 0027 (`homophily_local`),
  ADR 0028 (metrics / transfer loss), ADR 0030 (variant-A completion gate), ADR 0031 + amendment (logQ —
  probed here). Root-cause measurements: `docs/feasibility-completion.md` (heterophily), and this session's
  git-clean diagnosis probes (filter exclusion, frequency↔rank, degree, router comparison, `α`/`α_res`
  sweeps).
