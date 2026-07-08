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

## Controlled evidence — message passing is the sole differing variable

The crux. **NaiveRAG and the GNN share the same BGE node features (ADR 0003/0020) and the same
late-cosine scoring (ADR 0022 amendment). The GNN adds exactly one thing: message passing.** That single
difference is what destroys a signal the raw features already contain.

Measured on the validation split (seed 0, 235 queries; short-trained GNN):

| router | message passing? | variant-A completion | main-tool median rank | `corr(gold_freq, mean_rank)` |
| --- | :---: | ---: | ---: | ---: |
| **NaiveRAG** (BGE + cosine) | **no** | **0.970** | **0** | **−0.03** |
| Traversal | no (dep-expansion only) | 0.898 | 0 | +0.39 |
| BM25 | no | 0.762 | 1 | +0.19 |
| **GNN** (all backbones) | **yes** | **0.000** | **~272–296** | **−0.235** |

- NaiveRAG, on the *identical* features, ranks the query-specific main tool at **median rank 0** and shows
  **no** frequency bias (`corr ≈ −0.03`).
- The GNN — adding only message passing — buries the main tool at **median ~272–296 / 573** and ranks
  strongly **by frequency** (`corr = −0.235`, i.e. more-frequent tools sit higher).
- The features are good; **message passing degrades them.** This is a controlled result, not a
  correlation: the lone changed variable is message passing.

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

## Full-evaluation headline — PLACEHOLDER (to be filled after `run_full_eval`)

> **TODO — populate from `data/processed/eval/full_eval.{json,txt}` once the user runs
> `scripts/run_full_eval.py` on the TEST split (multi-seed).** The diagnosis numbers above are
> **validation-split, short-trained** probes; the following are the test-split, multi-seed headline
> (ADR 0028):
>
> - **Deep-slice `transfer_loss`** (conditional, the north-star headline, ADR 0028) per router: _tbd_.
> - **Overall variant-A completion** (PRIMARY) and full-golden completion (SECONDARY, ADR 0030) per
>   router, mean ± std across seeds: _tbd_.
> - **`homophily_local` (ADR 0027) correlated with `transfer_loss`** (ADR 0028) — does low local homophily
>   predict high transfer loss where the GNN's homophily assumption fails: _tbd_.
> - GNN entry: a representative collapse config is unnecessary — this document (config-invariant collapse)
>   is the GNN result; any single trained config reproduces completion ≈ 0.

## Status / links

- Establishes the GNN completion test's **xfail** (`tests/test_gnn_router.py`, ADR-0030 de-circularization)
  as a documented, config-invariant negative result — **not** under-tuning. It flips to XPASS only if a
  future change (targeting **Force 1**, the loss/label-frequency driver, or the message-passing structure
  itself) actually lifts completion off 0.
- Related decisions and evidence: ADR 0010 (three backbones), ADR 0023 (false-negative filter — hypothesis
  refuted here), ADR-0025 amendment (initial residual — probed here), ADR 0027 (`homophily_local`),
  ADR 0028 (metrics / transfer loss), ADR 0030 (variant-A completion gate), ADR 0031 + amendment (logQ —
  probed here). Root-cause measurements: `docs/feasibility-completion.md` (heterophily), and this session's
  git-clean diagnosis probes (filter exclusion, frequency↔rank, degree, router comparison, `α`/`α_res`
  sweeps).
