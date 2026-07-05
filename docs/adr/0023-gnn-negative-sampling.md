# 0023 — GNN negative sampling: in-batch negatives + dependency-structure false-negative filter; hard negatives kept conservative

## Status

Accepted

## Context

The GNN router (ADR 0022) scores tool nodes conditioned on the query and is trained against the gold
tool sets. But the gold (`golden_function_names`, ~6 tools/query) is **positive-only** — there are no
labeled negatives — so negatives must be *derived*. The central risk in deriving them is **false
negatives**: treating a genuinely relevant tool as a negative, which pushes the model away from correct
tools and destabilizes training.

The dense-retrieval literature has studied this extensively:

- **In-batch negatives are the practical default.** The negative-sampling survey treats in-batch
  negatives as the widely-adopted baseline (random in-batch negatives are the reference point other
  methods build on), and notes that "for a long time a lot of research effort on negative sampling in
  NLP focused on false negative mitigation," with top-k filtering among the mitigations
  ([Negative Sampling Techniques in IR: A Survey, arXiv:2603.18005](https://arxiv.org/abs/2603.18005)).
- **False negatives are the failure mode, and hard negatives amplify it.** Hard negatives are the
  passages closest to the query and therefore the **most likely to be false negatives**, injecting
  noise into contrastive training ([Contrastive Confidence Regularization, AAAI'24,
  arXiv:2401.00165](https://arxiv.org/abs/2401.00165)). When such mislabeled negatives are repeatedly
  sampled they "yield conflicting gradients, pushing representations away from genuinely relevant
  content and degrading both effectiveness and training stability" — i.e. **aggressive hard-negative
  mining can backfire** ([Hard Negatives or False Negatives: Correcting Pooling Bias in Dense
  Retrieval, SIGIR'22, DOI 10.1145/3511808.3557343](https://dl.acm.org/doi/10.1145/3511808.3557343)).
- **State-of-the-art mitigation is *positive-aware* filtering.** NV-Retriever anchors on the positive
  relevance score to filter likely false negatives out of the mined set (TopK-MarginPos / TopK-PercPos)
  ([NV-Retriever, arXiv:2407.15831](https://arxiv.org/abs/2407.15831)) — but this filter is a
  *similarity estimate* of "is this negative actually relevant?".

**Our structural advantage.** All of the above must *estimate* which negatives are false, because in
text retrieval relevance is latent. We do not: our **dependency graph makes the most obvious false
negatives definite**. A query's gold tools' `PARAMETER_*` dependencies are *known to be relevant*
(they are part of the closure the task needs), so we can exclude them from the negative set **exactly**,
not by a similarity threshold.

**Honest limitation.** Every source above is **dense-retrieval (text) training**, not GNN training on a
graph. The **principles** — in-batch negatives as the default, false-negative mitigation as the priority,
hard-negative caution — transfer. The **batch construction** for our GNN (how queries and their tool
sets are packed into a batch, how the graph is shared across a batch) we design ourselves at
implementation time; no cited paper does that for a tool-dependency graph.

## Decision

- **Primary = in-batch negatives.** Within a training batch, the gold tools of *other* queries serve as
  the current query's negatives. This sidesteps the positive-only-gold problem without any hand-labeled
  negatives.
- **Dependency-structure false-negative filter (our advantage).** Before using in-batch tools as
  negatives for a query, **exclude that query's gold tools' `PARAMETER_*` dependencies** (ADR 0013)
  from its negative set. This is done **structurally** — dependencies are definite from the graph — so
  it is a **more precise** false-negative removal than the similarity-based positive-aware methods
  (NV-Retriever / CCR), for the specific case of dependency tools.
- **Hard negatives kept conservative.** No aggressive hard-negative mining (e.g. BM25 top-k as
  negatives); the IR literature shows it can backfire via false negatives / conflicting gradients.
  Aggressive or mined hard negatives are flagged as a **possible future ablation**, not the default.
- **Loss = contrastive / ranking** consistent with ADR 0022: maximize the gold tools' query-conditioned
  scores, minimize the (filtered) in-batch negatives' scores.

## Consequences

- The GNN **trains from positive-only gold** with no hand-labeled negatives.
- The **structural filter removes the most obvious false negatives precisely** — the dependency tools a
  similarity filter would only *probabilistically* catch — which is a concrete benefit of having the
  graph.
- The **conservative hard-negative stance** avoids the documented IR failure mode (backfire /
  conflicting gradients / instability).
- **Batch-construction details are deferred** to implementation; the **train/val/test split** is a
  separate later ADR (the next GNN sub-decision).

## Alternatives considered

- **Random negatives only** — rejected: weak signal, and random draws can still land on a gold tool's
  dependency (a false negative) with no filter to catch it.
- **Aggressive hard-negative mining (BM25 top-k as negatives)** — rejected as the primary strategy: the
  IR pooling-bias literature shows hard negatives are disproportionately false negatives and can
  degrade effectiveness and training stability. Kept as a **future ablation** only.
- **Similarity-based false-negative filtering (NV-Retriever style)** — noted and respected, but for the
  **dependency** case our structural filter is strictly more precise (definite vs. estimated); a
  similarity filter could still be layered on for non-dependency false negatives as future work.

## Sources

- Negative Sampling Techniques in IR: A Survey — in-batch negatives as the practical default; top-k
  filtering; false-negative mitigation a major theme: <https://arxiv.org/abs/2603.18005>
- Contrastive Confidence Regularization (AAAI'24) — hard negatives are more likely false negatives →
  training noise: <https://arxiv.org/abs/2401.00165>
- Hard Negatives or False Negatives: Correcting Pooling Bias in Dense Retrieval (SIGIR'22) — false
  negatives yield conflicting gradients, degrading effectiveness and training stability:
  <https://dl.acm.org/doi/10.1145/3511808.3557343>
- NV-Retriever — positive-aware hard-negative mining, anchoring on the positive score to filter false
  negatives: <https://arxiv.org/abs/2407.15831>
