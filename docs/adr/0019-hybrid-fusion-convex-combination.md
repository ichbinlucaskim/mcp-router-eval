# 0019 — Hybrid fusion uses convex combination of normalized scores, not RRF

## Status

Accepted

## Context

The hybrid RAG baseline (§5.1) fuses a **lexical** signal (BM25, ADR 0018) with a **dense** signal
(the embedding provider, ADR 0003). There are two dominant ways to fuse two rankers:

- **Convex combination (CC)** — a weighted sum of the two rankers' **normalized scores**:
  `α·norm(dense) + (1−α)·norm(sparse)`.
- **Reciprocal Rank Fusion (RRF)** — a **rank-based** sum, `Σ 1/(k + rank)`, that ignores the actual
  scores and uses only each item's position.

Two facts about this project decide the choice:

- We have **abundant gold labels** — 1,569 queries (ADR 0008) — so tuning CC's single weight `α`
  against labels is entirely feasible.
- **ADR 0018 already mandates min-max normalization** of every router's scores to `[0,1]`. RRF's usual
  advantage is that it sidesteps BM25's unbounded, incomparable score scale by discarding scores
  altogether; but our normalization already removes that problem, so the main reason to reach for RRF
  does not apply here.

The empirical comparison of these fusion functions supports CC when labels are available. Bruch, Gai &
Ingber study CC vs RRF directly and find that **RRF is sensitive to its parameters**, that learning a
**convex combination is largely agnostic to the choice of score normalization**, that **CC outperforms
RRF in both in-domain and out-of-domain settings**, and that **CC is sample-efficient — it needs only a
small set of training examples to tune its single parameter** to a target domain
([Bruch et al., "An Analysis of Fusion Functions for Hybrid Retrieval", ACM TOIS 42(1) art. 20, 2023;
DOI 10.1145/3596512](https://dl.acm.org/doi/10.1145/3596512);
[arXiv:2210.11934](https://arxiv.org/abs/2210.11934);
[Pinecone summary](https://www.pinecone.io/research/an-analysis-of-fusion-functions-for-hybrid-retrieval/)).

## Decision

- **Hybrid fusion = convex combination** of the two normalized scores:

  `score(t) = α · norm(dense(t)) + (1 − α) · norm(sparse(t))`

  where both `dense` and `sparse` scores are **min-max normalized to `[0,1]`** (ADR 0018).
- **`α` default = 0.5** (a neutral start giving lexical and dense equal weight); `α` is **tunable**
  against the gold labels (sample-efficient per Bruch et al.).
- **Consistent with ADR 0018:** fusion operates on normalized **scores**, so the hybrid router's
  `confidence` stays **score-based** and comparable to the other routers' confidence — the same
  min-max-normalized signal the gate consumes. RRF, being rank-only, would discard the score
  information the gate relies on.
- **RRF is retained as a fallback / future ablation**, not the primary method. Its no-normalization,
  zero-shot strengths are real, but with labels available and normalization already in place, CC is
  preferred.

## Consequences

- Tuning `α` is **sample-efficient** given our 1,569 labeled queries, so a domain-fit weight is cheap
  to obtain.
- The hybrid router's `confidence` remains **score-based**, keeping the gate's input signal consistent
  across all routers (ADR 0018).
- CC **uses the score-distribution information** (how far apart the candidates are) that RRF discards
  by collapsing to ranks — relevant here because our completion gate reasons about confidence, not
  just order.
- One tunable hyperparameter (`α`) enters the experiment surface; its default (0.5) and any tuned value
  are recorded per run for reproducibility.

## Alternatives considered

- **RRF as the primary fusion** — rejected: it discards score-distribution information, is sensitive to
  its parameters, is outperformed by CC in-domain and out-of-domain (Bruch et al.), and is inconsistent
  with ADR 0018's normalized-**score** direction. Kept only as a fallback / ablation.
- **Raw-score convex combination** (no normalization) — rejected: BM25 scores are unbounded and not
  comparable to cosine similarities; ADR 0018's min-max normalization already solves this, and CC on
  normalized scores is what Bruch et al. evaluate.

## Sources

- Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval", ACM TOIS 42(1) art. 20
  (2023) — CC vs RRF: RRF parameter-sensitive; CC outperforms RRF in- and out-of-domain; CC
  sample-efficient (one parameter, small training set): <https://dl.acm.org/doi/10.1145/3596512> ·
  <https://arxiv.org/abs/2210.11934> ·
  <https://www.pinecone.io/research/an-analysis-of-fusion-functions-for-hybrid-retrieval/>
