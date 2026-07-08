# 0031 — GNN popularity debiasing: logQ correction in InfoNCE (contrastive–IPW equivalence), with a note on our false-negative filter

## Status

Accepted

## Context

De-circularizing the completion tests (checkup step 5, ADR-0030) surfaced a concrete failure of the GNN
router: it does **not** recover a query's required-arg dependency spine, and instead ranks
high-frequency, query-irrelevant tools at the top.

Measured this session on q240 (rgcn, seed 0, default hyperparameters):

- the query's **main** gold tool `download_audible_book` ranks **~387/573 at 5 epochs and sinks to
  ~507/573 by 50–150 epochs** — i.e. **training longer makes it worse**;
- the top-k is dominated by **high-frequency `TOOL_*` connectivity tools** (`get_wifi_status`,
  `set_wifi_status`, `get_cellular_service_status`, …). `get_wifi_status` appears in **~184/235**
  validation gold sets (ADR-0004 amendment / ADR-0030) — it is a near-ubiquitous label association, not
  a query-specific signal;
- consequently the variant-A required-arg set is **not** a subset of the GNN's selection, and the
  de-circularized GNN completion test **xfails** (`tests/test_gnn_router.py::test_full_pipeline_integration`).

This is a textbook **popularity-amplification / query-agnostic collapse**: the model wins average loss by
ranking frequently-co-occurring items regardless of the query. Two sources verified this session frame
both the diagnosis and the remedy:

- **CLRec** — Zhou, Ma, Zhang, Zhou & Yang, *"Contrastive Learning for Debiased Candidate Generation in
  Large-Scale Recommender Systems"* (KDD'21). **Verified this session:** a contrastive loss has the
  **same optimum as an inverse-propensity-weighted (IPW) loss** when negatives are drawn from the
  propensity (exposure) distribution; the sampled-softmax logit carries a **`−log p_n(y)` proposal
  (logQ) correction** that **down-weights popular items**, so contrastive learning **reduces
  exposure/popularity bias**. (<https://arxiv.org/abs/2005.12964>)
- **DPAA / Debiasing Message Passing** — Islam, Faruk, Medya & Zheleva (arXiv:2605.11145). **Verified
  this session:** in GNNs, **skewed degree/frequency distributions + repeated message passing amplify
  popular items**, pulling node embeddings into a **popularity-dominated region** of the space and
  raising their scores; **IPW-family reweighting** is an established remedy.
  (<https://arxiv.org/abs/2605.11145>)

**Honest gap (CF vs. tool routing).** Both papers are **collaborative-filtering** (user–item) settings
with **queue / global** negative sampling. Ours is **query-conditioned tool routing** (ADR 0022) with
**in-batch** negatives (ADR 0026) and a **dependency-structure false-negative filter** (ADR 0023). The
**IPW / logQ principle transfers** — a frequency-skewed candidate set amplified by message passing is
exactly our situation — but the **precise application to our scoring is ours to design**; we do not claim
CLRec's user–item proof carries over verbatim. What we take is the principle: a `−log f` frequency
correction is the IPW-equivalent, principled way to stop popularity from being a free signal.

## Decision

### Primary — logQ correction in the InfoNCE logit

Subtract a popularity term from each candidate tool's logit before the softmax:

> **`score'(q, t) = cos(q, t) − α · log f(t)`**

where

- `f(t)` is tool `t`'s **gold frequency computed on the TRAIN split only** (ADR 0024 — train-only
  statistics, **no leakage** into validation/test);
- `α ≥ 0` is a coefficient **tuned on validation** and **folded into the ADR-0029 grid** (it enters the
  deterministic search like the other hyperparameters; `α = 0` recovers the current model as a grid
  point).

This down-weights high-frequency tools (`get_wifi_status`, battery/wifi/connectivity) so the GNN can no
longer win average loss by ranking them query-agnostically, forcing the query-specific signal to carry
the ranking. It is the **CLRec `−log p_n(y)` proposal correction** adapted to our query-conditioned
scoring — a scoring-level, IPW-equivalent debiasing.

### Insight to record (our setting) — the false-negative filter may have weakened popularity suppression

Per CLRec, part of what suppresses popular items in a contrastive objective is that **popular items
appear frequently as negatives** (drawn in proportion to their exposure), and each such appearance pushes
their scores down. Our **dependency false-negative filter (ADR 0023)** deliberately **excludes a query's
gold tools' `PARAMETER_*` dependencies from that query's negatives**. That filter is correct for its
purpose (it removes *definite* false negatives), but it may have **unintentionally reduced the
popularity-suppression** that in-batch negatives otherwise provide — a frequent tool that is some other
query's dependency is filtered out of the negatives that would have down-weighted it.

The logQ correction **restores that suppression explicitly and independently of the filter**: it does not
depend on a popular tool being sampled as a negative. **State this as a hypothesis to verify in step 7,
not a settled cause** — we have not measured how much suppression the filter removed; logQ is justified
on its own (CLRec) regardless of whether this interaction is confirmed.

### Ablation (not primary) — uniformity regularization

A uniformity / even-distribution regularization term is kept as an **alternative debiasing family for
comparison only** (survey-level; not a targeted frequency correction). It is an ablation, not the primary
mechanism.

### Evaluation link

**Whether the logQ correction lifts the GNN out of collapse is THE research question**, not an
implementation detail. It is measured by:

- the previously-xfailed GNN completion test flipping to a **real pass/fail** (variant-A recovery on
  q240 and across the test split, ADR-0030);
- **correlating** the effect with **`homophily_local`** (ADR 0027) and the **deep-slice
  `transfer_loss`** (ADR 0028/0030) — does debiasing raise required-arg recovery where dense retrieval's
  homophily assumption fails?

A **documented negative result** (logQ does not rescue the GNN) is an acceptable, honest outcome and is
reported as such.

## Consequences

- Reframes the finding from *"the GNN lost"* to *"the GNN hit a known failure mode (popularity
  amplification), which we address with a principled, IPW-equivalent correction, and report honestly
  (win or documented negative result)."*
- **`α` enters the re-tuning grid** (ADR 0029); the grid must be re-run after this lands (also required
  by ADR-0030's harness change — the two re-runs compose).
- **`f(t)` is train-only** — computed on the ADR-0024 train split, never on validation/test, so no
  leakage is introduced by the correction.
- The **xfailed GNN completion test becomes the pass/fail signal** for whether debiasing worked (checkup
  step 7); `strict=False` means a fix surfaces as XPASS.
- Docs/ADR only here — **no code, no harness, no grid re-run** in this ADR (later steps).

## Alternatives considered

- **(a) No debiasing** — rejected: leaves the collapse in place; the GNN keeps ranking high-frequency
  `TOOL_*` tools query-agnostically and the required-arg spine stays unrecovered.
- **(b) Heuristic `1/f` positive up-weighting** — rejected: an untheorized reweighting knob. logQ is the
  **IPW-equivalent standard** with a proof of the contrastive↔IPW correspondence (CLRec), so it is
  preferred over an ad-hoc heuristic.
- **(c) Aggregation-time reweighting (DPAA-style message-passing debiasing)** — rejected **as primary**:
  it is CF-specific and harder to port than a scoring-level logQ term; a `−log f` correction on the logit
  is a smaller, better-understood change to our query-conditioned scorer. Kept as background motivation,
  not the mechanism.
- **(d) Uniformity-only regularization** — **kept as an ablation**, not primary: it is less directly
  targeted at the frequency signal than an explicit `−α·log f(t)` term.

## Sources

*(Only these two papers are cited; both verified this session.)*

- Zhou, Ma, Zhang, Zhou & Yang, *"Contrastive Learning for Debiased Candidate Generation in Large-Scale
  Recommender Systems"* (KDD'21) — contrastive loss ≡ IPW-corrected loss under propensity-distributed
  negatives; sampled-softmax `−log p_n(y)` (logQ) proposal correction down-weights popular items;
  contrastive learning reduces exposure/popularity bias: <https://arxiv.org/abs/2005.12964>
- Islam, Faruk, Medya & Zheleva, *"Debiasing Message Passing / DPAA"* (arXiv:2605.11145) — skewed
  distributions + repeated message passing amplify popular items into a popularity-dominated embedding
  region and inflate their scores; IPW-family reweighting is an established remedy:
  <https://arxiv.org/abs/2605.11145>

## Amendment 2026-07-05 — logQ is a training-time logit correction, removed at inference

This **corrects the form** of the Primary decision above (it does **not** overturn the choice to debias
via logQ). The original wrote the correction as `score'(q,t) = cos(q,t) − α·log f(t)`, which reads as a
term applied **everywhere** (training *and* inference) with a **free per-tool weight**. The standard,
verified form is narrower: it is a **training-time importance-sampling correction on the logits, removed
at inference.**

### Corrected form

- **Training only (in the InfoNCE / sampled-softmax logits).** Subtract the log in-batch sampling
  probability from each candidate's logit, for **both the positive and the in-batch negatives**:

  > **`logit(q, t) = cos(q, t) / τ − log Q(t)`**

  where `Q(t)` is tool `t`'s **in-batch appearance probability ≈ its TRAIN-split gold frequency / total**
  (ADR 0024 — train-only, **no leakage**). This offsets the popularity penalty that in-batch negatives
  impose (frequent tools appear as negatives more often); `−log Q` is the **importance-sampling
  correction** that removes that sampling bias — **not** a free `1/f` reweight.

- **Inference (router `rank`) uses PLAIN cosine — no correction.** At serving there are **no in-batch
  negatives** (all **573** tools are scored for the query), so **there is no sampling bias to correct**.
  Applying `−log Q` at inference would **wrongly suppress popular tools**. This is the key fix relative to
  the original phrasing: the correction lives in the **loss**, not in the served score.

- **`α` is a correction-strength switch, not a per-tool weight.** `α = 1` is the standard `−log Q`;
  `α = 0` disables it (recovers today's model); `α` is tuned in the ADR-0029 grid over a small discrete
  set (e.g. `{0, 0.5, 1}`). So the training logit is `cos(q,t)/τ − α·log Q(t)`, applied to positive and
  in-batch negatives alike, and dropped entirely at inference.

### Decision — use the standard logQ

Adopt the **standard `−log Q` training correction**. The *positive-deterministic corrected variant*
(Khrylchenko et al., below — the standard form is slightly mis-derived because the positive is
deterministic, not sampled) is recorded as a **future ablation, not primary**: at our 573-tool scale the
standard form is sufficient and simpler.

### Unchanged from the original ADR-0031

Train-only frequency; the **ADR-0023 false-negative-filter interaction remains a hypothesis** to test in
step 7 (and note: with in-batch negatives now the *explicit* locus of the correction, the filter's effect
on the effective negative distribution `Q(t)` is exactly what step 7 must measure); **uniformity
regularization stays an ablation**; the **evaluation link** (correlate with `homophily_local` / deep-slice
`transfer_loss`) and the **"a documented negative result is acceptable"** stance both stand.

### Sources (this amendment; verified this session)

- Khrylchenko, Pashkov, et al. (Yandex), *"Correcting the LogQ Correction"* (RecSys'25, arXiv:2507.09331)
  — **verified this session:** exactly our setting (**two-tower + in-batch negatives + sampled softmax**);
  the **standard logQ subtracts `−log Q` from the training logits** to offset the popularity penalty from
  in-batch negatives; it also shows the standard form is **slightly mis-derived** (the positive is
  deterministic, not sampled) and gives a corrected variant: <https://arxiv.org/abs/2507.09331>
- CLRec (arXiv:2005.12964) — the contrastive↔IPW equivalence, already cited above.

**Honest gap (unchanged in spirit).** These are **recommendation / retrieval**, not tool routing. The
**two-tower + in-batch-negatives + logQ** structure matches ours closely (closer than the original
CF citations), but our **dependency false-negative filter (ADR 0023)** changes the **effective negative
distribution `Q(t)`** — an interaction we design and verify ourselves, not one the papers cover.
