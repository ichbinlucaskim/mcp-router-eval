# 0020 — All routers index/embed the same tool document text (fair comparison)

## Status

Accepted

## Context

The lexical router (BM25, ADR 0018) and the dense routers (NaiveRAG, HybridRAG, ADR 0019) each turn a
tool into a piece of text — BM25 tokenizes it, the dense routers embed it (ADR 0003). If those routers
see **different** text, a measured performance gap could come from the **input** rather than the
**method**. Our thesis is a *method* comparison ("the dependency-aware GNN beats the other routers"),
so the input must be held fixed or the comparison is confounded.

This confound is real and documented in exactly the BM25-vs-dense setting:

- The SAGE retrieval study standardizes the indexed text across retrievers — "we embed the first 32,000
  tokens of each markdown file with the corresponding retriever … matching the maximum input length" —
  and explicitly warns that **"asymmetric text coverage can favor dense retrievers under early-answer
  locality"** (App. A.4). In other words, unequal text handed to lexical vs. dense retrievers biases
  the comparison, so they equalize it ([SAGE, arXiv:2602.05975](https://arxiv.org/abs/2602.05975)).
- A controlled legal-document study compares BM25 and dense retrieval (all-MiniLM-L6-v2) on the **same**
  documents, varying only the method, and finds them within 0.3 points — a fair head-to-head is only
  meaningful because the input is held constant
  ([Can Small Models Reason About Legal Documents?, arXiv:2603.25944](https://arxiv.org/abs/2603.25944)).

Our tools are short structured records, not long PDFs, so truncation is not the issue; the principle is
the same, though — the text each router consumes must be identical.

## Decision

- **Every router uses the same tool document text.** BM25, NaiveRAG, HybridRAG — and later the GNN's
  node-feature text — all consume the **identical** per-tool document: `tool_id` (as words) + any tool
  description + parameter descriptions. (The ToolLinkOS dataset ships **no** standalone tool
  description, so in practice this is the tool_id words + the JSON-Schema parameter `description`
  strings — exactly what BM25 already indexes, ADR 0018.)
- **This is ablation-A hygiene extended to the input.** Vary only the method; hold everything else —
  including the tool text — fixed. It is the same discipline that moved closure expansion into a shared
  stage (ADR 0018 amendment) and separated invariants/attribution from the routers.
- **One shared helper composes the text.** The document-building function (the existing
  `tool_document()` used by BM25) is the single source every router reuses, so there is no duplicated or
  divergent notion of "a tool's text." Relocating it to a neutral shared module (rather than living in
  `baselines.py`) is an implementation detail of the vector-baseline step, not decided here.

## Consequences

- The router comparison isolates **pure method difference** — no input-asymmetry bias of the kind SAGE
  warns about.
- The GNN's node-feature text follows the **same** convention, so the GNN-vs-baseline comparison is on
  equal textual footing too.
- A single shared helper means a future change to "what text represents a tool" changes **all** routers
  together, preventing silent drift that would reintroduce the confound.

## Alternatives considered

- **Method-optimized, different text per router** (e.g. richer text for dense, keyword-y text for
  BM25) — rejected: sensible when *optimizing a single deployed method*, but it injects input bias into
  a *comparison*, which is precisely our goal (SAGE's "asymmetric text coverage" bias).
- **Description-only text for the dense routers** — rejected: it would mismatch BM25's input (which
  also uses the tool_id), reintroducing the asymmetry this ADR exists to remove.

## Sources

- SAGE — standardizes indexed text across retrievers and warns "asymmetric text coverage can favor
  dense retrievers under early-answer locality" (App. A.4): <https://arxiv.org/abs/2602.05975>
- Can Small Models Reason About Legal Documents? — controlled BM25-vs-dense comparison on the same
  documents, varying only the method: <https://arxiv.org/abs/2603.25944>
