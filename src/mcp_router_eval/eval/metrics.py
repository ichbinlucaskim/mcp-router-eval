"""Evaluation metrics — PURE functions over already-produced per-query results (ADR 0028).

Stage 1 of the evaluation harness: the metric functions only. They consume :class:`QueryResult`
records (a ranked tool list, the gold set, the structural-completion outcome + its decomposition, and
the deterministic attribution) — they **never run a router or the executor** (that is stage 2). This
keeps them unit-testable with small hand-computable synthetic inputs.

Three metric groups (ADR 0028), all computed on whatever result set is passed (the harness passes the
**test split**, sliced by closure depth — see :mod:`eval.slices`):

1. **Standard retrieval** (established, uncited): ``map_at_k`` / ``recall_at_k`` / ``ndcg_at_k`` — binary
   relevance (a ranked tool is relevant iff it is in the gold set), ``k`` default 10.
2. **Structural completion** (ADR 0004): ``completion_rate`` plus decomposable **sub-rates**
   (name validity / schema-type adherence / dependency compliance / runtime success — our decomposition
   mapped onto MCP-Bench's rule-based framing, ADR 0028).
3. **North-star transfer loss** (ADR 0028): **primary** = ``1 − P(completion | retrieval success)``
   (conditional), **secondary** = ``retrieval_metric − completion_rate`` (descriptive difference).

Plus **failure-attribution aggregation**: the ROUTING/CONTRACT/EXECUTION breakdown over failed queries.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from mcp_router_eval.contracts import Blame

__all__ = [
    "QueryResult",
    "AttributionBreakdown",
    "recall_at_k",
    "average_precision_at_k",
    "ndcg_at_k",
    "map_at_k",
    "mean_recall_at_k",
    "mean_ndcg_at_k",
    "completion_rate",
    "completion_rate_full_golden",
    "completion_sub_rates",
    "retrieval_success",
    "transfer_loss_conditional",
    "transfer_loss_difference",
    "attribution_breakdown",
]


@dataclass(frozen=True)
class QueryResult:
    """One query's evaluated outcome — the pure input to every metric (produced by the stage-2 harness).

    Retrieval fields (``ranked_tools`` best-first, ``gold``) feed the retrieval metrics; the completion
    booleans feed completion + transfer loss; ``blame`` feeds attribution; ``closure_depth`` feeds
    slicing (:mod:`eval.slices`). The completion decomposition (ADR 0004 / 0028): ``completed`` is the
    overall structural verdict; the four sub-flags say *why* it passed/failed.
    """

    query_id: str
    ranked_tools: tuple[str, ...]      # full ranking, best-first (tool_ids)
    gold: frozenset[str]               # full golden set — retrieval target (recall/map/nDCG)
    completed: bool                    # PRIMARY structural completion — variant-A required-set (ADR 0004/0030)
    name_valid: bool                   # correct tool SET (all required invoked, none spurious)
    schema_valid: bool                 # call args type-valid against the schema
    dependency_compliant: bool         # PARAMETER_* order respected + every sourced arg available
    runtime_success: bool              # every call ok
    blame: Blame | None                # deterministic attribution (None on success)
    closure_depth: int                 # size of the PARAMETER_* closure (for depth slicing)
    router_name: str = "?"             # for the per-router breakdown
    completed_full_golden: bool = False  # SECONDARY completion — full golden_function_names (ADR 0030; reported, never in transfer_loss)
    required_set: frozenset[str] = frozenset()  # variant-A required-arg spine — transfer_loss's PRIMARY retrieval-success target (ADR 0028 amend / 0030)


# --------------------------------------------------------------------------- #
# 1. Standard retrieval (binary relevance; ADR 0028 — implemented, uncited)
# --------------------------------------------------------------------------- #
def recall_at_k(ranked: Sequence[str], gold: frozenset[str], k: int = 10) -> float:
    """``|gold ∩ top-k| / |gold|``. Empty gold → ``1.0`` (nothing to recall)."""
    if not gold:
        return 1.0
    topk = set(ranked[:k])
    return len(topk & gold) / len(gold)


def average_precision_at_k(ranked: Sequence[str], gold: frozenset[str], k: int = 10) -> float:
    """AP@k = ``(Σ_{i≤k} P(i)·rel(i)) / min(|gold|, k)`` with binary relevance (0.0 if empty gold)."""
    if not gold:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, tool in enumerate(ranked[:k], start=1):
        if tool in gold:
            hits += 1
            precision_sum += hits / i          # precision at this relevant position
    denom = min(len(gold), k)
    return precision_sum / denom if denom else 0.0


def ndcg_at_k(ranked: Sequence[str], gold: frozenset[str], k: int = 10) -> float:
    """nDCG@k = DCG@k / IDCG@k, binary relevance, gain ``1/log2(i+1)`` (0.0 if empty gold)."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, tool in enumerate(ranked[:k], start=1) if tool in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def map_at_k(results: Sequence[QueryResult], k: int = 10) -> float:
    """Mean Average Precision @k over the result set (0.0 if empty)."""
    if not results:
        return 0.0
    return sum(average_precision_at_k(r.ranked_tools, r.gold, k) for r in results) / len(results)


def mean_recall_at_k(results: Sequence[QueryResult], k: int = 10) -> float:
    if not results:
        return 0.0
    return sum(recall_at_k(r.ranked_tools, r.gold, k) for r in results) / len(results)


def mean_ndcg_at_k(results: Sequence[QueryResult], k: int = 10) -> float:
    if not results:
        return 0.0
    return sum(ndcg_at_k(r.ranked_tools, r.gold, k) for r in results) / len(results)


# --------------------------------------------------------------------------- #
# 2. Structural completion (ADR 0004) — decomposable (ADR 0028)
# --------------------------------------------------------------------------- #
def completion_rate(results: Sequence[QueryResult]) -> float:
    """PRIMARY completion (ADR 0004/0030): fraction completing against the **variant-A required-set**.

    0.0 if empty. This is the north-star completion; ``transfer_loss`` and the sub-rates key off it.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.completed) / len(results)


def completion_rate_full_golden(results: Sequence[QueryResult]) -> float:
    """SECONDARY completion (ADR 0030): fraction completing against the **full golden set**.

    Reported alongside the PRIMARY :func:`completion_rate` for transparency (the required-set choice is
    auditable), but **never** fed into ``transfer_loss``. 0.0 if empty.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.completed_full_golden) / len(results)


_SUB_RATE_KEYS = ("name_validity", "schema_adherence", "dependency_compliance", "runtime_success")


def completion_sub_rates(results: Sequence[QueryResult]) -> dict[str, float]:
    """The decomposable completion sub-rates (ADR 0028) — each a fraction over the result set.

    ``name_validity`` (correct tool set), ``schema_adherence`` (type-valid args),
    ``dependency_compliance`` (PARAMETER_* order + sourced), ``runtime_success`` (all calls ok). A low
    overall ``completion_rate`` is diagnosable by which sub-rate is low.
    """
    n = len(results)
    if n == 0:
        return {k: 0.0 for k in _SUB_RATE_KEYS}
    return {
        "name_validity": sum(1 for r in results if r.name_valid) / n,
        "schema_adherence": sum(1 for r in results if r.schema_valid) / n,
        "dependency_compliance": sum(1 for r in results if r.dependency_compliant) / n,
        "runtime_success": sum(1 for r in results if r.runtime_success) / n,
    }


# --------------------------------------------------------------------------- #
# 3. North-star transfer loss (ADR 0028)
# --------------------------------------------------------------------------- #
def _retrieval_target(result: QueryResult, target: str) -> frozenset[str]:
    """The recall target for transfer_loss (ADR-0028 amendment). ``'required'`` (PRIMARY) = the variant-A
    required-arg spine (``required_set``, the SAME set completion uses, ADR-0030); ``'gold'`` (SECONDARY)
    = the full ``golden_function_names``."""
    return result.required_set if target == "required" else result.gold


def retrieval_success(
    result: QueryResult, k: int = 10, *, threshold: float = 1.0, target: str = "required"
) -> bool:
    """Did retrieval succeed for this query? ``recall@k of the target ≥ threshold`` (default 1.0).

    PRIMARY (ADR-0028 amendment / ADR-0030): the target is the **variant-A required-set** — the required-arg
    ``PARAMETER_*`` spine completion uses — so the transfer conditions on retrieving the tools the task
    actually needs, not the full label-noisy gold. ``target='gold'`` gives the SECONDARY full-gold
    condition. Threshold semantics are unchanged (recall of the target ≥ threshold).
    """
    return recall_at_k(result.ranked_tools, _retrieval_target(result, target), k) >= threshold


def transfer_loss_conditional(
    results: Sequence[QueryResult], k: int = 10, *, threshold: float = 1.0, target: str = "required"
) -> float:
    """PRIMARY transfer loss = ``1 − P(completion | retrieval success)`` (ADR 0028, amended).

    Of the queries whose retrieval succeeded (**variant-A spine** recalled at ``k`` — ``target='required'``;
    or the full gold — ``target='gold'`` for the SECONDARY number), the fraction that then **fail**
    structural completion — GRETEL's ``P(functional|semantic)`` operationalized. **Empty denominator**
    (no query in the group retrieved the target) → ``float('nan')`` — an honest "couldn't even retrieve the
    required tools", not a crash and not a fake 0.
    """
    succeeded = [r for r in results if retrieval_success(r, k, threshold=threshold, target=target)]
    if not succeeded:
        return float("nan")
    failed = sum(1 for r in succeeded if not r.completed)
    return failed / len(succeeded)


def transfer_loss_difference(
    results: Sequence[QueryResult], k: int = 10, *, retrieval: str = "recall", target: str = "required"
) -> float:
    """SECONDARY (descriptive) transfer loss = ``mean retrieval@k of the target − completion_rate``.

    Recomputed against the **variant-A spine** by default (``target='required'``, ADR-0028 amendment), so it
    is target-consistent with completion; ``target='gold'`` gives the full-gold descriptive gap.
    """
    per_query = {"recall": recall_at_k, "map": average_precision_at_k, "ndcg": ndcg_at_k}[retrieval]
    if not results:
        return -completion_rate(results)  # 0.0
    mean_retrieval = sum(
        per_query(r.ranked_tools, _retrieval_target(r, target), k) for r in results
    ) / len(results)
    return mean_retrieval - completion_rate(results)


# --------------------------------------------------------------------------- #
# Failure-attribution aggregation (deterministic ROUTING/CONTRACT/EXECUTION)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttributionBreakdown:
    """Counts + fractions of ROUTING/CONTRACT/EXECUTION blame over the **failed** queries."""

    total_failed: int
    counts: dict[Blame, int]
    fractions: dict[Blame, float] = field(default_factory=dict)


def attribution_breakdown(results: Sequence[QueryResult]) -> AttributionBreakdown:
    """Deterministic blame breakdown over failed (``completed=False``) queries (ADR 0028 / §3.4).

    Group by :class:`~mcp_router_eval.contracts.Blame`; fractions are over the failed count. Callers
    slice by router / depth first (via :mod:`eval.slices`) then call this per group.
    """
    failed = [r for r in results if not r.completed]
    counts: dict[Blame, int] = {}
    for r in failed:
        label = r.blame if r.blame is not None else Blame.NONE
        counts[label] = counts.get(label, 0) + 1
    total = len(failed)
    fractions = {b: c / total for b, c in counts.items()} if total else {}
    return AttributionBreakdown(total_failed=total, counts=counts, fractions=fractions)
