"""Evaluation metrics (ADR 0004/0028) — hand-computable synthetic cases, no real data / no harness."""
from __future__ import annotations

import math

from mcp_router_eval.contracts import Blame
from mcp_router_eval.eval.metrics import (
    attribution_breakdown,
    average_precision_at_k,
    completion_rate,
    completion_rate_full_golden,
    completion_sub_rates,
    map_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    ndcg_at_k,
    QueryResult,
    recall_at_k,
    transfer_loss_conditional,
    transfer_loss_difference,
)


def _qr(qid, ranked, gold, *, completed=True, completed_full_golden=None, name=True, schema=True,
        dep=True, runtime=True, blame=None, depth=3, router="r"):
    # completed_full_golden defaults to `completed` (full-golden ⇒ variant-A, since variant-A ⊆ gold);
    # pass it explicitly to model a query that completes against variant-A but NOT the full golden set.
    return QueryResult(
        query_id=qid, ranked_tools=tuple(ranked), gold=frozenset(gold), completed=completed,
        completed_full_golden=completed if completed_full_golden is None else completed_full_golden,
        name_valid=name, schema_valid=schema, dependency_compliant=dep, runtime_success=runtime,
        blame=blame, closure_depth=depth, router_name=router,
    )


# --------------------------------------------------------------------------- #
# Standard retrieval — hand values
# --------------------------------------------------------------------------- #
def test_recall_at_k_hand():
    ranked, gold = ("a", "b", "c", "d"), frozenset({"a", "c"})
    assert recall_at_k(ranked, gold, 4) == 1.0
    assert recall_at_k(ranked, gold, 1) == 0.5
    assert recall_at_k(ranked, frozenset(), 4) == 1.0        # empty gold → nothing to recall


def test_average_precision_at_k_hand():
    # a@1 (P=1), c@3 (P=2/3); AP = (1 + 2/3) / min(2,4) = 0.8333…
    assert math.isclose(average_precision_at_k(("a", "b", "c", "d"), frozenset({"a", "c"}), 4),
                        (1 + 2 / 3) / 2)
    assert average_precision_at_k(("a", "c", "b", "d"), frozenset({"a", "c"}), 4) == 1.0  # perfect


def test_ndcg_at_k_hand():
    # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(ndcg_at_k(("a", "b", "c", "d"), frozenset({"a", "c"}), 4), dcg / idcg)
    assert ndcg_at_k(("a", "c", "b", "d"), frozenset({"a", "c"}), 4) == 1.0                # perfect


def test_corpus_means():
    r1 = _qr("q1", ["a", "b", "c", "d"], {"a", "c"})   # AP 0.8333, recall@4 1.0
    r2 = _qr("q2", ["x", "a", "c", "d"], {"a", "c"})   # a@2 (0.5), c@3 (0.667) → AP (0.5+0.667)/2
    assert math.isclose(map_at_k([r1, r2], 4),
                        (((1 + 2 / 3) / 2) + ((0.5 + 2 / 3) / 2)) / 2)
    assert mean_recall_at_k([r1, r2], 4) == 1.0
    assert math.isclose(mean_ndcg_at_k([r1, r2], 4),
                        (ndcg_at_k(r1.ranked_tools, r1.gold, 4) + ndcg_at_k(r2.ranked_tools, r2.gold, 4)) / 2)


# --------------------------------------------------------------------------- #
# Structural completion — rate + decomposable sub-rates
# --------------------------------------------------------------------------- #
def test_completion_rate():
    results = [_qr(f"q{i}", ["a"], {"a"}, completed=(i < 7)) for i in range(10)]  # 7/10 complete
    assert completion_rate(results) == 0.7
    assert completion_rate([]) == 0.0


def test_primary_variant_a_completion_exceeds_secondary_full_golden():
    # ADR-0030: PRIMARY (variant-A required-set) completion is >= SECONDARY (full-golden) completion,
    # because variant-A ⊆ golden (fewer tools required ⇒ easier to satisfy). Model the direction of the
    # real BM25 spread (0.877 vs 0.098): all complete against variant-A, only some against full golden.
    results = [
        _qr("a", ["x"], {"x"}, completed=True, completed_full_golden=True),
        _qr("b", ["x"], {"x"}, completed=True, completed_full_golden=False),
        _qr("c", ["x"], {"x"}, completed=True, completed_full_golden=False),
    ]
    assert completion_rate(results) == 1.0                     # PRIMARY (variant-A)
    assert completion_rate_full_golden(results) == 1 / 3       # SECONDARY (full golden) — strictly lower
    assert completion_rate(results) >= completion_rate_full_golden(results)
    assert completion_rate_full_golden([]) == 0.0


def test_sub_rates_isolate_the_cause():
    # 4 queries; one fails ONLY dependency order (name set is correct) → dependency_compliance down,
    # name_validity stays full. This is the "decomposition isolates the cause" check.
    good = [_qr(f"g{i}", ["a"], {"a"}) for i in range(3)]
    order_fail = _qr("bad", ["a"], {"a"}, completed=False, name=True, schema=True,
                     dep=False, runtime=False, blame=Blame.EXECUTION)
    rates = completion_sub_rates(good + [order_fail])
    assert rates["name_validity"] == 1.0                    # tool set was correct for all 4
    assert rates["schema_adherence"] == 1.0
    assert rates["dependency_compliance"] == 0.75           # 3/4 — the order failure isolated here
    assert rates["runtime_success"] == 0.75
    assert completion_sub_rates([]) == {
        "name_validity": 0.0, "schema_adherence": 0.0, "dependency_compliance": 0.0, "runtime_success": 0.0,
    }


# --------------------------------------------------------------------------- #
# North-star transfer loss — primary conditional + empty denominator + secondary
# --------------------------------------------------------------------------- #
def test_transfer_loss_conditional_hand():
    # 10 queries: 8 fully recall gold (retrieval success), 2 do not. Of the 8, 3 fail completion.
    results = []
    for i in range(8):        # recall@k == 1.0 (gold in top-k); 3 of them fail completion
        results.append(_qr(f"s{i}", ["g", "x"], {"g"}, completed=(i >= 3)))
    for i in range(2):        # retrieval FAILS (gold not recalled)
        results.append(_qr(f"f{i}", ["x", "y"], {"g"}, completed=False))
    assert transfer_loss_conditional(results, k=10) == 3 / 8   # 1 - P(completion|retrieval)


def test_transfer_loss_empty_denominator_is_nan():
    # no query recalls gold → denominator empty → NaN (not a crash, not a fake 0)
    results = [_qr(f"q{i}", ["x"], {"g"}, completed=False) for i in range(3)]
    assert math.isnan(transfer_loss_conditional(results, k=10))


def test_transfer_loss_difference_secondary():
    # recall mean 1.0, completion 0.5 → difference 0.5
    results = [_qr(f"q{i}", ["g"], {"g"}, completed=(i % 2 == 0)) for i in range(4)]
    assert mean_recall_at_k(results, 10) == 1.0 and completion_rate(results) == 0.5
    assert transfer_loss_difference(results, 10, retrieval="recall") == 0.5


# --------------------------------------------------------------------------- #
# Failure attribution aggregation
# --------------------------------------------------------------------------- #
def test_attribution_breakdown():
    results = [
        _qr("a", ["g"], {"g"}, completed=True),                                # success → excluded
        _qr("b", ["g"], {"g"}, completed=False, blame=Blame.ROUTING),
        _qr("c", ["g"], {"g"}, completed=False, blame=Blame.CONTRACT),
        _qr("d", ["g"], {"g"}, completed=False, blame=Blame.CONTRACT),
    ]
    b = attribution_breakdown(results)
    assert b.total_failed == 3
    assert b.counts == {Blame.ROUTING: 1, Blame.CONTRACT: 2}
    assert b.fractions[Blame.CONTRACT] == 2 / 3
    assert attribution_breakdown([]).total_failed == 0
