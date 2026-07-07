"""Closure-depth slicing (ADR 0005/0024) — synthetic, no real data / no harness."""
from __future__ import annotations

from mcp_router_eval.contracts import EdgeType
from mcp_router_eval.contract_layer.invariants import Dep
from mcp_router_eval.eval.metrics import QueryResult, completion_rate
from mcp_router_eval.eval.slices import (
    DEEP,
    MEDIUM,
    SHALLOW,
    closure_size,
    depth_bucket,
    partition,
    slice_by_depth,
)


def _qr(qid, *, depth, completed=True, router="r"):
    return QueryResult(
        query_id=qid, ranked_tools=("a",), gold=frozenset({"a"}), completed=completed,
        name_valid=True, schema_valid=True, dependency_compliant=True, runtime_success=True,
        blame=None, closure_depth=depth, router_name=router,
    )


def test_depth_bucket_boundaries():
    assert depth_bucket(2) == SHALLOW and depth_bucket(3) == SHALLOW
    assert depth_bucket(4) == MEDIUM and depth_bucket(5) == MEDIUM
    assert depth_bucket(6) == DEEP and depth_bucket(9) == DEEP


def test_slice_by_depth_partitions():
    results = [_qr("s2", depth=2), _qr("s3", depth=3), _qr("m4", depth=4),
               _qr("d6", depth=6), _qr("d7", depth=7)]
    buckets = slice_by_depth(results)
    assert {b: [r.query_id for r in rs] for b, rs in buckets.items()} == {
        SHALLOW: ["s2", "s3"], MEDIUM: ["m4"], DEEP: ["d6", "d7"],
    }
    # every query lands in exactly one bucket; all three keys always present
    assert sum(len(v) for v in buckets.values()) == len(results)
    assert set(buckets) == {SHALLOW, MEDIUM, DEEP}


def test_metric_per_slice_matches_hand_value():
    # deep bucket: 2 complete of 3 → 2/3 ; shallow bucket: 1 complete of 2 → 0.5
    results = [
        _qr("d1", depth=6, completed=True), _qr("d2", depth=7, completed=True),
        _qr("d3", depth=6, completed=False),
        _qr("s1", depth=2, completed=True), _qr("s2", depth=3, completed=False),
    ]
    buckets = slice_by_depth(results)
    assert completion_rate(buckets[DEEP]) == 2 / 3
    assert completion_rate(buckets[SHALLOW]) == 0.5


def test_closure_size_from_param_deps():
    # C → B → A (all PARAMETER_*); a TOOL_* edge that must NOT expand the closure.
    tool_deps = {
        "C": [Dep(source="B", param="x", relation=EdgeType.PARAM_DIRECT),
              Dep(source="T", param=None, relation=EdgeType.TOOL_DIRECT)],
        "B": [Dep(source="A", param="y", relation=EdgeType.PARAM_INDIRECT)],
        "A": [],
    }
    assert closure_size(["C"], tool_deps) == 3          # {A, B, C}; T excluded (TOOL_*)
    assert closure_size(["A"], tool_deps) == 1          # dependency-free


def test_partition_by_router():
    results = [_qr("q1", depth=3, router="bm25"), _qr("q2", depth=3, router="gnn"),
               _qr("q3", depth=3, router="bm25")]
    grouped = partition(results, key=lambda r: r.router_name)
    assert {k: [r.query_id for r in v] for k, v in grouped.items()} == {
        "bm25": ["q1", "q3"], "gnn": ["q2"],
    }
