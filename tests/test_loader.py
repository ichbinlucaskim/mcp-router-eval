"""Data pipeline step 2 — loader (processed → contract objects + injected data).

Runs against real data/processed/ (generate via `python -m mcp_router_eval.data.preprocess`).
"""
from pathlib import Path

import pytest

from mcp_router_eval.contracts import ToolSpec
from mcp_router_eval.data.loader import Dataset, load

pytestmark = pytest.mark.skipif(
    not (Path("data/processed") / "tools.jsonl").exists(),
    reason="processed data absent; run `python -m mcp_router_eval.data.preprocess`",
)


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return load()


def test_load_counts(ds):
    assert len(ds.tools) == 573
    assert sum(1 for t in ds.tools.values() if not t.is_core) == 523
    assert sum(1 for t in ds.tools.values() if t.is_core) == 50
    assert len(ds.queries) == 1569


def test_query_ids_are_synthetic_indexed(ds):
    assert ds.queries[0].query_id == "q0"
    assert ds.queries[240].query_id == "q240"
    assert all(q.query_id == f"q{i}" for i, q in enumerate(ds.queries))


def test_toolspec_integrity(ds):
    for tid, ts in ds.tools.items():
        assert isinstance(ts, ToolSpec)
        assert ts.tool_id == tid  # tool_id == name (ADR 0008)
        assert isinstance(ts.schema_, dict) and ts.schema_["type"] == "object"


def test_core_tool_with_deps_keeps_deps(ds):
    # get_current_location is a core tool that HAS dependencies (no core=>leaf assumption)
    ts = ds.tools["get_current_location"]
    assert ts.is_core is True and len(ts.deps) > 0


def test_gold_and_dep_referential_integrity(ds):
    names = set(ds.tools)
    for q in ds.queries:
        assert set(q.required_tools) <= names, f"{q.query_id} has unknown gold tools"
        assert q.main in names
    for tid, deps in ds.tool_deps.items():
        for d in deps:
            assert d.source in names, f"{tid} depends on unknown tool {d.source}"


# --- Execution order (ADR 0012): PARAMETER_* topo-sort, main NOT first --------------------- #
def test_execution_order_puts_main_last_q240(ds):
    q = ds.query_by_id("q240")  # the Audible query
    assert q.main == "download_audible_book"
    assert q.required_tools[0] == q.main  # stored order is main-FIRST (raw)
    order = ds.execution_order(q.required_tools)
    assert order[-1] == q.main  # topo run order: main runs LAST, after its deps
    # dependency precedes dependent along the Audible spine
    assert order.index("validate_email") < order.index("audible_account_login") < order.index("download_audible_book")


def test_execution_order_is_deterministic(ds):
    q = ds.query_by_id("q240")
    assert ds.execution_order(q.required_tools) == ds.execution_order(q.required_tools)


# --- Missing processed artifacts -> clear error -------------------------------------------- #
def test_missing_processed_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load(processed_dir=tmp_path)  # empty dir
    assert "preprocess" in str(exc.value)
