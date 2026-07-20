"""Data pipeline step 3 — graph_build (loader → PyG typed-edge graph).

Runs against real data/processed/ (generate via preprocess). Includes a model smoke test:
the graph must be consumable by RGCNConv in a single forward pass.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.nn import RGCNConv  # noqa: E402

from mcp_router_eval.data.graph_build import (  # noqa: E402
    EDGE_TYPE_TO_INT,
    IS_CORE_COL,
    NUM_RELATIONS,
    ToolGraph,
    build_graph,
)
from mcp_router_eval.data.loader import load  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (Path("data/processed") / "tools.jsonl").exists(),
    reason="processed data absent; run `python -m mcp_router_eval.data.preprocess`",
)


@pytest.fixture(scope="module")
def graph() -> ToolGraph:
    return build_graph(load())


def test_node_and_edge_counts_reconcile(graph):
    d = graph.data
    assert d.x.shape[0] == 573  # 573 tools
    ds = load()
    expected_edges = sum(len(v) for v in ds.tool_deps.values())
    assert d.edge_index.shape[1] == expected_edges
    assert d.edge_type.shape[0] == expected_edges
    assert expected_edges == ds.metadata["counts"]["edges"]  # == 1496


def test_edge_type_values_and_num_relations(graph):
    from collections import Counter

    from mcp_router_eval.contracts import EdgeType

    assert sorted(set(graph.data.edge_type.tolist())) == [0, 1, 2, 3]
    assert graph.num_relations == NUM_RELATIONS == 4
    # per-relation counts match preprocess's edge_relations report
    counts = Counter(graph.data.edge_type.tolist())
    rel = load().metadata["edge_relations"]
    assert counts[EDGE_TYPE_TO_INT[EdgeType.PARAM_DIRECT]] == rel["param_direct"]
    assert counts[EDGE_TYPE_TO_INT[EdgeType.TOOL_DIRECT]] == rel["tool_direct"]


def test_mapping_is_bijective_and_stable(graph):
    n = graph.data.x.shape[0]
    assert len(graph.node_ids) == n == len(set(graph.node_ids))
    for tid in graph.node_ids:
        assert graph.tool_at(graph.index_of(tid)) == tid  # id -> index -> id round-trip
    # known tool lands where expected
    idx = graph.index_of("download_audible_book")
    assert 0 <= idx < n and graph.tool_at(idx) == "download_audible_book"


def test_is_core_feature(graph):
    x = graph.data.x
    assert float(x[graph.index_of("validate_email"), IS_CORE_COL]) == 1.0  # core tool
    assert float(x[graph.index_of("download_audible_book"), IS_CORE_COL]) == 0.0  # regular tool


def test_no_orphan_or_dangling_edges(graph):
    d = graph.data
    n = d.x.shape[0]
    assert int(d.edge_index.max()) < n and int(d.edge_index.min()) >= 0  # every endpoint is a valid node


def test_edge_direction_dependency_to_dependent(graph):
    # download_audible_book depends on audible_account_login -> edge (login -> download) must exist
    d = graph.data
    src_dep = graph.index_of("audible_account_login")
    dst_dependent = graph.index_of("download_audible_book")
    pairs = set(zip(d.edge_index[0].tolist(), d.edge_index[1].tolist(), strict=False))
    assert (src_dep, dst_dependent) in pairs


def test_rgcn_forward_smoke(graph):
    # the graph must be consumable by the model: one shape-valid forward pass, no training.
    d = graph.data
    in_dim = d.x.shape[1]
    hidden = 8
    conv = RGCNConv(in_dim, hidden, num_relations=graph.num_relations)
    out = conv(d.x, d.edge_index, d.edge_type)
    assert out.shape == (d.x.shape[0], hidden)
