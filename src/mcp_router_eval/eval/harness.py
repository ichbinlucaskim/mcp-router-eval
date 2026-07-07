"""Evaluation harness — run the five routers on the TEST split, produce the comparison (ADR 0028).

Stage 2 of the evaluation harness. For every (query, router) on the **test split** (ADR 0024 — the
same deterministic split the trainer used, so the GNN never sees these queries), it drives the real
pipeline once —

    router.rank → shared closure (ADR 0021) → invariants → deterministic mock executor (ADR 0015)
                → deterministic attribution (ROUTING/CONTRACT/EXECUTION)

— and fills a stage-1 :class:`~mcp_router_eval.eval.metrics.QueryResult`. It then applies the stage-1
metrics (:mod:`eval.metrics`) and slices (:mod:`eval.slices`) to compare all five routers per
depth-slice, and saves a comparison artifact (JSON + a readable table) under ``data/processed/eval/``
(regenerable, gitignored).

The GNN router is loaded from a checkpoint **path** the caller supplies (any backbone); if no checkpoint
is given, the GNN is **skipped with a clear message** so baseline-only runs work. Existing code is
reused, not modified. Deterministic: a fixed seed + fixed split → reproducible numbers (measured latency
is intentionally excluded from the artifact — ADR 0017).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mcp_router_eval.contract_layer.attribution import attribute
from mcp_router_eval.contract_layer.invariants import check_invariants
from mcp_router_eval.contracts import ExecPlan, GateDecision, RouteResult
from mcp_router_eval.data.graph_build import build_graph
from mcp_router_eval.data.loader import Dataset, Query, topo_order
from mcp_router_eval.embedding.base import Embedder
from mcp_router_eval.eval import metrics as M
from mcp_router_eval.eval import slices as S
from mcp_router_eval.eval.metrics import QueryResult
from mcp_router_eval.executor.mock_tools import run as mock_run
from mcp_router_eval.routers.baselines import BM25Router, HybridRAGRouter, NaiveRAGRouter, TraversalRouter
from mcp_router_eval.routers.closure import assemble_route_result
from mcp_router_eval.routers.gnn import GNNRouter
from mcp_router_eval.routers.gnn_train import split_queries

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVAL_DIR = _PKG_ROOT / "data" / "processed" / "eval"  # gitignored (data/processed/*)


@dataclass
class EvalConfig:
    k: int = 10                 # retrieval / recall cutoff (ADR 0028; k=10 default)
    seed: int = 0               # MUST match the trainer's split seed (no leakage, ADR 0024)
    limit: int | None = None    # cap the number of test queries (small verification runs)
    threshold: float = 1.0      # retrieval-success threshold for transfer_loss (recall@k ≥ threshold)


# --------------------------------------------------------------------------- #
# Router construction (GNN from a checkpoint path; baseline-only fallback)
# --------------------------------------------------------------------------- #
def build_routers(
    dataset: Dataset, graph, embedder: Embedder, *, gnn_checkpoint: Path | None = None
) -> dict[str, object]:
    """The five routers keyed by name; the GNN is included only if a checkpoint is supplied."""
    bm25 = BM25Router(dataset)
    naive = NaiveRAGRouter(dataset, embedder)
    hybrid = HybridRAGRouter(bm25, naive)
    routers: dict[str, object] = {
        "bm25": bm25,
        "naive_rag": naive,
        "hybrid_rag": hybrid,
        "traversal": TraversalRouter(hybrid, dataset.tool_deps),
    }
    if gnn_checkpoint is not None:
        gnn = GNNRouter.from_checkpoint(Path(gnn_checkpoint), dataset, graph, embedder)
        routers[gnn.name] = gnn
    else:
        print("[harness] no GNN checkpoint supplied — skipping the GNN router (baseline-only run).")
    return routers


# --------------------------------------------------------------------------- #
# One (query, router) → QueryResult (drives the real pipeline once)
# --------------------------------------------------------------------------- #
def _route(router, query: Query, tool_deps) -> RouteResult:
    # GNNRouter.route() computes the real homophily_local; baselines use the shared assemble (sentinel).
    if isinstance(router, GNNRouter):
        return router.route(query.query_text, query.query_id)
    return assemble_route_result(router.rank(query.query_text, query.query_id), tool_deps)


def evaluate_query(router, query: Query, dataset: Dataset) -> QueryResult:
    """Run rank → closure → invariants → executor → attribution for one (query, router)."""
    tool_deps = dataset.tool_deps
    gold = frozenset(query.required_tools)
    route = _route(router, query, tool_deps)

    report = check_invariants(route, tool_deps)
    order = topo_order(route.selected_tools, tool_deps)
    plan = ExecPlan(
        query_id=query.query_id, query_text=query.query_text,
        bound_tools=[dataset.tools[t] for t in order], invariant_report=report,
        gate_decision=GateDecision.PASS, trace_id=f"eval-{router_name(router)}-{query.query_id}",
    )
    result = mock_run(plan, tool_deps, list(gold))
    att = attribute(route, result, report, required_tools=list(gold))

    # Decompose completion (ADR 0028) from the concrete run signals (no error-string coupling):
    tools_used = set(result.tools_used)
    name_valid = gold <= tools_used                                     # correct tool SET recalled
    runtime_success = all(c.ok for c in result.call_trace)              # every call ok
    dependency_compliant = report.closure_complete and not report.dangling_params
    # A runtime failure not explained by dependency non-compliance is a schema/type failure.
    schema_valid = runtime_success or not dependency_compliant

    return QueryResult(
        query_id=query.query_id,
        ranked_tools=tuple(ts.tool_id for ts in route.ranked_tools),
        gold=gold,
        completed=result.completed,
        name_valid=name_valid,
        schema_valid=schema_valid,
        dependency_compliant=dependency_compliant,
        runtime_success=runtime_success,
        blame=None if result.completed else att.blame,
        closure_depth=S.closure_size(list(gold), tool_deps),
        router_name=route.router_name,
    )


def router_name(router) -> str:
    return getattr(router, "name", type(router).__name__)


# --------------------------------------------------------------------------- #
# Metric aggregation → comparison artifact
# --------------------------------------------------------------------------- #
def _nan_to_none(x: float) -> float | None:
    return None if isinstance(x, float) and math.isnan(x) else x


def _metric_block(results: list[QueryResult], cfg: EvalConfig) -> dict:
    """The three metric groups + attribution for one result set (a router × slice)."""
    k = cfg.k
    att = M.attribution_breakdown(results)
    return {
        "n": len(results),
        "retrieval": {
            f"map@{k}": M.map_at_k(results, k),
            f"recall@{k}": M.mean_recall_at_k(results, k),
            f"ndcg@{k}": M.mean_ndcg_at_k(results, k),
        },
        "completion": {
            "rate": M.completion_rate(results),
            "sub_rates": M.completion_sub_rates(results),
        },
        "transfer_loss": {
            "conditional": _nan_to_none(M.transfer_loss_conditional(results, k, threshold=cfg.threshold)),
            "difference": M.transfer_loss_difference(results, k),
        },
        "attribution": {b.value: c for b, c in att.counts.items()},
    }


def _router_report(results: list[QueryResult], cfg: EvalConfig) -> dict:
    buckets = S.slice_by_depth(results)
    return {
        "overall": _metric_block(results, cfg),
        "slices": {name: _metric_block(rs, cfg) for name, rs in buckets.items()},
    }


def build_comparison(results_by_router: dict[str, list[QueryResult]], cfg: EvalConfig) -> dict:
    """Five routers × depth slices × metric groups, with the deep-slice transfer_loss as headline."""
    routers = {name: _router_report(rs, cfg) for name, rs in results_by_router.items()}
    headline = {
        name: routers[name]["slices"][S.DEEP]["transfer_loss"]["conditional"]
        for name in routers
    }
    return {
        "config": {"k": cfg.k, "seed": cfg.seed, "threshold": cfg.threshold,
                   "n_queries": sum(len(v) for v in results_by_router.values()) // max(len(results_by_router), 1),
                   "routers": list(routers)},
        "headline_deep_transfer_loss": headline,
        "routers": routers,
    }


def render_table(comparison: dict) -> str:
    """A compact readable table: router × {overall completion, deep completion, deep transfer_loss}."""
    lines = ["router            overall_compl  deep_compl  deep_transfer_loss"]
    for name, rep in comparison["routers"].items():
        overall = rep["overall"]["completion"]["rate"]
        deep = rep["slices"][S.DEEP]
        deep_compl = deep["completion"]["rate"]
        deep_tl = deep["transfer_loss"]["conditional"]
        deep_tl_s = "  n/a" if deep_tl is None else f"{deep_tl:.3f}"
        lines.append(f"{name:16s}  {overall:12.3f}  {deep_compl:9.3f}  {deep_tl_s:>17s}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def evaluate(
    dataset: Dataset,
    embedder: Embedder,
    *,
    config: EvalConfig | None = None,
    gnn_checkpoint: Path | None = None,
    out_dir: Path = EVAL_DIR,
    save: bool = True,
) -> dict:
    """Run all available routers on the test split and return (and optionally save) the comparison."""
    cfg = config or EvalConfig()
    graph = build_graph(dataset)
    routers = build_routers(dataset, graph, embedder, gnn_checkpoint=gnn_checkpoint)

    split = split_queries(len(dataset.queries), seed=cfg.seed)   # SAME split the trainer used → test-only
    test_queries = [dataset.queries[i] for i in split.test]
    if cfg.limit is not None:
        test_queries = test_queries[: cfg.limit]

    results_by_router = {
        name: [evaluate_query(router, q, dataset) for q in test_queries]
        for name, router in routers.items()
    }
    comparison = build_comparison(results_by_router, cfg)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"comparison_seed{cfg.seed}.json").write_text(json.dumps(comparison, indent=2, default=str))
        (out_dir / f"comparison_seed{cfg.seed}.txt").write_text(render_table(comparison))
    return comparison
