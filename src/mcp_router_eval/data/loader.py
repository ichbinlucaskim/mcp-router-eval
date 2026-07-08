"""Loader — read normalized artifacts from ``data/processed/`` into contract objects + injected data.

The first real data → contract-layer connection: turns the preprocess output into
:class:`~mcp_router_eval.contracts.ToolSpec` objects, query records (with synthetic ``q{index}`` ids,
ADR 0008), and the injected dependency map that :mod:`contract_layer.invariants` /
:mod:`contract_layer.attribution` consume — so they run on real data, not fixtures.

**Reads PROCESSED only (ADR 0011).** This module reads only
``data/processed/{tools.jsonl, queries.jsonl, metadata.json}``. It never reads ``data/raw/`` and
never re-runs normalization (that lives in :mod:`data.preprocess`). If the processed artifacts are
missing it raises a clear error telling the user to run preprocess.

**Execution order (ADR 0012).** :meth:`Dataset.execution_order` topologically sorts the
``PARAMETER_*`` sub-graph only; it never trusts ``golden_function_names`` list order (which is
main-first, not runnable). Cycles were already excluded by preprocess's validation hook, but the
sub-slice is re-checked defensively and raises on any cycle.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mcp_router_eval.contract_layer.invariants import Dep
from mcp_router_eval.contracts import ORDERING_RELATIONS, EdgeType, ToolSpec

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = _PKG_ROOT / "data" / "processed"

__all__ = ["Query", "Dataset", "load", "topo_order"]


def topo_order(gold: Sequence[str], tool_deps: Mapping[str, Sequence[Dep]]) -> list[str]:
    """Topologically valid run order (deps first) for ``gold`` using ``PARAMETER_*`` edges only.

    The shared kernel behind :meth:`Dataset.execution_order`; extracted so the executor (Layer 3)
    can reuse the exact ordering helper without carrying a whole :class:`Dataset` (ADR 0012/0013).
    Ties are broken by tool_id (``sorted``) so the order is **deterministic**. Raises ``ValueError``
    if the sub-slice contains a cycle (defensive; preprocess already guarantees global acyclicity).
    """
    nodes = set(gold)
    # u depends on v (PARAMETER_* only, restricted to the gold set)
    deps_in: dict[str, set[str]] = {
        u: {
            d.source
            for d in tool_deps.get(u, ())
            if d.relation in ORDERING_RELATIONS and d.source in nodes
        }
        for u in nodes
    }
    indeg = {u: len(deps_in[u]) for u in nodes}
    dependents: dict[str, set[str]] = defaultdict(set)
    for u in nodes:
        for v in deps_in[u]:
            dependents[v].add(u)
    q = deque(sorted(u for u in nodes if indeg[u] == 0))
    order: list[str] = []
    while q:
        n = q.popleft()
        order.append(n)
        for u in sorted(dependents[n]):
            indeg[u] -= 1
            if indeg[u] == 0:
                q.append(u)
    if len(order) != len(nodes):
        cyclic = sorted(n for n in nodes if indeg[n] > 0)
        raise ValueError(f"PARAMETER_* cycle within gold set: {cyclic}")
    return order


@dataclass(frozen=True)
class Query:
    """One query instance from ``queries.jsonl``."""

    query_id: str  # synthetic q{index} (ADR 0008)
    query_text: str
    main: str  # main_golden_function_name
    required_tools: tuple[str, ...]  # golden_function_names — the Attribution required set


@dataclass(frozen=True)
class Dataset:
    """Loaded dataset: contract objects + the injected data the contract layer consumes."""

    tools: dict[str, ToolSpec]  # tool_id -> ToolSpec
    tool_deps: dict[str, list[Dep]]  # tool_id -> injected deps (all relations; PARAMETER_* filtered downstream)
    queries: list[Query]  # in q{index} order
    metadata: dict

    def query_by_id(self, query_id: str) -> Query:
        for q in self.queries:
            if q.query_id == query_id:
                return q
        raise KeyError(f"no query with id {query_id!r}")

    def execution_order(self, gold: Sequence[str]) -> list[str]:
        """Return a topologically valid run order (deps first) for a gold/selected set.

        Uses **only** ``PARAMETER_*`` edges within ``gold`` (ADR 0012/0013); ``golden_function_names``
        list order is ignored. Raises ValueError if the sub-slice contains a cycle (defensive; the
        preprocess hook already guarantees the global PARAMETER_* sub-graph is acyclic).
        """
        return topo_order(gold, self.tool_deps)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"processed artifact missing: {path}. Run `python -m mcp_router_eval.data.preprocess` "
            "to generate data/processed/ from data/raw/."
        )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(processed_dir: Path = PROCESSED_DIR) -> Dataset:
    """Load processed artifacts into a :class:`Dataset`. Raises if processed data is absent."""
    tools_path = _require(processed_dir / "tools.jsonl")
    queries_path = _require(processed_dir / "queries.jsonl")
    metadata_path = _require(processed_dir / "metadata.json")

    tools: dict[str, ToolSpec] = {}
    tool_deps: dict[str, list[Dep]] = {}
    for rec in _read_jsonl(tools_path):
        tid = rec["tool_id"]
        tools[tid] = ToolSpec(
            tool_id=tid,
            is_core=rec["is_core"],
            schema=rec["schema"],  # alias for schema_
            deps=sorted({d["source"] for d in rec["deps"]}),
        )
        # ADR-0030 §3: mark each dep as required/optional by whether its sourced param is a REQUIRED
        # argument of THIS (dependent) tool. Only required-arg sources are completion requirements;
        # optional-arg (or param-less) sources are ordering-only.
        _required_args = set(rec["schema"].get("required") or [])
        tool_deps[tid] = [
            Dep(
                source=d["source"],
                param=d["param"],
                relation=EdgeType(d["relation"]),
                required=d["param"] is not None and d["param"] in _required_args,
            )
            for d in rec["deps"]
        ]

    queries = [
        Query(
            query_id=rec["query_id"],
            query_text=rec["query_text"],
            main=rec["main"],
            required_tools=tuple(rec["golden"]),
        )
        for rec in _read_jsonl(queries_path)
    ]

    metadata = json.loads(metadata_path.read_text())
    return Dataset(tools=tools, tool_deps=tool_deps, queries=queries, metadata=metadata)
