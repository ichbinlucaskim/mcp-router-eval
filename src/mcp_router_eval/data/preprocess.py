"""Preprocessing stage (ADR 0011 / 0014): normalize raw ToolLinkOS JSON → processed artifacts.

Reads ``data/raw/{regular_tools,core_tools,instances}.json`` → normalizes the dirty data → writes
``data/processed/{tools.jsonl,queries.jsonl,metadata.json}`` (ADR 0014). Standard library only.

Every normalization rule locked in ADR 0011 is applied here, and a **validation hook** asserts the
result and **fails loudly** (raises :class:`ValidationError`) on violation. Downstream (loader,
graph_build, invariants) reads the processed output only — never raw.

Run as a script::

    python -m mcp_router_eval.data.preprocess          # writes to data/processed/
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root
RAW_DIR = _PKG_ROOT / "data" / "raw"
PROCESSED_DIR = _PKG_ROOT / "data" / "processed"

# --- normalization vocabularies ------------------------------------------------------------- #
#: raw param ``type`` → canonical JSON-Schema type. Canonical spellings are JSON Schema's own
#: ("integer"/"number"/"boolean"/"object"/"array"/"string"), so bool→boolean, int→integer,
#: float→number, dict→object, list→array.
TYPE_MAP: dict[str, str] = {
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "list": "array",
    "array": "array",
}
CANONICAL_TYPES = frozenset({"string", "integer", "number", "boolean", "object", "array"})
#: default JSON-Schema type for a param whose ``type`` is missing/None (documented in ADR 0011).
DEFAULT_TYPE = "string"

#: raw ``dependence_type`` → canonical EdgeType value (contracts.EdgeType). The malformed
#: ``PARAMETER_DEPENDS_ON`` rows normalize to ``param_direct`` (ADR 0006/0011).
RELATION_MAP: dict[str, str] = {
    "PARAMETER_DIRECTLY_DEPENDS_ON": "param_direct",
    "PARAMETER_INDIRECTLY_DEPENDS_ON": "param_indirect",
    "TOOL_DIRECTLY_DEPENDS_ON": "tool_direct",
    "TOOL_INDIRECTLY_DEPENDS_ON": "tool_indirect",
    "PARAMETER_DEPENDS_ON": "param_direct",  # malformed (2 rows) → param_direct
}
PARAM_RELATIONS = frozenset({"param_direct", "param_indirect"})  # ORDERING_RELATIONS (ADR 0013)


class ValidationError(RuntimeError):
    """Raised by the post-normalization validation hook when an invariant fails (fail loudly)."""


# --- per-record normalization --------------------------------------------------------------- #
def normalize_type(raw_type: Any, counts: Counter | None = None) -> str:
    """Map a raw param type (or missing/None) to a canonical JSON-Schema type."""
    if raw_type is None:
        if counts is not None:
            counts["missing_type_defaulted"] += 1
        return DEFAULT_TYPE
    canon = TYPE_MAP.get(raw_type)
    if canon is None:
        raise ValidationError(f"unknown param type {raw_type!r} — extend TYPE_MAP")
    if counts is not None and canon != raw_type:
        counts[f"type:{raw_type}->{canon}"] += 1
    return canon


def build_schema(params: Iterable[Mapping[str, Any]], counts: Counter | None = None) -> dict:
    """Build a JSON-Schema object dict from a tool's raw ``parameters[]`` (ADR 0011).

    Folds ``enum``/``default`` side-keys in; tolerates a param missing ``type`` (→ default) or
    ``required`` (→ not required). Non-scalar raw types (dict/list/array) become object/array.
    """
    properties: dict[str, dict] = {}
    required: list[str] = []
    for p in params:
        name = p["name"]
        prop: dict[str, Any] = {"type": normalize_type(p.get("type"), counts)}
        if "description" in p:
            prop["description"] = p["description"]
        if "enum" in p:
            prop["enum"] = p["enum"]
        if "default" in p:
            prop["default"] = p["default"]
        properties[name] = prop
        if p.get("required") is True:
            required.append(name)
        elif "required" not in p and counts is not None:
            counts["missing_required_defaulted_false"] += 1
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def normalize_tool(raw: Mapping[str, Any], counts: Counter) -> dict:
    """Normalize one raw tool into a processed record (matches contracts.ToolSpec + injected deps).

    ``is_core`` is derived from ``func_type`` with NO "core ⇒ leaf" assumption — a core tool keeps
    whatever deps it has.
    """
    deps: list[dict] = []
    for d in raw.get("depends_on", []):
        raw_rel = d.get("dependence_type")
        rel = RELATION_MAP.get(raw_rel)
        if rel is None:
            raise ValidationError(f"unknown dependence_type {raw_rel!r} on tool {raw['name']!r}")
        if raw_rel == "PARAMETER_DEPENDS_ON":
            counts["malformed_dep_rows_fixed"] += 1
        deps.append({"source": d["name"], "param": d.get("parameter_name"), "relation": rel})
    return {
        "tool_id": raw["name"],
        "func_type": raw["func_type"],
        "is_core": raw["func_type"] == "core",
        "schema": build_schema(raw.get("parameters", []), counts),
        "deps": deps,
    }


def normalize_query(index: int, raw: Mapping[str, Any]) -> dict:
    """Normalize one raw instance; assigns synthetic ``query_id = q{index}`` (ADR 0008)."""
    return {
        "query_id": f"q{index}",
        "query_text": raw["user_query"],
        "main": raw["main_golden_function_name"],
        "golden": list(raw["golden_function_names"]),
    }


# --- graph helper: cycle detection (Kahn) --------------------------------------------------- #
def _cyclic_nodes(nodes: set[str], edges: Mapping[str, set[str]]) -> list[str]:
    """Return nodes left unresolved by a topological sort (i.e. involved in a cycle)."""
    indeg = {n: 0 for n in nodes}
    adj: dict[str, set[str]] = defaultdict(set)
    for u in nodes:
        for v in edges.get(u, ()):  # u depends on v
            if v in nodes:
                indeg[u] += 1
                adj[v].add(u)
    q = deque(n for n in nodes if indeg[n] == 0)
    seen = 0
    while q:
        n = q.popleft()
        seen += 1
        for u in adj[n]:
            indeg[u] -= 1
            if indeg[u] == 0:
                q.append(u)
    return sorted(n for n in nodes if indeg[n] > 0)


# --- validation hook (fail loudly) ---------------------------------------------------------- #
def validate(tools: list[dict], queries: list[dict]) -> dict:
    """Assert the normalized data is clean; raise :class:`ValidationError` on any violation.

    Checks: canonical type vocabulary only; 573 tools (523 regular + 50 core); referential integrity
    of every dependency target and every golden/main name; the PARAMETER_* sub-graph is acyclic
    (the cycle blocker). The full 4-type graph being cyclic is EXPECTED and does not fail.
    """
    names = {t["tool_id"] for t in tools}

    # 1. canonical type vocabulary only
    stray = sorted(
        {pr["type"] for t in tools for pr in t["schema"]["properties"].values()} - CANONICAL_TYPES
    )
    if stray:
        raise ValidationError(f"non-canonical param types remain after normalization: {stray}")

    # 2. tool counts
    n_regular = sum(1 for t in tools if not t["is_core"])
    n_core = sum(1 for t in tools if t["is_core"])
    if (len(tools), n_regular, n_core) != (573, 523, 50):
        raise ValidationError(
            f"tool counts off: total={len(tools)} regular={n_regular} core={n_core} "
            "(expected 573 / 523 / 50)"
        )

    # 3. referential integrity
    dangling_deps = sorted(
        {d["source"] for t in tools for d in t["deps"] if d["source"] not in names}
    )
    dangling_gold = sorted(
        {g for q in queries for g in ([q["main"], *q["golden"]]) if g not in names}
    )
    if dangling_deps or dangling_gold:
        raise ValidationError(
            f"referential integrity broken: dangling deps={dangling_deps} "
            f"dangling golden names={dangling_gold}"
        )

    # 4. PARAMETER_* sub-graph must be acyclic; full graph is expected-cyclic (not a failure)
    param_edges: dict[str, set[str]] = {
        t["tool_id"]: {d["source"] for d in t["deps"] if d["relation"] in PARAM_RELATIONS}
        for t in tools
    }
    param_cyclic = _cyclic_nodes(names, param_edges)
    if param_cyclic:
        raise ValidationError(
            f"PARAMETER_* sub-graph is cyclic (should be a DAG) — offending tools: {param_cyclic}"
        )
    full_edges: dict[str, set[str]] = {
        t["tool_id"]: {d["source"] for d in t["deps"]} for t in tools
    }
    full_cyclic = _cyclic_nodes(names, full_edges)  # expected non-empty; informational only

    return {
        "canonical_types_only": True,
        "tool_count_ok": True,
        "referential_integrity": True,
        "parameter_subgraph_acyclic": True,
        "full_graph_cyclic_expected": len(full_cyclic) > 0,
        "full_graph_cyclic_node_count": len(full_cyclic),
    }


# --- driver --------------------------------------------------------------------------------- #
def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def preprocess(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, write: bool = True) -> dict:
    """Normalize raw ToolLinkOS → processed artifacts. Returns the metadata/normalization report.

    Raises :class:`ValidationError` (via :func:`validate`) if the normalized data is not clean.
    """
    regular = json.loads((raw_dir / "regular_tools.json").read_text())
    core = json.loads((raw_dir / "core_tools.json").read_text())
    instances = json.loads((raw_dir / "instances.json").read_text())

    counts: Counter = Counter()
    tools = [normalize_tool(t, counts) for t in (regular + core)]
    queries = [normalize_query(i, q) for i, q in enumerate(instances)]

    validation = validate(tools, queries)

    edge_relations = Counter(d["relation"] for t in tools for d in t["deps"])
    nonscalar = sum(
        1
        for t in tools
        for pr in t["schema"]["properties"].values()
        if pr["type"] in {"object", "array"}
    )
    metadata = {
        "counts": {
            "tools": len(tools),
            "regular": sum(1 for t in tools if not t["is_core"]),
            "core": sum(1 for t in tools if t["is_core"]),
            "queries": len(queries),
            "edges": sum(len(t["deps"]) for t in tools),
        },
        "edge_relations": dict(sorted(edge_relations.items())),
        "normalizations": {
            "canonical_types": sorted(CANONICAL_TYPES),
            "type_aliases_normalized": {k: v for k, v in sorted(counts.items()) if k.startswith("type:")},
            "malformed_dep_rows_fixed": counts.get("malformed_dep_rows_fixed", 0),
            "params_missing_type_defaulted": counts.get("missing_type_defaulted", 0),
            "params_missing_required_defaulted_false": counts.get("missing_required_defaulted_false", 0),
            "nonscalar_params": nonscalar,
        },
        "validation": validation,
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "tools.jsonl", tools)
        _write_jsonl(out_dir / "queries.jsonl", queries)
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    return metadata


def main() -> int:
    report = preprocess()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
