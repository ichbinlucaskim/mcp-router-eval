"""T (data pipeline) — preprocessing normalization + validation hook (ADR 0011/0014).

Runs against the real raw ToolLinkOS data in data/raw/ (fetched via scripts/fetch_data.py).
"""
import json
from pathlib import Path

import pytest

from mcp_router_eval.data.preprocess import (
    ValidationError,
    _cyclic_nodes,
    build_schema,
    normalize_tool,
    normalize_type,
    preprocess,
    validate,
)

RAW = Path("data/raw")
pytestmark = pytest.mark.skipif(
    not (RAW / "regular_tools.json").exists(),
    reason="raw data absent; run scripts/fetch_data.py",
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --------------------------------------------------------------------------- #
# Type vocabulary normalization
# --------------------------------------------------------------------------- #
def test_type_alias_normalization():
    assert normalize_type("bool") == "boolean"
    assert normalize_type("boolean") == "boolean"
    assert normalize_type("int") == "integer"
    assert normalize_type("integer") == "integer"
    assert normalize_type("float") == "number"
    assert normalize_type("dict") == "object"
    assert normalize_type("list") == "array"
    assert normalize_type("array") == "array"
    assert normalize_type(None) == "string"  # missing type -> documented default


def test_unknown_type_fails_loudly():
    with pytest.raises(ValidationError):
        normalize_type("complex128")


# --------------------------------------------------------------------------- #
# JSON-Schema builder: non-scalar, enum/default, missing keys
# --------------------------------------------------------------------------- #
def test_nonscalar_params_become_valid_json_schema():
    schema = build_schema([
        {"name": "settings", "type": "dict", "required": True},
        {"name": "items", "type": "list", "required": False},
    ])
    assert schema["properties"]["settings"]["type"] == "object"
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["required"] == ["settings"]  # only required=True collected


def test_enum_and_default_folded_in():
    schema = build_schema([
        {"name": "cuisine", "type": "string", "required": False,
         "enum": ["italian", "thai"], "default": "italian"},
    ])
    prop = schema["properties"]["cuisine"]
    assert prop["enum"] == ["italian", "thai"] and prop["default"] == "italian"


def test_missing_type_and_required_tolerated():
    schema = build_schema([
        {"name": "a", "description": "no type key"},          # missing type -> default string
        {"name": "b", "type": "string"},                       # missing required -> not required
    ])
    assert schema["properties"]["a"]["type"] == "string"
    assert "required" not in schema  # neither param was required=True


# --------------------------------------------------------------------------- #
# Malformed dependency rows -> param_direct; core-with-deps keeps deps
# --------------------------------------------------------------------------- #
def test_malformed_dep_row_becomes_param_direct():
    from collections import Counter
    tool = {
        "name": "cancel_doctors_appointment", "func_type": "regular", "parameters": [],
        "depends_on": [{"name": "get_doctor_appointments", "dependence_type": "PARAMETER_DEPENDS_ON",
                        "parameter_name": "appointment_id"}],
    }
    out = normalize_tool(tool, Counter())
    assert out["deps"][0]["relation"] == "param_direct"


def test_core_tool_keeps_its_deps_no_leaf_assumption():
    from collections import Counter
    tool = {
        "name": "get_current_location", "func_type": "core", "parameters": [],
        "depends_on": [{"name": "get_location_service_status",
                        "dependence_type": "TOOL_DIRECTLY_DEPENDS_ON", "parameter_name": None}],
    }
    out = normalize_tool(tool, Counter())
    assert out["is_core"] is True and len(out["deps"]) == 1


# --------------------------------------------------------------------------- #
# Cycle helper (the blocker check)
# --------------------------------------------------------------------------- #
def test_cyclic_nodes_helper():
    assert _cyclic_nodes({"a", "b"}, {"a": {"b"}, "b": {"a"}}) == ["a", "b"]  # mutual cycle
    assert _cyclic_nodes({"a", "b", "c"}, {"a": {"b"}, "b": {"c"}}) == []      # chain = DAG


def test_validate_fails_loudly_on_noncanonical_type():
    bad = [{"tool_id": "t", "is_core": False,
            "schema": {"type": "object", "properties": {"p": {"type": "int"}}}, "deps": []}]
    with pytest.raises(ValidationError):
        validate(bad, [])


# --------------------------------------------------------------------------- #
# Real-data integration: hook passes, counts, acyclicity, referential integrity
# --------------------------------------------------------------------------- #
def test_preprocess_real_data_validation_passes():
    md = preprocess(write=False)
    v = md["validation"]
    assert v["canonical_types_only"] and v["tool_count_ok"] and v["referential_integrity"]
    assert v["parameter_subgraph_acyclic"] is True
    assert v["full_graph_cyclic_expected"] is True and v["full_graph_cyclic_node_count"] == 485
    assert md["counts"] == {"tools": 573, "regular": 523, "core": 50, "queries": 1569, "edges": 1496}
    assert md["normalizations"]["malformed_dep_rows_fixed"] == 2
    assert md["normalizations"]["nonscalar_params"] == 21


def test_real_full_graph_cyclic_but_param_subgraph_acyclic(tmp_path):
    preprocess(out_dir=tmp_path)
    tools = _load_jsonl(tmp_path / "tools.jsonl")
    names = {t["tool_id"] for t in tools}
    full = {t["tool_id"]: {d["source"] for d in t["deps"]} for t in tools}
    param = {t["tool_id"]: {d["source"] for d in t["deps"] if d["relation"].startswith("param")}
             for t in tools}
    assert _cyclic_nodes(names, full)          # full graph cyclic (expected)
    assert _cyclic_nodes(names, param) == []   # PARAMETER sub-graph acyclic


# --------------------------------------------------------------------------- #
# q240 Audible chain survives with correct param deps
# --------------------------------------------------------------------------- #
def test_q240_audible_chain_survives(tmp_path):
    preprocess(out_dir=tmp_path)
    tools = {t["tool_id"]: t for t in _load_jsonl(tmp_path / "tools.jsonl")}
    dl = tools["download_audible_book"]
    login = tools["audible_account_login"]
    # param-source spine (PARAMETER_DIRECT)
    assert {"source": "audible_account_login", "param": "session_id", "relation": "param_direct"} in dl["deps"]
    assert {"source": "validate_email", "param": "email", "relation": "param_direct"} in login["deps"]
    assert tools["validate_email"]["is_core"] is True  # validate_email is a core tool


# --------------------------------------------------------------------------- #
# Round-trip: processed JSONL reloads to equivalent structures
# --------------------------------------------------------------------------- #
def test_jsonl_round_trip(tmp_path):
    preprocess(out_dir=tmp_path)
    tools = _load_jsonl(tmp_path / "tools.jsonl")
    queries = _load_jsonl(tmp_path / "queries.jsonl")
    assert len(tools) == 573 and len(queries) == 1569
    # each line reloads to a dict equal to itself re-serialized (stable round-trip)
    line0 = (tmp_path / "tools.jsonl").read_text().splitlines()[0]
    assert json.loads(line0) == tools[0]
    assert queries[0]["query_id"] == "q0"  # synthetic id (ADR 0008)
