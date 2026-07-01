"""T1.1 proof — the four boundary contracts round-trip, reject drift, and thread query_id.

These tests are the deliverable's evidence that contracts.py is frozen and self-consistent.
"""
import pytest
from pydantic import ValidationError

from mcp_router_eval.contracts import (
    ORDERING_RELATIONS,
    Attribution,
    Blame,
    Edge,
    EdgeType,
    ExecPlan,
    ExecResult,
    GateDecision,
    InvariantReport,
    LatencyMs,
    Outcome,
    RouteResult,
    ToolCall,
    ToolScore,
    ToolSpec,
)


# --------------------------------------------------------------------------- #
# Fixtures — one valid instance of each contract for a fake query.
# --------------------------------------------------------------------------- #
def _route_result() -> RouteResult:
    return RouteResult(
        query_id="q240",
        query_text="download 'The Great Gatsby' from Audible",
        ranked_tools=[
            ToolScore(tool_id="download_audible_book", score=0.9, rank=0),
            ToolScore(tool_id="audible_account_login", score=0.7, rank=1),
        ],
        selected_tools=["download_audible_book", "audible_account_login", "validate_email"],
        closure_edges=[
            Edge(src="download_audible_book", dst="audible_account_login", type=EdgeType.PARAM_DIRECT),
            Edge(src="audible_account_login", dst="validate_email", type=EdgeType.PARAM_DIRECT),
        ],
        confidence=0.82,
        homophily_local=0.13,
        router_name="rgcn",
    )


def _tool_spec() -> ToolSpec:
    return ToolSpec(
        tool_id="audible_account_login",
        is_core=False,
        schema={  # via alias
            "type": "object",
            "properties": {"email": {"type": "string"}, "password": {"type": "string"}},
            "required": ["email", "password"],
        },
        deps=["validate_email"],
    )


def _exec_plan() -> ExecPlan:
    return ExecPlan(
        query_id="q240",
        query_text="download 'The Great Gatsby' from Audible",
        bound_tools=[_tool_spec()],
        invariant_report=InvariantReport(closure_complete=True, dangling_params=[], violations=[]),
        gate_decision=GateDecision.PASS,
        trace_id="trace-q240",
    )


def _exec_result() -> ExecResult:
    return ExecResult(
        query_id="q240",
        trace_id="trace-q240",
        call_trace=[
            ToolCall(tool_id="validate_email", args={"email": "x@y.com"}, ok=True, error=None, t_ms=3),
            ToolCall(tool_id="audible_account_login", args={"email": "x@y.com", "password": "p"}, ok=True, error=None, t_ms=12),
        ],
        completed=True,
        latency_ms=LatencyMs(routing=5, contract=2, execution=20, total=27),
        tools_used=["validate_email", "audible_account_login"],
    )


def _attribution() -> Attribution:
    return Attribution(
        query_id="q240",
        outcome=Outcome.SUCCESS,
        blame=Blame.NONE,
        evidence="Closure complete; calls in topo-valid order; all args type-valid.",
    )


ALL_FACTORIES = [_route_result, _tool_spec, _exec_plan, _exec_result, _attribution]


# --------------------------------------------------------------------------- #
# 1. Round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_round_trip_by_name(factory):
    obj = factory()
    assert type(obj).model_validate(obj.model_dump()) == obj


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_round_trip_by_alias(factory):
    obj = factory()
    assert type(obj).model_validate(obj.model_dump(by_alias=True)) == obj


def test_schema_alias_round_trips_both_ways():
    spec = _tool_spec()
    # dump by alias -> "schema" key present, "schema_" absent
    dumped_alias = spec.model_dump(by_alias=True)
    assert "schema" in dumped_alias and "schema_" not in dumped_alias
    assert ToolSpec.model_validate(dumped_alias) == spec
    # dump by field name -> "schema_" key, still round-trips (populate_by_name)
    dumped_name = spec.model_dump()
    assert "schema_" in dumped_name and "schema" not in dumped_name
    assert ToolSpec.model_validate(dumped_name) == spec


# --------------------------------------------------------------------------- #
# 2. Rejection (drift / invalid values)
# --------------------------------------------------------------------------- #
def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        Attribution(query_id="q1", outcome=Outcome.SUCCESS, blame=Blame.NONE, evidence="x", bogus=1)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_confidence_out_of_range_rejected(bad):
    with pytest.raises(ValidationError):
        r = _route_result()
        RouteResult(**{**r.model_dump(), "confidence": bad})


def test_negative_t_ms_rejected():
    with pytest.raises(ValidationError):
        ToolCall(tool_id="t", args={}, ok=False, error="boom", t_ms=-1)


def test_negative_rank_rejected():
    with pytest.raises(ValidationError):
        ToolScore(tool_id="t", score=0.5, rank=-1)


# --------------------------------------------------------------------------- #
# 3. Enum literals (guard against drift from raw dependence_type names)
# --------------------------------------------------------------------------- #
def test_edge_type_literals_are_canonical():
    assert {e.value for e in EdgeType} == {
        "param_direct",
        "param_indirect",
        "tool_direct",
        "tool_indirect",
    }


def test_ordering_relations_are_parameter_only():
    assert ORDERING_RELATIONS == frozenset({EdgeType.PARAM_DIRECT, EdgeType.PARAM_INDIRECT})
    # TOOL_* are representation-only (ADR 0013)
    assert EdgeType.TOOL_DIRECT not in ORDERING_RELATIONS
    assert EdgeType.TOOL_INDIRECT not in ORDERING_RELATIONS


def test_other_enum_literals():
    assert {e.value for e in Blame} == {"none", "routing", "contract", "execution"}
    assert {e.value for e in Outcome} == {"success", "failure"}
    assert {e.value for e in GateDecision} == {"pass", "fallback"}


# --------------------------------------------------------------------------- #
# 4. End-to-end shape — query_id threads through unchanged (no logic, just shape)
# --------------------------------------------------------------------------- #
def test_query_id_threads_through_pipeline():
    qid = "q240"
    rr, ep, er, at = _route_result(), _exec_plan(), _exec_result(), _attribution()
    assert rr.query_id == ep.query_id == er.query_id == at.query_id == qid
    # trace_id threads Contract -> Executor
    assert ep.trace_id == er.trace_id
