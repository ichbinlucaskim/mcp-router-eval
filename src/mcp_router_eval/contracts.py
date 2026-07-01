"""Data contracts (§3) — the frozen spine of the pipeline.

Router → **RouteResult** → Contract → **ExecPlan** → Executor → **ExecResult** → Eval → **Attribution**.

These pydantic v2 models are the boundary interfaces between the three layers. Decisions encoded here
(see ``docs/adr/``):

- **ADR 0008** — tool identity *is* the tool ``name`` string; there is no separate id in the data.
  The field is called ``tool_id`` but its value == the tool name. ``query_id`` is a synthetic
  ``q{index}`` assigned at load; a plain ``str`` here.
- **ADR 0006 / 0013** — dependency edges have exactly 4 relation types
  (``param_direct``, ``param_indirect``, ``tool_direct``, ``tool_indirect``). All 4 are carried on
  the contracts (the *representation* role). Only the ``PARAMETER_*`` subset is *ordering*-relevant
  (``ORDERING_RELATIONS``); ordering logic lives elsewhere and is **not** implemented here.
- **ADR 0004** — ``ExecResult.completed`` is a **structural proxy** (correct tool set + dependency
  order + type-valid args), *not* semantic task success. ``ToolCall.args`` are **synthesized** by the
  mock executor (the data ships no gold args), hence ``ToolCall.synthetic`` defaults to ``True``.

Every model sets ``extra="forbid"`` so unknown fields raise (catches producer/consumer drift), and
``populate_by_name=True`` so the ``schema_`` ↔ ``"schema"`` alias round-trips both ways.

Scope (T1.1): the four boundary contracts + nested types only. ToolGraph / Embedder contracts are
intentionally **not** frozen here (YAGNI — they freeze in their own module weeks).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EdgeType",
    "ORDERING_RELATIONS",
    "GateDecision",
    "Outcome",
    "Blame",
    "ToolScore",
    "Edge",
    "ToolSpec",
    "InvariantReport",
    "ToolCall",
    "LatencyMs",
    "RouteResult",
    "ExecPlan",
    "ExecResult",
    "Attribution",
]


# --------------------------------------------------------------------------- #
# Enums / constants
# --------------------------------------------------------------------------- #
class EdgeType(str, Enum):
    """The 4 canonical dependency relation types (ADR 0006 / 0013).

    Literal values are the frozen contract vocabulary — deliberately *not* the raw dataset's
    ``dependence_type`` strings (which the preprocessing stage, ADR 0011, normalizes into these).
    """

    PARAM_DIRECT = "param_direct"
    PARAM_INDIRECT = "param_indirect"
    TOOL_DIRECT = "tool_direct"
    TOOL_INDIRECT = "tool_indirect"


#: Relations that are *ordering*-relevant (ADR 0013): a tool needs an argument value produced by
#: another tool, so its source must run first. ``TOOL_*`` relations are representation-only and are
#: excluded from execution ordering. Documented here for downstream ordering logic; **not used** in
#: this module (no ordering logic lives in the contracts).
ORDERING_RELATIONS: frozenset[EdgeType] = frozenset(
    {EdgeType.PARAM_DIRECT, EdgeType.PARAM_INDIRECT}
)


class GateDecision(str, Enum):
    """Contract-layer gate outcome (§3.2)."""

    PASS = "pass"
    FALLBACK = "fallback"


class Outcome(str, Enum):
    """Task-level outcome for Attribution (§3.4)."""

    SUCCESS = "success"
    FAILURE = "failure"


class Blame(str, Enum):
    """Deterministic failure attribution target (§3.4)."""

    NONE = "none"
    ROUTING = "routing"
    CONTRACT = "contract"
    EXECUTION = "execution"


# --------------------------------------------------------------------------- #
# Nested types
# --------------------------------------------------------------------------- #
class ToolScore(BaseModel):
    """One tool's score+rank in the router's full ranking (feeds retrieval metrics)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tool_id: str = Field(description="Tool identity == tool name (ADR 0008).")
    score: float = Field(description="Router score for this tool given the query.")
    rank: int = Field(ge=0, description="0-indexed position in the ranking.")


class Edge(BaseModel):
    """A typed dependency edge; ``src`` depends on ``dst`` (both are tool_ids/names)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    src: str = Field(description="Dependent tool_id (name).")
    dst: str = Field(description="Dependency tool_id (name) that src depends on.")
    type: EdgeType = Field(description="One of the 4 canonical relation types (ADR 0006/0013).")


class ToolSpec(BaseModel):
    """A validated, closure-ready tool binding for the executor (§3.2).

    Describes the **normalized** shape (ADR 0011), not the raw dataset shape.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tool_id: str = Field(description="Tool identity == tool name (ADR 0008).")
    is_core: bool = Field(
        description="From func_type. A label only — carries NO 'core ⇒ leaf' implication "
        "(inspection: 30/50 core tools have dependencies)."
    )
    schema_: dict = Field(
        alias="schema",
        description="Raw JSON-Schema dict built from the tool's parameters[] "
        "(not a nested model). Aliased to 'schema' to avoid pydantic's reserved name.",
    )
    deps: list[str] = Field(
        default_factory=list,
        description="tool_ids this tool depends on (from depends_on[].name).",
    )


class InvariantReport(BaseModel):
    """What the contract layer checked and guarantees about the bound tool set (§3.2)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    closure_complete: bool = Field(description="All deps of selected tools are present.")
    dangling_params: list[str] = Field(
        default_factory=list,
        description="Required params with no sourcing tool in the set.",
    )
    violations: list[str] = Field(
        default_factory=list, description="Human-readable invariant violations."
    )


class ToolCall(BaseModel):
    """One actual tool invocation in the executor's call trace (§3.3)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tool_id: str = Field(description="Invoked tool_id (name).")
    args: dict = Field(description="Arguments passed to the call.")
    ok: bool = Field(description="Whether the call succeeded.")
    error: str | None = Field(default=None, description="Error string if the call failed, else None.")
    t_ms: int = Field(ge=0, description="Call duration in milliseconds.")
    synthetic: bool = Field(
        default=True,
        description="Args are synthesized by the mock executor — the data ships no gold args "
        "(ADR 0004). True by default; the whole benchmark is structural, not semantic.",
    )


class LatencyMs(BaseModel):
    """Per-layer + total latency in milliseconds (§3.3)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    routing: int = Field(ge=0)
    contract: int = Field(ge=0)
    execution: int = Field(ge=0)
    total: int = Field(ge=0)


# --------------------------------------------------------------------------- #
# The 4 boundary contracts
# --------------------------------------------------------------------------- #
class RouteResult(BaseModel):
    """Router → Contract (§3.1)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query_id: str = Field(description="Synthetic q{index} (ADR 0008).")
    query_text: str
    ranked_tools: list[ToolScore] = Field(
        default_factory=list, description="Full ranking, for retrieval metrics."
    )
    selected_tools: list[str] = Field(
        default_factory=list, description="top-k tool_ids after closure expansion."
    )
    closure_edges: list[Edge] = Field(
        default_factory=list, description="Dependency edges used to expand the selection."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Router self-estimate in [0,1].")
    homophily_local: float = Field(
        description="Mean neighbor similarity of the selected set (not bounded to [0,1])."
    )
    router_name: str = Field(description="For ablation bookkeeping.")


class ExecPlan(BaseModel):
    """Contract → Executor (§3.2)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query_id: str
    query_text: str
    bound_tools: list[ToolSpec] = Field(
        default_factory=list, description="Validated, closure-complete tool set."
    )
    invariant_report: InvariantReport
    gate_decision: GateDecision
    trace_id: str = Field(description="Opened trace handle.")


class ExecResult(BaseModel):
    """Executor → Eval (§3.3)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query_id: str
    trace_id: str
    call_trace: list[ToolCall] = Field(
        default_factory=list, description="Actual calls, in order."
    )
    completed: bool = Field(
        description="STRUCTURAL-proxy success (ADR 0004): correct tool set + dependency order + "
        "type-valid args. NOT semantic task completion."
    )
    latency_ms: LatencyMs
    tools_used: list[str] = Field(
        default_factory=list, description="Subset of bound_tools actually invoked."
    )


class Attribution(BaseModel):
    """Eval output — the differentiator (§3.4)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query_id: str
    outcome: Outcome
    blame: Blame
    evidence: str = Field(description="Human-readable justification for the blame assignment.")
