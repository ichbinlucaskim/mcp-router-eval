"""Data contracts (§3): RouteResult, ExecPlan, ExecResult, Attribution.

These pydantic schemas are the frozen spine of the pipeline (Router → Contract → Executor → Eval).
Edge types are the 4 real dependence types and ``is_core`` is a node feature, not an edge (ADR 0006).
Tool identity is the ``name`` string; ``query_id`` is a synthetic ``q{index}`` (ADR 0008).

STUB — schemas defined next session (T1.1).
"""

raise NotImplementedError("contracts.py: schemas not implemented yet (T1.1)")
