"""Generic schema-driven mock executor (ADR 0004).

ToolLinkOS tools are fictional and do not run, and instances carry no gold args/answers. One
generic stub per ToolSpec validates args against the built JSONSchema and returns a canned
type-keyed value. Completion is a STRUCTURAL PROXY: correct tool set + dependency order +
type-valid args — not semantic success.

STUB.
"""

raise NotImplementedError("executor.mock_tools: not implemented yet (T2.1)")
