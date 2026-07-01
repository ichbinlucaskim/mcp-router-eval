"""Claude Code executor via claude-agent-sdk (ADR 0002).

Drives the installed Claude Code CLI, registering the mock tools as in-process MCP tools, and
collects the structured call trace into an ExecResult (§3.3). The raw anthropic SDK alone cannot
drive the agent loop; the agent SDK provides the tool-call stream.

STUB.
"""

raise NotImplementedError("executor.claude_exec: not implemented yet (T2.1)")
