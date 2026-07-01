"""Deterministic failure attribution (§3.4): blame ∈ {none, ROUTING, CONTRACT, EXECUTION}.

Post-hoc rule over RouteResult + ExecResult:
  retrieval missed a required tool      -> ROUTING
  closure incomplete / dangling param   -> CONTRACT
  tools present & valid but call failed  -> EXECUTION
Required-tool set comes from golden_function_names.

STUB.
"""

raise NotImplementedError("contract_layer.attribution: not implemented yet (T1.3)")
