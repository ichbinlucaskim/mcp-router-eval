# 0017 — Latency is measured wall-clock (not synthetic); failure scenarios are deterministic, point-injected

## Status

Accepted

## Context

The deterministic mock runner (ADR 0015) executes **real code** on every call: it validates arguments
against the built JSON Schema (ADR 0014), topo-sorts the `PARAMETER_*` sub-graph (ADR 0012), and
synthesizes arguments (ADR 0016). Because that work is real, proposal **T2.2** ("per-layer latency +
trace logging") should **measure actual cost**, not simulate network degradation — the only thing
*absent* is the fictional tools' side-effecting "work," which merely makes execution time small, not
synthetic.

Failure-path tests need failures to attribute, but the **router does not exist yet** (T3), so failures
cannot come from a real bad routing decision. They are instead produced by **deliberately corrupting
the gold selection** — the same corruptions already worked through in
`docs/completion-scoring-examples.md`: drop one `PARAMETER_*` dependency (**Scenario B → CONTRACT**),
reverse dependency order (**Scenario C → EXECUTION**), or break an argument's type. This inverts the
attribution taxonomy (ROUTING / CONTRACT / EXECUTION, upstream-wins) by construction.

Fault-injection practice says such tests should be **deterministic, point-injected, and fixed** rather
than random:

- Make fault injection deterministic by injecting at **specific points** rather than randomly, use
  **fixed** values rather than ranges, and run each test multiple times to confirm consistent results
  **before** adding it to CI ([Total Shift Left — Fault Injection Testing
  Explained](https://totalshiftleft.ai/blog/fault-injection-testing-explained)).
- Corruption-style faults are applied **deterministically, not by a probability draw** — MockServer's
  body corruption "is deterministic — it is not subject to a probability draw," applied whenever the
  request is inside the active window ([MockServer — Chaos Testing & Fault
  Injection](https://www.mock-server.com/mock_server/chaos_testing.html)).
- A deterministic simulator makes faults **replayable from a seed**: with the same seed, all injected
  drops/latencies are identical and can be replayed exactly ([redis-rust — Simulated Network and Fault
  Injection](https://deepwiki.com/nerdsane/redis-rust/4.3-simulated-network-and-fault-injection)).
- The **Fault → Error → Failure** cycle (a fault causes an error; an error observed at the system
  boundary is a failure) is the dependability framing our upstream-wins attribution already follows —
  blame the earliest fault, not the boundary failure it cascades into ([Microsoft Engineering
  Fundamentals Playbook — Fault Injection
  Testing](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/fault-injection-testing/)).

## Decision

1. **Latency = measured wall-clock, per layer.** Time the routing / contract / execution layers with
   real wall-clock measurement and record a `LatencyMs` whose parts **reconcile** (sum == total). It is
   **not** a table of synthetic fixed values. Tests assert **only** reconciliation + structure — all
   layers present, values non-negative — and **never** assert absolute values, which are
   environment-dependent. Rationale: latency is a real signal, and later the GNN router's real
   inference time is measured the identical way, so the mechanism must measure, not fabricate.
2. **Failure scenarios = deterministic, point-injected.** Failures are produced by **specific,
   fixed corruptions of the gold set**, never randomly:
   - drop one `PARAMETER_*` dependency → **Scenario B / CONTRACT**;
   - reverse dependency order → **Scenario C / EXECUTION**;
   - break an argument's type → arg-validity failure.
   Each failure test is **repeatable** — same input → same verdict → same blame — confirmed by running
   the assertion path deterministically (per Total Shift Left, run it multiple times before trusting it
   in CI). No router is required: gold corruption stands in until T3.

## Consequences

- Latency reflects **real cost** (the mock's own code today; the GNN's real inference later), measured
  by one mechanism.
- **Absolute latency is not a reproducibility-test target** — only reconciliation and structure are,
  keeping CI stable across machines.
- Failure tests are **deterministic → stable in CI**: fixed corruptions give a fixed verdict and blame.
- **Failure injection needs no router**: gold-set corruption produces CONTRACT/EXECUTION failures until
  the router exists (T3), letting the attribution path be tested now.

## Relationship to proposal T2.2

This **refines** T2.2's "synthetic" nuance: execution latency is **measured, not fabricated**. Only the
fictional tools' work is absent, which simply makes execution time small — it does **not** make the
latency synthetic. What T2.2 logs is real per-layer wall-clock, reconciled to a total.

## Alternatives considered

- **Synthetic fixed latency values** — rejected: fabricating per-layer numbers measures nothing; our
  goal is real-cost **measurement** (including later GNN inference), not network-fault simulation.
- **Random failure injection** — rejected: non-deterministic and unstable in CI, and contrary to
  fault-injection best practice, which calls for **fixed, point-injected** faults confirmed
  reproducible before entering CI.

## Sources

- Total Shift Left — deterministic, point-injected, fixed fault injection; confirm reproducibility
  before CI: <https://totalshiftleft.ai/blog/fault-injection-testing-explained>
- MockServer — body corruption is deterministic, not a probability draw:
  <https://www.mock-server.com/mock_server/chaos_testing.html>
- redis-rust — seeded deterministic replay of injected latency/faults:
  <https://deepwiki.com/nerdsane/redis-rust/4.3-simulated-network-and-fault-injection>
- Microsoft Engineering Fundamentals Playbook — Fault → Error → Failure cycle:
  <https://microsoft.github.io/code-with-engineering-playbook/automated-testing/fault-injection-testing/>
