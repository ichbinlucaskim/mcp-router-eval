# 0009 — Record architecture decisions as numbered ADRs before implementing them

## Status

Accepted

## Context

This is the foundational, meta-level decision that governs ADRs 0001–0008; it is recorded
retroactively as 0009 because ADR numbers are immutable once assigned, and renumbering the existing
eight would break their cross-references and the index. The number reflects insertion order, not
precedence.

The capstone spans many choices that *look* reversible in the moment — benchmark selection,
embedding provider, dependency edge typing, completion semantics — but whose rationale must survive
to the write-up, defense, and interviews. Those decisions were being made verbally across sessions
with no durable record, risking silent drift and lost justification.

## Decision

Every significant decision gets a numbered ADR in `docs/adr/` **before** the code that implements it
is written. Each ADR states Context / Decision / Consequences / Alternatives considered and follows
the Status lifecycle **Proposed → Accepted → (Superseded by XXXX)**. Numbers are never reused or
reassigned; a reversed decision is captured by a new ADR that supersedes the old one.

## Consequences

- Slight overhead per decision (write the ADR first).
- Full traceability from any design choice back to its rationale and the evidence behind it.
- Change happens by **supersession**, not silent edit — history stays auditable.
- The `docs/README.md` index and this process are the single entry point for "why is it this way?".

## Alternatives considered

- **Inline code comments** — rejected: not discoverable, scattered, and lost when code is refactored.
- **Commit messages** — rejected: too granular and hard to browse as a decision log.
- **A single running `decisions.md`** — rejected: no per-decision diff, no clean supersession, and
  it grows into an unstructured wall of text.
