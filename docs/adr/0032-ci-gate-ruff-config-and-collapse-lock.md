# 0032 — CI gate: ruff config, line width, and the collapse-lock xfail

## Status

Accepted (2026-07-20). Amends ADR-0030 (flips the collapse xfail to `strict=True`).

## Context

The repo had no CI and no linter config: `pyproject.toml [tool.*]` held only pytest
settings. The sibling repo `osm-link-inference` already lints with ruff (families
`E, F, I, UP, B, SIM, RUF`) and we want the portfolio to enforce the same gate rather than
a per-repo, tutorial-grade badge.

Two facts shaped the config:

1. **Line width.** Applying osm's exact config (`line-length = 100`) to this code produced
   **522 ruff findings**, of which **401 were E501** — and all but 4 of those lines fall in
   the 101–120 range. This code was written to a ~120-column style. Measured branch: osm
   *passes cleanly at 100* (`ruff check` exit 0), so 100 is genuinely osm's width, not an
   accommodation. Widening osm to 120 to "match" would loosen a gate that already passes,
   which would be dishonest.
2. **Ambiguous-unicode rules.** 92 findings were RUF002/RUF003 flagging `α`, `×`, `−`, `–`
   in comments and docstrings. These characters are deliberate mathematical notation carried
   from the ADRs and the GNN-collapse write-up (`α_res` residual weight, `384×` dims, signed
   deltas). A linter calling them typos is a false positive; rewriting them to `alpha`/`x`
   would degrade exactly the documentation those comments exist for.

Separately, `tests/test_gnn_router.py::test_full_pipeline_integration` was marked
`xfail(strict=False)` under ADR-0030: it locks the de-circularized GNN-collapse check and
was left non-strict so a future debiasing fix would surface as XPASS "not an error." With CI
now running that test, a silent XPASS would let a materially changed result pass unnoticed.

## Decision

- **Line width = 120** for this repo (osm stays at 100). Each repo lints at the width its code
  was actually written to; the *rule families are identical* across both repos.
- **Ignore RUF002 and RUF003** (both repos, for config symmetry): intentional math notation in
  comments/docstrings is not a defect.
- **CI gate** (`.github/workflows/ci.yml`, `push` + `pull_request`, matrix 3.11/3.12):
  `ruff check` → `fetch_data.py` → `preprocess` → `pytest`, so the full data-dependent suite
  runs, not just the data-free subset.
- **Flip the collapse xfail to `strict=True`.** An XPASS (the GNN recovering the variant-A
  spine) now **fails the build**, forcing the ADR-0030 re-evaluation rather than passing
  silently. This supersedes the `strict=False` choice recorded in ADR-0030.

## Consequences

- The gate enforces: ruff (all families above), import health across every module, the contract
  /invariant/metrics suites, and the full router + GNN integration suite on real data. 213 tests
  collect; 212 pass and 1 xfails today.
- The collapse finding is now self-guarding: the day it stops holding, CI goes red on purpose.
- The ruff config diverges from osm in exactly two, documented ways (width, RUF unicode ignore);
  anything else that fails ruff must be fixed in code, not ignored.
- `claude-agent-sdk` (a declared but off-critical-path SDK-replay stub the suite never imports)
  is deliberately excluded from the CI install, matching the environment the suite is verified in.

## Alternatives considered

- **Match osm byte-for-byte at line-length 100.** Rejected: requires hand-wrapping ~401 lines and
  de-notating 92 math-notation comments (`α`→`alpha`, `×`→`x`) — a ~500-line invasive diff that
  degrades documentation for zero correctness gain.
- **Widen osm to 120 for a single portfolio width.** Rejected: loosens a gate osm already passes;
  calling that "consistency" would be dishonest.
- **Blanket `# noqa` / drop E501 from `select`.** Rejected: that is a decorative badge, not a gate.
- **Leave the xfail non-strict.** Rejected: it would let a recovered GNN slip through CI unnoticed,
  defeating the test's purpose.
