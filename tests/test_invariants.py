"""T1.2 — contract-layer invariant checks. Hand-built fixtures, no loader.

Grounded partly in the real q240 Audible PARAMETER chain from
docs/completion-scoring-examples.md: validate_email -> audible_account_login -> download_audible_book.
"""
from mcp_router_eval.contract_layer.invariants import Dep, check_invariants
from mcp_router_eval.contracts import EdgeType, InvariantReport, RouteResult


def _route(selected: list[str]) -> RouteResult:
    """Minimal valid RouteResult carrying just the selected set the checker consults."""
    return RouteResult(
        query_id="q-test",
        query_text="test",
        selected_tools=selected,
        confidence=0.5,
        homophily_local=0.0,
        router_name="test",
    )


# Real q240 Audible PARAMETER chain (all PARAMETER_DIRECT).
AUDIBLE_DEPS = {
    "download_audible_book": [Dep("audible_account_login", "session_id", EdgeType.PARAM_DIRECT)],
    "audible_account_login": [Dep("validate_email", "email", EdgeType.PARAM_DIRECT)],
    "validate_email": [],
}
AUDIBLE_FULL = ["download_audible_book", "audible_account_login", "validate_email"]


# --------------------------------------------------------------------------- #
# PASS — complete closure
# --------------------------------------------------------------------------- #
def test_complete_closure_passes():
    rep = check_invariants(_route(AUDIBLE_FULL), AUDIBLE_DEPS)
    assert rep == InvariantReport(closure_complete=True, dangling_params=[], violations=[])


def test_return_type_is_invariant_report():
    assert isinstance(check_invariants(_route(AUDIBLE_FULL), AUDIBLE_DEPS), InvariantReport)


def test_tool_with_no_deps_entry_is_fine():
    # selected tool not present in tool_deps => treated as no dependencies
    rep = check_invariants(_route(["validate_email"]), {})
    assert rep.closure_complete is True and rep.dangling_params == [] and rep.violations == []


# --------------------------------------------------------------------------- #
# FAIL — missing dependency (closure) + dangling param  (Scenario B)
# --------------------------------------------------------------------------- #
def test_missing_dependency_breaks_closure_and_dangles_param():
    # drop validate_email from the selection; audible_account_login.email is now unsourced
    rep = check_invariants(_route(["download_audible_book", "audible_account_login"]), AUDIBLE_DEPS)
    assert rep.closure_complete is False
    assert rep.dangling_params == ["audible_account_login.email"]
    assert "missing dependency validate_email required by audible_account_login" in rep.violations
    assert "dangling param audible_account_login.email" in rep.violations


def test_dangling_param_format_and_determinism():
    # two independent missing param-sources -> sorted, deduped output
    deps = {
        "book_restaurant": [
            Dep("get_current_location", "location", EdgeType.PARAM_INDIRECT),
            Dep("validate_email", "email", EdgeType.PARAM_DIRECT),
        ],
    }
    rep = check_invariants(_route(["book_restaurant"]), deps)
    assert rep.closure_complete is False
    assert rep.dangling_params == ["book_restaurant.email", "book_restaurant.location"]  # sorted
    # violations sorted & contain both missing-dep + dangling messages
    assert rep.violations == sorted(rep.violations)
    assert "missing dependency get_current_location required by book_restaurant" in rep.violations


# --------------------------------------------------------------------------- #
# core-with-deps — no "skip core" regression
# --------------------------------------------------------------------------- #
def test_core_tool_with_deps_is_still_checked():
    # A tool that is conceptually is_core=True but HAS a param-dep. is_core is not an input to the
    # checker, so nothing is skipped: the missing dep must still be flagged.
    deps = {"get_current_location": [Dep("get_location_service_status", "status", EdgeType.PARAM_DIRECT)]}
    rep = check_invariants(_route(["get_current_location"]), deps)
    assert rep.closure_complete is False
    assert rep.dangling_params == ["get_current_location.status"]


# --------------------------------------------------------------------------- #
# edge exclusion — TOOL_* deps never cause failure
# --------------------------------------------------------------------------- #
def test_tool_edge_dependency_is_ignored():
    # download_audible_book also has a TOOL_DIRECT dep on get_wifi_status that is NOT selected.
    # Because TOOL_* is excluded (ADR 0013), this must NOT break closure or dangle anything.
    deps = {
        "download_audible_book": [
            Dep("audible_account_login", "session_id", EdgeType.PARAM_DIRECT),
            Dep("get_wifi_status", None, EdgeType.TOOL_DIRECT),  # absent from selection, must be ignored
            Dep("get_battery_status", None, EdgeType.TOOL_INDIRECT),
        ],
        "audible_account_login": [],
    }
    rep = check_invariants(_route(["download_audible_book", "audible_account_login"]), deps)
    assert rep == InvariantReport(closure_complete=True, dangling_params=[], violations=[])


def test_only_parameter_relation_types_participate():
    # A PARAMETER_INDIRECT missing dep DOES fail (it's in ORDERING_RELATIONS); a TOOL_* one does not.
    deps = {
        "t": [
            Dep("param_src", "p", EdgeType.PARAM_INDIRECT),
            Dep("tool_src", None, EdgeType.TOOL_DIRECT),
        ]
    }
    rep = check_invariants(_route(["t"]), deps)
    assert rep.closure_complete is False
    assert rep.dangling_params == ["t.p"]
    assert not any("tool_src" in v for v in rep.violations)


# --------------------------------------------------------------------------- #
# ADR-0030 §3 — an OPTIONAL PARAMETER_* source is ordering-only, never a completion requirement
# --------------------------------------------------------------------------- #
# The real q240 shape: audible_account_login needs a REQUIRED `email` (from validate_email) and an
# OPTIONAL `language` (from get_system_language). Dep.required encodes the distinction.
_OPT_DEPS = {
    "audible_account_login": [
        Dep("validate_email", "email", EdgeType.PARAM_DIRECT, required=True),
        Dep("get_system_language", "language", EdgeType.PARAM_INDIRECT, required=False),
    ],
    "validate_email": [],
}


def test_absent_optional_arg_source_is_not_dangling_and_keeps_closure():
    # get_system_language (optional `language`) is NOT selected. Under ADR-0030 that must NOT break
    # closure or dangle a param — the optional source is ordering-only-if-present.
    rep = check_invariants(_route(["audible_account_login", "validate_email"]), _OPT_DEPS)
    assert rep == InvariantReport(closure_complete=True, dangling_params=[], violations=[])


def test_absent_required_arg_source_still_dangles_and_breaks_closure():
    # Counterpart: dropping validate_email (REQUIRED `email`) DOES dangle + break closure (CONTRACT),
    # while the also-absent OPTIONAL get_system_language is never flagged.
    rep = check_invariants(_route(["audible_account_login"]), _OPT_DEPS)
    assert rep.closure_complete is False
    assert rep.dangling_params == ["audible_account_login.email"]      # only the required-arg source
    assert not any("get_system_language" in v for v in rep.violations)  # optional source never flagged
    assert not any("language" in v for v in rep.violations)
