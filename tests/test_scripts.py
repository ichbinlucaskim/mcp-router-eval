"""Grid-search + full-eval script wiring (ADR 0029) — unit tests, NO training/full run."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from mcp_router_eval.eval.tuning import (
    GridRecord,
    grid_size,
    iter_configs,
    load_best_configs,
    save_best_configs,
    select_best,
    training_grid,
)
from mcp_router_eval.routers.gnn_train import GNNTrainConfig

_SCRIPTS = Path("scripts")


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Scripts import and expose a main() without running any work
# --------------------------------------------------------------------------- #
def test_scripts_import_and_have_main():
    for name in ("run_grid_search", "run_full_eval"):
        mod = _import_script(name)
        assert callable(mod.main)


# --------------------------------------------------------------------------- #
# Grid enumeration — exact config counts (ADR 0025/0026/0029)
# --------------------------------------------------------------------------- #
def test_grid_size_exact():
    t = len(training_grid())                         # |τ × lr × weight_decay|
    assert grid_size("rgcn") == 3 * 3 * t            # hidden(3) × dropout(3) × training-set
    assert grid_size("sage") == 3 * 3 * t
    assert grid_size("gat") == 3 * 2 * 3 * t         # × heads(2)
    # the enumerator produces exactly that many, all with the right backbone
    cfgs = iter_configs("gat", epochs=1, seed=0)
    assert len(cfgs) == grid_size("gat")
    assert all(c.backbone == "gat" and c.epochs == 1 and c.seed == 0 for c in cfgs)
    # non-GAT never varies heads (heads fixed at 2)
    assert {c.heads for c in iter_configs("rgcn", epochs=1, seed=0)} == {2}


# --------------------------------------------------------------------------- #
# Selection — completion_rate primary, mAP@k tiebreaker (ADR 0029)
# --------------------------------------------------------------------------- #
def test_select_best_completion_then_map_tiebreaker():
    recs = [
        GridRecord("rgcn", {"hidden": 32}, val_completion=0.50, val_map=0.90),  # lower completion
        GridRecord("rgcn", {"hidden": 64}, val_completion=0.60, val_map=0.70),  # tie completion, lower map
        GridRecord("rgcn", {"hidden": 128}, val_completion=0.60, val_map=0.85),  # tie completion, HIGHER map
    ]
    best = select_best(recs)
    assert best.config["hidden"] == 128 and best.val_completion == 0.60 and best.val_map == 0.85
    assert select_best([]) is None


# --------------------------------------------------------------------------- #
# best_configs.json round-trips
# --------------------------------------------------------------------------- #
def test_best_config_save_load_roundtrip(tmp_path):
    cfg = GNNTrainConfig(backbone="gat", hidden=128, dropout=0.3, heads=4, tau=0.05,
                         lr=5e-4, weight_decay=1e-3, epochs=30, seed=0)
    rec = GridRecord("gat", vars(cfg), val_completion=0.42, val_map=0.55)
    path = save_best_configs({"gat": rec}, tmp_path / "best_configs.json")
    loaded = load_best_configs(path)
    assert set(loaded) == {"gat"}
    got = loaded["gat"]
    assert isinstance(got, GNNTrainConfig)
    assert (got.backbone, got.hidden, got.dropout, got.heads, got.tau, got.lr, got.weight_decay) == \
           ("gat", 128, 0.3, 4, 0.05, 5e-4, 1e-3)
