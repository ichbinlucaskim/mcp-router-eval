"""Grid-search tuning + multi-seed full-eval logic (ADR 0024/0025/0026/0028/0029).

The reusable core behind ``scripts/run_grid_search.py`` and ``scripts/run_full_eval.py`` (thin CLI
wrappers). Kept in the package so it is importable and unit-testable without running the full work.

Reuses (never modifies) the stage-2 trainer (:mod:`routers.gnn_train`) and the stage-2 harness
(:mod:`eval.harness`). ADR 0029: deterministic **grid** search over the small discrete space, selection
by **validation `completion_rate`** (``mAP@10`` tiebreaker), on the validation split only (test
untouched, ADR 0024); the chosen config is then trained over **multiple seeds** (mean ± std).
"""
from __future__ import annotations

import itertools
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def _default_log(msg: str) -> None:
    """Live progress line to stdout (flushed so it appears immediately, not buffered)."""
    print(msg, flush=True)


def _fmt_dur(seconds: float) -> str:
    """``2m14s`` / ``1h03m20s`` style elapsed/ETA formatting."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"

from mcp_router_eval.data.graph_build import build_graph
from mcp_router_eval.data.loader import Dataset, Query
from mcp_router_eval.embedding.base import Embedder
from mcp_router_eval.eval import metrics as M
from mcp_router_eval.eval import slices as S
from mcp_router_eval.eval.harness import EVAL_DIR, EvalConfig, evaluate_query, build_routers
from mcp_router_eval.eval.metrics import QueryResult
from mcp_router_eval.routers.gnn import GNNRouter
from mcp_router_eval.routers.gnn_train import GNNTrainConfig, GNNTrainer, split_queries

# --- ADR 0025/0026 discrete search space --------------------------------------------------------- #
HIDDENS = (32, 64, 128)          # ADR 0025
DROPOUTS = (0.0, 0.3, 0.5)       # ADR 0025
GAT_HEADS = (2, 4)               # ADR 0025 (GAT only)
TAUS = (0.05, 0.1, 0.2)          # ADR 0026 InfoNCE temperature
LRS = (1e-3, 5e-4)               # ADR 0026 (around 1e-3)
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)  # ADR 0026 (1e-4..1e-2)
# logQ popularity-correction strength (ADR 0031 amendment) — a real grid axis so the re-tune actually
# explores debiasing. α=1.0 = the standard full −log Q correction (verified: "Correcting the LogQ
# Correction", arXiv:2507.09331); α=0.0 disables it (recovers today's collapsed model); α=0.5 is a
# partial correction included because our re-tune PROBE showed α=1 at a small budget *lowered* val_map
# (0.369→0.272) — full correction can over-correct at our non-web scale — so strength is validation-tuned
# (ADR 0029). Grounds: our own probe + ADR-0029 grid tuning (no unverified paper cited).
ALPHAS = (0.0, 0.5, 1.0)

BACKBONES = ("rgcn", "gat", "sage")

__all__ = [
    "training_grid", "iter_configs", "grid_size", "select_best", "GridRecord",
    "score_on_validation", "run_grid", "save_best_configs", "load_best_configs",
    "train_seeds", "run_full_eval",
]


def training_grid() -> list[tuple[float, float, float]]:
    """The (τ, lr, weight_decay) discrete combinations — the continuous-as-discrete set (ADR 0026/0029)."""
    return [(t, lr, wd) for t in TAUS for lr in LRS for wd in WEIGHT_DECAYS]


def iter_configs(backbone: str, *, epochs: int, seed: int, proj_dim: int = 128) -> list[GNNTrainConfig]:
    """Enumerate the full grid for one backbone (ADR 0029/0031): architecture × (τ, lr, wd) × α."""
    head_choices = GAT_HEADS if backbone == "gat" else (2,)  # heads only matter for GAT
    configs: list[GNNTrainConfig] = []
    for hidden, dropout, heads, (tau, lr, wd), alpha in itertools.product(
        HIDDENS, DROPOUTS, head_choices, training_grid(), ALPHAS
    ):
        configs.append(GNNTrainConfig(
            backbone=backbone, hidden=hidden, dropout=dropout, heads=heads, proj_dim=proj_dim,
            tau=tau, lr=lr, weight_decay=wd, alpha=alpha, epochs=epochs, seed=seed,
        ))
    return configs


def grid_size(backbone: str) -> int:
    """Number of configs enumerated for ``backbone`` (for auditing / tests)."""
    arch = len(HIDDENS) * len(DROPOUTS) * (len(GAT_HEADS) if backbone == "gat" else 1)
    return arch * len(training_grid()) * len(ALPHAS)


# --------------------------------------------------------------------------- #
# Validation scoring + selection (ADR 0029: completion_rate, mAP@k tiebreaker)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridRecord:
    backbone: str
    config: dict
    val_completion: float
    val_map: float


def score_on_validation(
    router: GNNRouter, val_queries: list[Query], dataset: Dataset, k: int = 10
) -> tuple[float, float]:
    """(validation ``completion_rate``, validation ``mAP@k``) for a trained GNN router (ADR 0029)."""
    results: list[QueryResult] = [evaluate_query(router, q, dataset) for q in val_queries]
    return M.completion_rate(results), M.map_at_k(results, k)


def select_best(records: list[GridRecord]) -> GridRecord | None:
    """Best config by validation ``completion_rate``, then ``mAP@k`` tiebreaker (ADR 0029)."""
    if not records:
        return None
    return max(records, key=lambda r: (r.val_completion, r.val_map))


def run_grid(
    dataset: Dataset,
    embedder: Embedder,
    *,
    backbones: tuple[str, ...] = BACKBONES,
    epochs: int = 30,
    seed: int = 0,
    k: int = 10,
    out_dir: Path = EVAL_DIR,
    graph=None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, GridRecord], list[GridRecord]]:
    """Run the full grid per backbone; return ``(best_per_backbone, all_records)`` and write audit logs.

    Validation-only (ADR 0024): trains each config with best-validation checkpointing (early-stopping-
    equivalent — the best-val-loss epoch is kept within a bounded budget), scores on the **validation**
    split, and selects by ``completion_rate``/``mAP@k``. The **test split is never touched here.**
    ``on_progress`` receives live per-config progress lines (default: flushed stdout); logging only —
    it does not affect the grid, selection, training, splits, or artifacts.
    """
    log = on_progress or _default_log
    graph = graph if graph is not None else build_graph(dataset)
    split = split_queries(len(dataset.queries), seed=seed)
    val_queries = [dataset.queries[i] for i in split.val]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "grid_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total = sum(grid_size(b) for b in backbones)
    t0 = time.perf_counter()
    done = 0
    all_records: list[GridRecord] = []
    best: dict[str, GridRecord] = {}
    for backbone in backbones:
        records: list[GridRecord] = []
        for i, cfg in enumerate(iter_configs(backbone, epochs=epochs, seed=seed)):
            done += 1
            desc = (f"backbone={backbone} hidden={cfg.hidden} dropout={cfg.dropout} heads={cfg.heads} "
                    f"tau={cfg.tau} lr={cfg.lr} wd={cfg.weight_decay} alpha={cfg.alpha}")
            log(f"[grid] {done}/{total} START  {desc}")
            trainer = GNNTrainer(dataset, graph, embedder, cfg)
            ckpt = ckpt_dir / f"{backbone}_{i}.pt"
            trainer.train(save_best=True, checkpoint_path=ckpt)  # keep best-validation model
            router = GNNRouter.from_checkpoint(ckpt, dataset, graph, embedder)
            comp, mp = score_on_validation(router, val_queries, dataset, k)
            records.append(GridRecord(backbone=backbone, config=vars(cfg), val_completion=comp, val_map=mp))
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (total - done)
            log(f"[grid] {done}/{total} done   {desc}  val_completion={comp:.3f} val_map{k}={mp:.3f}  "
                f"(elapsed {_fmt_dur(elapsed)}, eta {_fmt_dur(eta)})")
        all_records.extend(records)
        chosen = select_best(records)
        if chosen is not None:
            best[backbone] = chosen
            c = chosen.config
            log(f"[grid] backbone={backbone} BEST: hidden={c['hidden']} dropout={c['dropout']} "
                f"heads={c['heads']} tau={c['tau']} lr={c['lr']} wd={c['weight_decay']} alpha={c['alpha']}  "
                f"val_completion={chosen.val_completion:.3f} val_map{k}={chosen.val_map:.3f}")

    # audit log of every config's validation score + the chosen best per backbone
    (out_dir / "grid_log.jsonl").write_text(
        "\n".join(json.dumps({"backbone": r.backbone, "val_completion": r.val_completion,
                              "val_map": r.val_map, "config": r.config}) for r in all_records)
    )
    save_best_configs(best, out_dir / "best_configs.json")
    log(f"[grid] wrote {out_dir / 'best_configs.json'} and {out_dir / 'grid_log.jsonl'}")
    return best, all_records


def save_best_configs(best: dict[str, GridRecord], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {bb: {"config": r.config, "val_completion": r.val_completion, "val_map": r.val_map}
               for bb, r in best.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_best_configs(path: Path) -> dict[str, GNNTrainConfig]:
    """Load ``best_configs.json`` into ``{backbone: GNNTrainConfig}`` (drops any unknown keys defensively)."""
    payload = json.loads(Path(path).read_text())
    fields = set(GNNTrainConfig.__dataclass_fields__)
    return {bb: GNNTrainConfig(**{k: v for k, v in entry["config"].items() if k in fields})
            for bb, entry in payload.items()}


# --------------------------------------------------------------------------- #
# Multi-seed full evaluation (ADR 0028/0029)
# --------------------------------------------------------------------------- #
def train_seeds(
    dataset: Dataset, graph, embedder: Embedder, config: GNNTrainConfig, seeds: list[int], ckpt_dir: Path,
    *, on_progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Train ``config`` once per seed; return the checkpoint paths (mean ± variance material, ADR 0029).

    ``on_progress`` logs per-seed training start/finish (with the best validation InfoNCE loss +
    elapsed); logging only — identical training/checkpoints.
    """
    log = on_progress or _default_log
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seeds:
        cfg = GNNTrainConfig(**{**vars(config), "seed": s})
        trainer = GNNTrainer(dataset, graph, embedder, cfg)
        ckpt = ckpt_dir / f"{cfg.backbone}_seed{s}.pt"
        log(f"[full-eval] gnn_{cfg.backbone} seed {s}: training…")
        t0 = time.perf_counter()
        history = trainer.train(save_best=True, checkpoint_path=ckpt)
        best_val = min(history["val"]) if history["val"] else float("nan")
        log(f"[full-eval] gnn_{cfg.backbone} seed {s}: trained (best val_loss={best_val:.4f}, "
            f"elapsed {_fmt_dur(time.perf_counter() - t0)})")
        paths.append(ckpt)
    return paths


def _mean_std(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(clean), "std": (statistics.pstdev(clean) if len(clean) > 1 else 0.0),
            "n": len(clean)}


def _results_for(router, queries: list[Query], dataset: Dataset) -> list[QueryResult]:
    return [evaluate_query(router, q, dataset) for q in queries]


def run_full_eval(
    dataset: Dataset,
    embedder: Embedder,
    best_configs: dict[str, GNNTrainConfig],
    *,
    seeds: list[int],
    k: int = 10,
    out_dir: Path = EVAL_DIR,
    graph=None,
    limit: int | None = None,
    save: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Full TEST-split evaluation: baselines once, each GNN backbone over ``seeds`` → mean ± std (ADR 0028/0029).

    ``on_progress`` logs per-baseline and per-seed progress (default: flushed stdout); logging only —
    identical routers, splits, metrics, and artifacts.
    """
    log = on_progress or _default_log
    graph = graph if graph is not None else build_graph(dataset)
    cfg = EvalConfig(k=k, seed=(seeds[0] if seeds else 0), limit=limit)
    split = split_queries(len(dataset.queries), seed=cfg.seed)
    test_queries = [dataset.queries[i] for i in split.test]
    if limit is not None:
        test_queries = test_queries[:limit]

    # baselines (no training) — one deterministic run each
    baselines = build_routers(dataset, graph, embedder, gnn_checkpoint=None)
    log(f"[full-eval] routers=baselines{list(baselines)}+gnn{list(best_configs)}  "
        f"seeds={seeds}  test_split={len(test_queries)} queries")
    baseline_reports = {}
    for name, r in baselines.items():
        log(f"[full-eval] eval {name} on test…")
        baseline_reports[name] = _router_report(_results_for(r, test_queries, dataset), cfg)
        log(f"[full-eval] eval {name}: done "
            f"(overall completion={baseline_reports[name]['overall']['completion']['rate']:.3f})")

    # GNN backbones — mean ± std across seeds
    gnn_summary: dict[str, dict] = {}
    ckpt_dir = out_dir / "eval_ckpts"
    for backbone, config in best_configs.items():
        per_seed = []
        for s, ckpt in zip(seeds, train_seeds(dataset, graph, embedder, config, seeds, ckpt_dir, on_progress=log)):
            log(f"[full-eval] gnn_{backbone} seed {s}: eval on test…")
            router = GNNRouter.from_checkpoint(ckpt, dataset, graph, embedder)
            per_seed.append(_router_report(_results_for(router, test_queries, dataset), cfg))
            log(f"[full-eval] gnn_{backbone} seed {s}: eval done "
                f"(overall completion={per_seed[-1]['overall']['completion']['rate']:.3f})")
        gnn_summary[f"gnn_{backbone}"] = {
            "seeds": seeds,
            "overall_completion": _mean_std([r["overall"]["completion"]["rate"] for r in per_seed]),
            "deep_completion": _mean_std([r["slices"][S.DEEP]["completion"]["rate"] for r in per_seed]),
            "deep_transfer_loss": _mean_std([r["slices"][S.DEEP]["transfer_loss"]["conditional"] for r in per_seed]),
            "deep_map": _mean_std([r["slices"][S.DEEP]["retrieval"][f"map@{k}"] for r in per_seed]),
            "per_seed": per_seed,
        }

    comparison = {
        "config": {"k": k, "seeds": seeds, "n_test": len(test_queries),
                   "baselines": list(baseline_reports), "gnn": list(gnn_summary)},
        "baselines": baseline_reports,
        "gnn": gnn_summary,
        "headline_deep_transfer_loss": {
            **{name: rep["slices"][S.DEEP]["transfer_loss"]["conditional"] for name, rep in baseline_reports.items()},
            **{name: g["deep_transfer_loss"] for name, g in gnn_summary.items()},
        },
    }
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "full_eval.json").write_text(json.dumps(comparison, indent=2, default=str))
        (out_dir / "full_eval.txt").write_text(render_full_table(comparison))
        log(f"[full-eval] wrote {out_dir / 'full_eval.json'} and {out_dir / 'full_eval.txt'}")
    log("[full-eval] headline (deep-slice transfer_loss):")
    for name, val in comparison["headline_deep_transfer_loss"].items():
        if isinstance(val, dict):  # GNN mean±std
            m, sd = val.get("mean"), val.get("std")
            log(f"[full-eval]   {name:16s} {'n/a' if m is None else f'{m:.3f}±{sd:.3f}'}")
        else:
            log(f"[full-eval]   {name:16s} {'n/a' if val is None else f'{val:.3f}'}")
    return comparison


def _router_report(results: list[QueryResult], cfg: EvalConfig) -> dict:
    # reuse the harness's per-router aggregation (overall + per-slice metric blocks)
    from mcp_router_eval.eval.harness import _router_report as harness_report
    return harness_report(results, cfg)


def render_full_table(comparison: dict) -> str:
    lines = ["router            overall_compl   deep_transfer_loss"]
    for name, rep in comparison["baselines"].items():
        oc = rep["overall"]["completion"]["rate"]
        tl = rep["slices"][S.DEEP]["transfer_loss"]["conditional"]
        lines.append(f"{name:16s}  {oc:12.3f}   {('n/a' if tl is None else f'{tl:.3f}'):>17s}")
    for name, g in comparison["gnn"].items():
        oc = g["overall_completion"]["mean"]
        tl = g["deep_transfer_loss"]
        oc_s = "n/a" if oc is None else f"{oc:.3f}"
        tl_s = "n/a" if tl["mean"] is None else f"{tl['mean']:.3f}±{tl['std']:.3f}"
        lines.append(f"{name:16s}  {oc_s:>12s}   {tl_s:>17s}")
    return "\n".join(lines)
