# Full-eval GNN config — grounded representative (not arbitrary)

The comparative full evaluation (`scripts/run_full_eval.py`, ADR 0028/0029) needs one GNN config per
backbone. This document is the **committed provenance** for that config; the runnable JSON lives at
`data/processed/eval/full_eval_gnn_config.json`, which is **gitignored** (`.gitignore:25` — `data/processed/*`),
so it is **local-only** — reproduce it from the block below.

## Why this is the representative (grounded, per `docs/findings-gnn-collapse.md`)

The GNN collapse is **config-invariant**: across all three backbones and the entire searched space —
`hidden × dropout × heads × τ × lr × weight_decay`, the logQ strength `α ∈ {0, 0.5, 1}` (ADR 0031), and
the initial-residual strength `α_res ∈ {0, 0.1, 0.5, 0.8}` (ADR-0025 amendment) — variant-A completion
stays **0.000** (`docs/findings-gnn-collapse.md`). So **no single config is "the best" on completion**;
any trained config reproduces completion ≈ 0. The representative is therefore chosen to be the **honest,
strongest-framing** anchor rather than an arbitrary pick:

- **Hyperparameters = the retrieval-best (val_map-selected) settings** the grid already picked per backbone
  (reused from the prior `best_configs.json`, which selected by the `val_map` tiebreaker because every
  `val_completion ≈ 0`). This makes the full-eval story: *"even the **retrieval-best** GNN collapses on
  completion."*
- **`alpha = 0.0` and `alpha_res = 0.0` forced** — a **plain** GNN with no debiasing (logQ) and no residual.
  This is the baseline GNN that anchors the documented negative result; the debiasing/residual probes
  already showed neither moves completion off 0.

**Not reused as-is from the stale `best_configs.json`:** those entries carried **no** `alpha`/`alpha_res`
keys (pre-dating both ADRs) and a stale, pre-variant-A-gate `val_completion`. Here the hyperparameters are
reused but `α`/`α_res` are set to 0 explicitly, and no stale completion number is copied (completion is
config-invariant 0.000 per the probes).

## The config (`data/processed/eval/full_eval_gnn_config.json`)

Per-backbone hyperparameters (retrieval-best) with `alpha = alpha_res = 0`:

| backbone | hidden | dropout | heads | proj_dim | τ | lr | weight_decay | alpha | alpha_res | epochs | seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rgcn | 128 | 0.0 | 2 | 128 | 0.05 | 1e-3 | 1e-4 | **0.0** | **0.0** | 30 | 0 |
| gat  | 128 | 0.0 | 2 | 128 | 0.05 | 1e-3 | 1e-4 | **0.0** | **0.0** | 30 | 0 |
| sage | 128 | 0.0 | 2 | 128 | 0.05 | 1e-3 | 1e-2 | **0.0** | **0.0** | 30 | 0 |

Schema mirrors `run_full_eval`'s loader (`load_best_configs`, `eval/tuning.py:190-195`): a top-level
`{backbone: {"config": {…GNNTrainConfig fields…}}}` map; sibling keys (`note`) are ignored by the loader.

```json
{
  "rgcn": { "config": { "backbone": "rgcn", "hidden": 128, "dropout": 0.0, "heads": 2, "proj_dim": 128,
    "tau": 0.05, "lr": 0.001, "weight_decay": 0.0001, "alpha": 0.0, "alpha_res": 0.0, "epochs": 30,
    "batch_size": null, "scheduler": "plateau", "grad_clip": null, "gat_warmup_epochs": 0, "seed": 0 } },
  "gat":  { "config": { "backbone": "gat", "hidden": 128, "dropout": 0.0, "heads": 2, "proj_dim": 128,
    "tau": 0.05, "lr": 0.001, "weight_decay": 0.0001, "alpha": 0.0, "alpha_res": 0.0, "epochs": 30,
    "batch_size": null, "scheduler": "plateau", "grad_clip": null, "gat_warmup_epochs": 0, "seed": 0 } },
  "sage": { "config": { "backbone": "sage", "hidden": 128, "dropout": 0.0, "heads": 2, "proj_dim": 128,
    "tau": 0.05, "lr": 0.001, "weight_decay": 0.01, "alpha": 0.0, "alpha_res": 0.0, "epochs": 30,
    "batch_size": null, "scheduler": "plateau", "grad_clip": null, "gat_warmup_epochs": 0, "seed": 0 } }
}
```

## Dry-parse (this session; no training, no eval run)

Loaded exactly as `run_full_eval` does — parsed into 3 valid `GNNTrainConfig` objects, all three encoders
instantiate, and `alpha == alpha_res == 0.0` (so `h0_proj is None` — a plain GNN, no residual module):

```
rgcn : … alpha=0.0 alpha_res=0.0 | encoder=RGCNEncoder h0_proj=None
gat  : … alpha=0.0 alpha_res=0.0 | encoder=GATEncoder  h0_proj=None
sage : … alpha=0.0 alpha_res=0.0 | encoder=SAGEEncoder h0_proj=None
```

## Run command (the USER runs this — not run here)

```
PYTHONPATH=src python scripts/run_full_eval.py --config data/processed/eval/full_eval_gnn_config.json --seeds 5
```

Trains each backbone over 5 seeds (mean ± std, ADR 0029), evaluates all five routers on the **test** split
(ADR 0024/0028), and writes `full_eval.{json,txt}` to the (gitignored) eval dir. The GNN entries are
expected to reproduce completion ≈ 0 (config-invariant collapse); the numbers fill the PLACEHOLDER in
`docs/findings-gnn-collapse.md`.
