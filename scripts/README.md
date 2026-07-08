# Running the evaluation (grid search → full multi-seed eval)

These two scripts run the ADR-0029 tuning and the ADR-0028 full evaluation. **They are meant to be run
by you, on your machine** — they train GNNs and evaluate five routers over the test split, which is
compute you should own. Nothing here was run to completion in the setup session (only a tiny wiring
smoke test).

## Prerequisites (once)

1. **Processed data** present at `data/processed/` — if missing:
   ```
   python scripts/fetch_data.py            # downloads ToolLinkOS raw JSON (pinned commit, ADR 0001)
   python -m mcp_router_eval.data.preprocess   # raw → processed (ADR 0011/0014)
   ```
2. **Embedding cache** — the first run downloads `BAAI/bge-small-en-v1.5` from huggingface.co and caches
   the 573 tool vectors + query vectors under `data/processed/embeddings/` (regenerable, gitignored).
   Needs network on first use only.
3. **Dependencies** — `torch`, `torch-geometric`, `sentence-transformers`, `rank-bm25` already installed
   (they are project dependencies; no new installs needed).

All outputs land in `data/processed/eval/` (gitignored, regenerable).

## Run order (exact commands)

### 1) Grid search — pick the best GNN config per backbone (ADR 0029)
```
python scripts/run_grid_search.py --backbone all --data data/processed --epochs 30 --seed 0
```
Enumerates the discrete grid per backbone (R-GCN/GAT/SAGE), trains each config with best-validation
checkpointing, and selects by **validation `completion_rate`** (mAP@10 tiebreaker) — **validation only,
test untouched**.
Produces:
- `data/processed/eval/best_configs.json` — the chosen config per backbone (input to step 2)
- `data/processed/eval/grid_log.jsonl` — every config's validation score (auditable)

*(Scope it with `--backbone rgcn` to tune one backbone first. Grid sizes: R-GCN/SAGE = 162 configs each,
GAT = 324 — `hidden×dropout(×heads) × τ×lr×weight_decay`.)*

### 2) Full evaluation — five routers on the test split, GNN over multiple seeds (ADR 0028/0029)
```
python scripts/run_full_eval.py --config data/processed/eval/best_configs.json --seeds 5
```
Trains each GNN backbone's best config over 5 seeds (baselines need no training), runs all five routers
{BM25, NaiveRAG, HybridRAG, Traversal, GNN} on the **full test split**, and reports the three metric
groups per depth slice (shallow/medium/deep) with the **deep-slice transfer_loss** as the headline —
GNN as **mean ± std** across seeds.
Produces:
- `data/processed/eval/full_eval.json` — the full comparison (baselines + per-seed GNN + mean±std)
- `data/processed/eval/full_eval.txt` — a readable table

## Rough runtime (CPU)

- **Grid search** (`--backbone all`): order of **minutes–tens of minutes** (hundreds of cheap
  full-batch, 2-layer, 573-node trainings). One backbone is a fraction of that.
- **Full eval** (`--seeds 5`): order of **minutes** (a handful of GNN trainings per backbone + one pass
  of five routers over ~236 test queries).

## Determinism

Both scripts are deterministic given the seed(s). Use the **same `--seed` in step 1** as `seed[0]` in
step 2 so the query-level split matches and the GNN is never tuned/evaluated on training queries
(ADR 0024, no leakage).
"""
