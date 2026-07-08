"""GNN training loop — vectorized from the start (ADR 0022 / 0023 / 0024 / 0026).

Stage 2 of the GNN router: **train** the stage-1 encoders (``gnn_models.py``). No ``GNNRouter``
integration (stage 3). Built vectorized per the profiling pass — every choice below is the
*identical-result* fast path (pure compute layout, no accuracy change):

- **R1 — one GNN forward per step.** :meth:`GNNTrainer._score_batch` calls ``node_embeddings`` **once**
  per step to get ``[N, d]`` and the whole query batch shares it (never a forward per query).
- **R3 — batch-embed queries once.** All train/val query texts are embedded up front via
  ``embedder.encode(list)`` into the versioned cache (never per step).
- **R2 — precompute the false-negative mask once.** :func:`build_masks` builds ``[Q, N]`` gold and
  dependency masks at setup (ADR 0023); reused every step, never rebuilt.
- **Scoring is a matmul.** ``scores = q_batch [B,d] @ node [N,d].T → [B,N]`` (same pattern as NaiveRAG's
  ``tool_matrix @ q``); no per-(query,node) loop.

Loss is masked InfoNCE (ADR 0026); optimizer AdamW (ADR 0026); the split is query-level and
transductive with train-only fitting (ADR 0024). Deterministic given a seed. Checkpoints go to
``data/processed/gnn_checkpoints/`` (regenerable, gitignored).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mcp_router_eval.contract_layer.invariants import Dep
from mcp_router_eval.contracts import ORDERING_RELATIONS
from mcp_router_eval.data.graph_build import ToolGraph
from mcp_router_eval.data.loader import Dataset, Query
from mcp_router_eval.embedding.base import Embedder
from mcp_router_eval.routers.gnn_models import (
    DEFAULT_DROPOUT,
    DEFAULT_HEADS,
    DEFAULT_HIDDEN,
    DEFAULT_PROJ_DIM,
    GATEncoder,
    GNNScorer,
    RGCNEncoder,
    SAGEEncoder,
    node_feature_matrix,
)

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_DIR = _PKG_ROOT / "data" / "processed" / "gnn_checkpoints"  # gitignored (data/processed/*)

_BACKBONES = {"rgcn": RGCNEncoder, "gat": GATEncoder, "sage": SAGEEncoder}

__all__ = [
    "CHECKPOINT_DIR",
    "Split",
    "GNNTrainConfig",
    "GNNTrainer",
    "split_queries",
    "build_masks",
    "train_log_q",
    "masked_infonce",
    "build_scorer",
    "set_seed",
]


def set_seed(seed: int) -> None:
    """Seed torch / numpy / python so a run is reproducible (CPU-deterministic here)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# Query-level split (ADR 0024) — no query in two splits; graph shared (transductive)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    train: list[int]
    val: list[int]
    test: list[int]
    seed: int

    def as_dict(self) -> dict:
        return {"train": self.train, "val": self.val, "test": self.test, "seed": self.seed}


def split_queries(n: int, *, seed: int = 0, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Split:
    """Partition ``range(n)`` query indices into train/val/test (deterministic; no overlap)."""
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train = sorted(perm[:n_train].tolist())
    val = sorted(perm[n_train : n_train + n_val].tolist())
    test = sorted(perm[n_train + n_val :].tolist())
    return Split(train=train, val=val, test=test, seed=seed)


# --------------------------------------------------------------------------- #
# Masks (ADR 0023) — precomputed ONCE; reused every step
# --------------------------------------------------------------------------- #
def build_masks(
    queries: list[Query],
    tool_index: dict[str, int],
    tool_deps: dict[str, list[Dep]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(gold, dep)`` boolean ``[len(queries), N]`` masks.

    ``gold[q, t]`` = tool ``t`` is a gold tool of query ``q`` (the positives). ``dep[q, t]`` = tool
    ``t`` is a ``PARAMETER_*`` dependency of one of query ``q``'s gold tools — the false negatives to
    exclude from the negative pool (ADR 0023). Built once at setup (the profiling-confirmed cheap path).
    """
    n = len(tool_index)
    gold = torch.zeros(len(queries), n, dtype=torch.bool)
    dep = torch.zeros(len(queries), n, dtype=torch.bool)
    for qi, q in enumerate(queries):
        for g in q.required_tools:
            gi = tool_index.get(g)
            if gi is not None:
                gold[qi, gi] = True
            for d in tool_deps.get(g, ()):
                if d.relation in ORDERING_RELATIONS:
                    di = tool_index.get(d.source)
                    if di is not None:
                        dep[qi, di] = True
    return gold, dep


# --------------------------------------------------------------------------- #
# Popularity correction (ADR 0031 amendment) — TRAIN-only log Q(t), applied in the training logits only
# --------------------------------------------------------------------------- #
def train_log_q(gold_train: torch.Tensor) -> torch.Tensor:
    """Add-1-smoothed log popularity ``log Q(t)`` over the **TRAIN** gold mask ``[n_train, N]`` → ``[N]``.

    ``Q(t) = (train gold count of t + 1) / (Σ_t counts + N)`` — a proper distribution over tools (sums to
    1), Laplace-smoothed so a never-gold tool still has a finite ``log Q``. **Train-only** (ADR 0024): the
    caller passes the *train* gold mask, so no val/test gold ever enters. Aligned to node order (the mask's
    columns are the graph node index). The additive normalizer ``Σ counts + N`` is a constant across tools
    and **cancels in InfoNCE's logsumexp**, so only the *relative* frequencies drive the correction.
    """
    counts = gold_train.sum(dim=0).to(torch.float)         # [N] train gold counts, in node order
    q = (counts + 1.0) / (counts.sum() + counts.numel())   # add-1 smoothed; Σ_t q = 1, all q > 0
    return torch.log(q)


# --------------------------------------------------------------------------- #
# Masked InfoNCE (ADR 0026) — vectorized, multi-positive
# --------------------------------------------------------------------------- #
def masked_infonce(
    scores: torch.Tensor,
    gold: torch.Tensor,
    dep: torch.Tensor,
    tau: float,
    *,
    log_q: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Multi-positive InfoNCE over ``[B, N]`` cosine scores with a false-negative mask.

    For each query the denominator is every tool that is gold **or** not a masked dependency (the
    positives plus the true negatives); the numerator is the query's gold tools. False-negative
    dependencies (``dep & ~gold``) are removed from the denominator so they are neither positive nor
    negative. ``loss = logsumexp(valid) − logsumexp(gold)`` averaged over queries that have gold.

    **Popularity correction (ADR 0031 amendment).** When ``alpha != 0`` and ``log_q`` is given, the
    training logit is ``cos(q,t)/τ − α·log Q(t)`` — one broadcasted subtraction of ``[N]`` over ``[B, N]``
    that hits the positive *and* the in-batch negatives alike (the whole row). ``alpha == 0`` (or
    ``log_q is None``) leaves the logits **exactly** as the pre-correction baseline. This is a
    **training-time** device only; inference (``GNNRouter.rank``) scores plain cosine with no such term.
    """
    logits = scores / tau
    if log_q is not None and alpha != 0.0:
        logits = logits - alpha * log_q         # −α·log Q(t), broadcast [N] over [B, N] (ADR 0031 amend.)
    invalid = dep & ~gold                       # false negatives → drop from the denominator
    denom = logits.masked_fill(invalid, float("-inf"))
    numer = logits.masked_fill(~gold, float("-inf"))
    has_gold = gold.any(dim=1)
    per_query = torch.logsumexp(denom, dim=1) - torch.logsumexp(numer, dim=1)
    return per_query[has_gold].mean()


# --------------------------------------------------------------------------- #
# Config + trainer
# --------------------------------------------------------------------------- #
@dataclass
class GNNTrainConfig:
    backbone: str = "rgcn"                 # rgcn | gat | sage (ADR 0010)
    hidden: int = DEFAULT_HIDDEN           # ADR 0025 search {32,64,128}
    dropout: float = DEFAULT_DROPOUT       # ADR 0025 search {0.0,0.3,0.5}
    heads: int = DEFAULT_HEADS             # GAT only (ADR 0025 search {2,4})
    alpha_res: float = 0.0                 # GCNII initial-residual strength (ADR 0025 amendment;
                                           # 0 = off/baseline; a future grid axis — probe first, ADR 0029)
    proj_dim: int | None = DEFAULT_PROJ_DIM
    tau: float = 0.1                       # InfoNCE temperature (ADR 0024/0025 validation-tuned)
    alpha: float = 0.0                     # logQ popularity-correction strength (ADR 0031 amendment;
                                           # 0 = off/baseline, 1 = standard −log Q; a grid axis, ADR 0029)
    lr: float = 1e-3                       # AdamW default (ADR 0026)
    weight_decay: float = 1e-4             # ADR 0026 search 1e-4..1e-2
    epochs: int = 30
    batch_size: int | None = None          # None => full-batch train (one step/epoch)
    scheduler: str | None = "plateau"      # plateau | cosine | None (ADR 0026)
    grad_clip: float | None = None         # optional stability (ADR 0026)
    gat_warmup_epochs: int = 0             # GAT-only opt-in warm-up, off by default (ADR 0026)
    seed: int = 0


def build_scorer(config: GNNTrainConfig, in_dim: int, query_dim: int) -> GNNScorer:
    """Build the backbone encoder + late-cosine scorer from a config (shared by the trainer + router).

    The single place a ``GNNScorer`` is constructed from a :class:`GNNTrainConfig`, so a checkpoint's
    saved config reconstructs the identical architecture before ``load_state_dict`` (ADR 0025/0022).
    """
    cls = _BACKBONES[config.backbone]
    kw: dict = {"hidden": config.hidden, "dropout": config.dropout, "alpha_res": config.alpha_res}
    if config.backbone == "gat":
        kw["heads"] = config.heads
    encoder = cls(in_dim, **kw)
    return GNNScorer(encoder, query_dim=query_dim, proj_dim=config.proj_dim)


@dataclass
class GNNTrainer:
    """Trains one backbone on the query-level split; vectorized (one forward/step, matmul scoring)."""

    dataset: Dataset
    graph: ToolGraph
    embedder: Embedder
    config: GNNTrainConfig = field(default_factory=GNNTrainConfig)

    def __post_init__(self) -> None:
        set_seed(self.config.seed)
        # --- static graph tensors (query-independent) ---
        self._x = node_feature_matrix(self.graph, self.dataset, self.embedder)
        self._edge_index = self.graph.data.edge_index
        self._edge_type = self.graph.data.edge_type
        self._tool_index = dict(self.graph.id_to_index)

        # --- query-level split (ADR 0024); fit NOTHING on val/test ---
        self.split = split_queries(len(self.dataset.queries), seed=self.config.seed)
        train_q = [self.dataset.queries[i] for i in self.split.train]
        val_q = [self.dataset.queries[i] for i in self.split.val]

        # --- R3: batch-embed query texts ONCE (train, then val); never per step ---
        self._n_query_embed_calls = 0
        self._q_train = self._embed_queries(train_q)   # [n_train, query_dim]
        self._q_val = self._embed_queries(val_q)       # [n_val, query_dim]

        # --- R2: precompute masks ONCE (train + val) ---
        deps = self.dataset.tool_deps
        self._gold_train, self._dep_train = build_masks(train_q, self._tool_index, deps)
        self._gold_val, self._dep_val = build_masks(val_q, self._tool_index, deps)

        # --- TRAIN-only popularity log Q(t) for the ADR-0031 logQ correction (never fit on val/test) ---
        self._log_q = train_log_q(self._gold_train)   # [N], aligned to node order (ADR 0024/0031)

        # --- model + optimizer (ADR 0026) ---
        query_dim = self._q_train.shape[1]
        self.scorer = self._build_scorer(query_dim)
        self.optimizer = torch.optim.AdamW(
            self.scorer.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.scheduler = self._build_scheduler()

        #: counts GNN node-forward calls — one per *step*, never per query (profiling R1).
        self.node_forward_count = 0
        self.steps = 0

    # ---- setup helpers ---------------------------------------------------- #
    def _embed_queries(self, queries: list[Query]) -> torch.Tensor:
        self._n_query_embed_calls += 1
        vecs = self.embedder.encode([q.query_text for q in queries])  # one batched call, cached
        return torch.as_tensor(vecs, dtype=torch.float)

    def _build_scorer(self, query_dim: int) -> GNNScorer:
        return build_scorer(self.config, self._x.shape[1], query_dim)

    def _build_scheduler(self):
        if self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        return None

    # ---- scoring: ONE forward per step, matmul over the batch (R1 + scoring) ---- #
    def _score_batch(self, q_batch: torch.Tensor) -> torch.Tensor:
        node = self.scorer.node_embeddings(self._x, self._edge_index, self._edge_type)  # [N, d] — 1 fwd
        self.node_forward_count += 1
        q = self.scorer.query_embedding(q_batch)  # [B, d]
        return q @ node.T                          # [B, N] cosine matrix (matmul, no loop)

    def _loss_on(self, q: torch.Tensor, gold: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
        # Training-time logQ correction (ADR 0031 amendment): −α·log Q on the logits. alpha=0 → baseline.
        return masked_infonce(
            self._score_batch(q), gold, dep, self.config.tau,
            log_q=self._log_q, alpha=self.config.alpha,
        )

    def _batches(self, n: int):
        bs = self.config.batch_size or n
        order = torch.randperm(n)  # seeded via set_seed; deterministic per run
        for start in range(0, n, bs):
            yield order[start : start + bs]

    # ---- training loop ---------------------------------------------------- #
    def train(self, *, save_best: bool = False, checkpoint_path: Path | None = None) -> dict:
        history = {"train": [], "val": []}
        best_val = float("inf")
        for _ in range(self.config.epochs):
            self.scorer.train()
            epoch_losses = []
            for idx in self._batches(self._q_train.shape[0]):
                self.optimizer.zero_grad()
                loss = self._loss_on(self._q_train[idx], self._gold_train[idx], self._dep_train[idx])
                loss.backward()
                if self.config.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.scorer.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.steps += 1
                epoch_losses.append(float(loss.detach()))
            train_loss = float(np.mean(epoch_losses))

            self.scorer.eval()
            with torch.no_grad():
                val_loss = float(self._loss_on(self._q_val, self._gold_val, self._dep_val))
            history["train"].append(train_loss)
            history["val"].append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step(val_loss) if self.config.scheduler == "plateau" else self.scheduler.step()
            if save_best and val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(checkpoint_path or (CHECKPOINT_DIR / f"{self.config.backbone}_best.pt"))
        return history

    # ---- checkpointing (gitignored dir) ----------------------------------- #
    def save_checkpoint(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.scorer.state_dict(), "config": vars(self.config)}, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(Path(path), weights_only=False)
        self.scorer.load_state_dict(ckpt["state_dict"])
