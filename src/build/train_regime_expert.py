"""
train_regime_expert.py
----------------------
Regime-Aware Mixture of Experts (MoE) for stock return prediction.

Design principle
----------------
Regimes are precomputed by get_regime.py and treated as **ground truth**.
Routing is EXPLICIT — the regime label, not a learned gate, selects the expert.
This enforces true specialisation: each expert sees only its regime's data and
cannot free-ride on patterns from other regimes.

Architecture
------------

  Input sequence  [T-SEQ_LEN .. T]  (n_features per day)
       │
       ▼
  Shared LSTM encoder ──► hidden state h
       │
  concat[h | current_features] = x
       │
  ┌────────── Routing ──────────────────────────────────────────┐
  │  HARD  : regime_label → Expert_r(x) → logits               │
  │  SOFT  : Σ regime_prob_r × Expert_r(x)  (weighted blend)   │
  │  BLEND : hard if confidence ≥ θ, else soft                  │
  └─────────────────────────────────────────────────────────────┘
       │
       ├─► (optional) + SharedMLP(x)   ← residual from global model
       │
       ▼
  CrossEntropy loss (class-weighted)

Why no gating loss?
-------------------
In train_mixoe.py we had balance / entropy / alignment losses to guide a
learned gate. Here routing is determined entirely by the precomputed regime,
so those losses are unnecessary. CrossEntropy on the task is sufficient.

Routing modes
-------------
  HARD  (default)
    Each sample is routed to exactly one expert based on its regime label.
    Forward pass batches samples by regime for efficiency.
    Gradient flows only through the selected expert.

  SOFT
    All experts see all inputs. Final logit = Σ_r regime_prob[r] × Expert_r(x).
    regime_prob is either one-hot (same as hard, differentiable) or a
    posterior from an external classifier.
    Gradient flows through all experts, scaled by their regime probability.

  BLEND
    Uses hard routing when max(regime_prob) >= CONFIDENCE_THRESHOLD,
    soft routing otherwise. Good when the regime detector is uncertain
    at transition points between regimes.

Shared residual
---------------
An optional global MLP (SharedMLP) processes all samples regardless of regime.
  final_logit = expert_logit + alpha * shared_logit
This lets the model capture cross-regime patterns (e.g. universal volume signals)
while experts handle regime-specific behaviour.
"""

from __future__ import annotations

import os
import sys
import types


# ── Stub transformers (required by vnstock_trade dependency) ──────────────────
def _stub_transformers():
    root        = types.ModuleType("transformers")
    config_utils = types.ModuleType("transformers.configuration_utils")
    class PretrainedConfig: pass
    config_utils.PretrainedConfig = PretrainedConfig
    root.configuration_utils     = config_utils
    sys.modules["transformers"]                      = root
    sys.modules["transformers.configuration_utils"]  = config_utils

_stub_transformers()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_regime import (
    build_conditional_features,
    infer_market_index_from_filename,
    REGIME_FEATURES,
    REGIME_NAME,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "./data/data-vn-20230228/stock-historical-data/VCB-VNINDEX-History.csv"
HORIZON     = 5        # days forward for return label
SEQ_LEN     = 20        # LSTM look-back window
N_EXPERTS   = 4         # one per regime: QUIET_BEAR, PANIC_BEAR, QUIET_BULL, VOLATILE_BULL
N_REGIMES   = 4
HIDDEN_LSTM = 64        # LSTM hidden dim
HIDDEN_MLP  = 64        # expert MLP hidden dim
DROPOUT     = 0.2
EPOCHS      = 80
BATCH_SIZE  = 64
LR          = 1e-5
WEIGHT_DECAY = 1e-4
TRAIN_FRAC  = 0.70
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Routing mode ──────────────────────────────────────────────────────────────
# "hard"  : deterministic — regime label selects one expert
# "soft"  : probabilistic — weighted blend of all experts
# "blend" : hard when confident, soft when uncertain
ROUTING_MODE         = "hard"
CONFIDENCE_THRESHOLD = 0.70   # used only in "blend" mode

# ── Optional shared residual ──────────────────────────────────────────────────
# A global MLP sees every sample regardless of regime.
# final_logit = expert_logit + SHARED_ALPHA * global_logit
USE_SHARED_RESIDUAL = True
SHARED_ALPHA        = 0.3     # how much the shared model contributes

# ── Early stopping ────────────────────────────────────────────────────────────
PATIENCE = 15   # stop if val loss does not improve for this many epochs

# ── Return label thresholds ───────────────────────────────────────────────────
SELL_THRESH = -0.03   # HORIZON-day return < -3% → sell (0)
BUY_THRESH  =  0.05   # HORIZON-day return > +5% → buy  (2)
                      # otherwise              → hold (1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, list[str]]:
    """
    Load CSV, run get_regime pipeline, engineer features, assign labels.
    feature_cols = union of REGIME_FEATURES — every column that any regime
    cares about, validated by domain knowledge in get_regime.py.
    """
    raw = pd.read_csv(data_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])

    market_index = infer_market_index_from_filename(data_path)
    df = build_conditional_features(raw, market_index=market_index)

    # Days elapsed since the current regime started — useful temporal context
    regime_change        = (df["regime"] != df["regime"].shift()).cumsum()
    df["days_in_regime"] = df.groupby(regime_change).cumcount()

    # Forward return
    df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1

    # Asymmetric labels: sell / hold / buy
    df["label"] = 1
    df.loc[df["future_return"] < SELL_THRESH, "label"] = 0
    df.loc[df["future_return"] > BUY_THRESH,  "label"] = 2

    # feature_cols = union of REGIME_FEATURES (ordered, deduplicated)
    seen: set[str] = set()
    ordered: list[str] = []
    for rid in sorted(REGIME_FEATURES):
        for feat in REGIME_FEATURES[rid]:
            if feat not in seen:
                seen.add(feat); ordered.append(feat)
    feature_cols = [f for f in ordered if f in df.columns] + ["days_in_regime"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["label", "future_return"])
    df = df.reset_index(drop=True)

    return df, feature_cols


def print_dataset_stats(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Print regime and label distributions to detect imbalance early."""
    print(f"\nDataset: {len(df)} rows  |  {len(feature_cols)} features")
    rc = df["regime_name"].value_counts()
    print("\nRegime distribution:")
    for name, count in rc.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        # Warn when a regime has < 5% of data — expert will be starved
        warn = "  ⚠ sparse (<5%)" if pct < 5 else ""
        print(f"  {name:<18s}  {count:5d}  ({pct:5.1f}%)  {bar}{warn}")

    lc = df["label"].value_counts().sort_index()
    print("\nLabel distribution (sell=0, hold=1, buy=2):")
    for lbl, count in lc.items():
        print(f"  {['sell','hold','buy'][lbl]}: {count}  ({count/len(df)*100:.1f}%)")


class RegimeDataset(Dataset):
    """
    Returns (seq, current, label, regime) per sample.
    seq     : [SEQ_LEN, n_features]  — LSTM input window
    current : [n_features]           — features at prediction day
    label   : int                    — return class
    regime  : int                    — precomputed regime from get_regime.py
    """

    def __init__(
        self,
        features: np.ndarray,
        labels:   np.ndarray,
        regimes:  np.ndarray,
        seq_len:  int = SEQ_LEN,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)
        self.regimes  = torch.tensor(regimes,  dtype=torch.long)
        self.seq_len  = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        seq     = self.features[idx : idx + self.seq_len]
        current = self.features[idx + self.seq_len - 1]
        label   = self.labels  [idx + self.seq_len - 1]
        regime  = self.regimes [idx + self.seq_len - 1]
        return seq, current, label, regime


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────────────────────────────────────

class SharedLSTMEncoder(nn.Module):
    """
    Encodes the input sequence into a fixed-size hidden state.
    Shared across all experts — one encoding, then route to the right expert.
    Sharing is efficient (avoids N full LSTM passes) and lets all experts
    benefit from the same temporal context.
    """

    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)   # h_n: [1, B, hidden]
        return h_n.squeeze(0)          # [B, hidden]


class ExpertMLP(nn.Module):
    """
    A single regime expert.
    Input: concat([current_features, lstm_hidden]) → logits over n_classes.
    Each expert is structurally identical; differentiation comes from
    seeing only its regime's data during training.
    """

    def __init__(self, n_input: int, hidden: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.LayerNorm(hidden),     # stabilises training with small per-regime batches
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedMLP(nn.Module):
    """
    Optional global model — sees every sample regardless of regime.
    Captures universal signals (e.g. conviction_close works in all regimes).
    Its output is added as a residual to the expert output, scaled by SHARED_ALPHA.
    """

    def __init__(self, n_input: int, hidden: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegimeAwareMoE(nn.Module):
    """
    Regime-Aware Mixture of Experts.

    Routing is determined by the precomputed regime label — not learned.
    Three modes:
      HARD  : one expert per sample, deterministic
      SOFT  : all experts contribute, weighted by regime_probs
      BLEND : hard when confident, soft when uncertain

    Parameters
    ----------
    n_features         : number of input features
    n_experts          : number of experts (= N_REGIMES)
    use_shared_residual: add a global MLP residual to the expert output
    routing_mode       : "hard" | "soft" | "blend"
    confidence_threshold: threshold for BLEND mode
    shared_alpha       : scale of the shared residual
    """

    def __init__(
        self,
        n_features:           int,
        n_experts:            int  = N_EXPERTS,
        hidden_lstm:          int  = HIDDEN_LSTM,
        hidden_mlp:           int  = HIDDEN_MLP,
        n_classes:            int  = 3,
        dropout:              float = DROPOUT,
        use_shared_residual:  bool  = USE_SHARED_RESIDUAL,
        routing_mode:         str   = ROUTING_MODE,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        shared_alpha:         float = SHARED_ALPHA,
    ) -> None:
        super().__init__()
        self.n_experts            = n_experts
        self.n_classes            = n_classes
        self.routing_mode         = routing_mode
        self.confidence_threshold = confidence_threshold
        self.shared_alpha         = shared_alpha

        # Shared LSTM: one encoding pass for the whole batch, regardless of regime
        self.encoder = SharedLSTMEncoder(n_features, hidden_lstm)

        # N regime experts — one per regime
        expert_input = n_features + hidden_lstm
        self.experts = nn.ModuleList([
            ExpertMLP(expert_input, hidden_mlp, n_classes, dropout)
            for _ in range(n_experts)
        ])

        # Optional global residual model
        self.shared_model = (
            SharedMLP(expert_input, hidden_mlp, n_classes, dropout)
            if use_shared_residual else None
        )

    def _encode(self, seq: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Shared encoding: concat(lstm_hidden, current_features)."""
        h = self.encoder(seq)                      # [B, hidden_lstm]
        return torch.cat([current, h], dim=-1)     # [B, n_features + hidden_lstm]

    def _hard_route(
        self, x: torch.Tensor, regime: torch.Tensor
    ) -> torch.Tensor:
        """
        Route each sample to exactly one expert based on regime label.
        Implemented by batching samples per regime → avoids per-sample Python loops.
        Gradient flows only through the selected expert for each sample.
        """
        logits = torch.zeros(x.size(0), self.n_classes, device=x.device)
        for r in range(self.n_experts):
            mask = (regime == r)
            if mask.any():
                logits[mask] = self.experts[r](x[mask])
        return logits

    def _soft_route(
        self, x: torch.Tensor, regime_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Weighted blend of all experts using regime_probs.
        regime_probs: [B, N_EXPERTS] — row-sums to 1.
        Gradient flows through all experts, scaled by their probability weight.
        """
        all_logits = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )                                                  # [B, N_EXPERTS, n_classes]
        weights = regime_probs.unsqueeze(-1)               # [B, N_EXPERTS, 1]
        return (all_logits * weights).sum(dim=1)           # [B, n_classes]

    def forward(
        self,
        seq:          torch.Tensor,                        # [B, SEQ_LEN, n_features]
        current:      torch.Tensor,                        # [B, n_features]
        regime:       torch.Tensor,                        # [B] int regime label
        regime_probs: torch.Tensor | None = None,          # [B, N_EXPERTS] optional
    ) -> torch.Tensor:
        x = self._encode(seq, current)                     # [B, expert_input_dim]

        if self.routing_mode == "hard":
            logits = self._hard_route(x, regime)

        elif self.routing_mode == "soft":
            # If no external probabilities, use one-hot of regime label
            if regime_probs is None:
                regime_probs = torch.zeros(
                    x.size(0), self.n_experts, device=x.device
                ).scatter_(1, regime.unsqueeze(1), 1.0)
            logits = self._soft_route(x, regime_probs)

        else:  # "blend"
            # Hard routing where confident, soft where uncertain
            if regime_probs is None:
                # No external probs → treat all as confident → pure hard
                logits = self._hard_route(x, regime)
            else:
                confidence = regime_probs.max(dim=-1).values   # [B]
                hard_mask  = confidence >= self.confidence_threshold
                soft_mask  = ~hard_mask

                logits = torch.zeros(x.size(0), self.n_classes, device=x.device)
                if hard_mask.any():
                    logits[hard_mask] = self._hard_route(
                        x[hard_mask], regime[hard_mask]
                    )
                if soft_mask.any():
                    logits[soft_mask] = self._soft_route(
                        x[soft_mask], regime_probs[soft_mask]
                    )

        # Optional shared residual: add global model output
        if self.shared_model is not None:
            logits = logits + self.shared_alpha * self.shared_model(x)

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:          RegimeAwareMoE,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    criterion:      nn.Module,
    device:         str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_regimes = [], [], []

    for seq, current, label, regime in loader:
        seq, current, label, regime = (
            seq.to(device), current.to(device), label.to(device), regime.to(device)
        )
        optimizer.zero_grad()
        logits = model(seq, current, regime)
        loss   = criterion(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(label)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_regimes.append(regime.cpu().numpy())

    preds   = np.concatenate(all_preds)
    labels  = np.concatenate(all_labels)
    regimes = np.concatenate(all_regimes)
    acc     = (preds == labels).mean()

    return {
        "loss":    total_loss / len(loader.dataset),
        "acc":     acc,
        "preds":   preds,
        "labels":  labels,
        "regimes": regimes,
    }


@torch.no_grad()
def evaluate(
    model:     RegimeAwareMoE,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> dict[str, float | np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_regimes, all_proba = [], [], [], []

    for seq, current, label, regime in loader:
        seq, current, label, regime = (
            seq.to(device), current.to(device), label.to(device), regime.to(device)
        )
        logits = model(seq, current, regime)
        loss   = criterion(logits, label)
        total_loss += loss.item() * len(label)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_regimes.append(regime.cpu().numpy())
        all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())

    preds   = np.concatenate(all_preds)
    labels  = np.concatenate(all_labels)
    regimes = np.concatenate(all_regimes)
    proba   = np.concatenate(all_proba)

    return {
        "loss":    total_loss / len(loader.dataset),
        "acc":     (preds == labels).mean(),
        "preds":   preds,
        "labels":  labels,
        "regimes": regimes,
        "proba":   proba,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation utilities
# ─────────────────────────────────────────────────────────────────────────────

def print_per_regime_report(
    preds: np.ndarray,
    labels: np.ndarray,
    regimes: np.ndarray,
    n_regimes: int = N_REGIMES,
) -> None:
    """
    Print classification report and per-regime accuracy.
    Key check: each expert should perform best on its own regime.
    If Expert r performs poorly on Regime r → feature mask or data volume issue.
    """
    CLASS_NAMES = ["sell", "hold", "buy"]

    print("\n=== Overall Classification Report ===")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
    print("=== Confusion Matrix (rows=actual, cols=predicted) ===")
    print(confusion_matrix(labels, preds))

    print("\n=== Per-Regime Accuracy ===")
    print(f"  {'Regime':<18s}  {'n':>5}  {'acc':>6}  {'sell_r':>7}  {'hold_r':>7}  {'buy_r':>7}")
    print("  " + "-" * 62)
    for r in range(n_regimes):
        mask = (regimes == r)
        if mask.sum() == 0:
            print(f"  {REGIME_NAME[r]:<18s}  {'no data':>5}")
            continue
        r_preds  = preds[mask]
        r_labels = labels[mask]
        r_acc    = (r_preds == r_labels).mean()

        # per-class recall within this regime
        recalls = []
        for c in range(3):
            c_mask = (r_labels == c)
            recalls.append((r_preds[c_mask] == c).mean() if c_mask.sum() > 0 else float("nan"))

        print(
            f"  {REGIME_NAME[r]:<18s}  {mask.sum():>5d}  {r_acc:>6.3f}"
            f"  {recalls[0]:>7.3f}  {recalls[1]:>7.3f}  {recalls[2]:>7.3f}"
        )


def print_expert_sanity_check(regimes: np.ndarray, n_regimes: int = N_REGIMES) -> None:
    """
    Verify Expert i is used exclusively for Regime i.
    In hard-routing this is definitional, but the check serves as a smoke test
    and helps when comparing against soft/blend runs.
    """
    print("\n=== Expert Usage (which regime each expert handled) ===")
    print(f"  {'Expert':<12s}  {'Regime served':<18s}  {'samples':>8}  {'exclusive?':>10}")
    print("  " + "-" * 58)
    for r in range(n_regimes):
        n = (regimes == r).sum()
        exclusive = "✓ yes" if n > 0 else "✗ no data"
        print(f"  Expert {r:<5d}  {REGIME_NAME[r]:<18s}  {n:>8d}  {exclusive:>10}")


def print_return_bucket_analysis(
    proba: np.ndarray,
    future_returns: np.ndarray,
    n_buckets: int = 5,
) -> None:
    """
    Sort test samples by predicted buy-probability into N buckets.
    Bucket 4 should have the highest mean return (monotonic relationship).
    If not monotonic → model's buy signal is weak or inverted.
    """
    df = pd.DataFrame({
        "proba_buy": proba[:, 2],
        "ret":       future_returns,
    })
    df["bucket"] = pd.qcut(df["proba_buy"], n_buckets, labels=False, duplicates="drop")
    stats = df.groupby("bucket")["ret"].agg(["mean", "std", "count"])
    stats["sharpe"] = stats["mean"] / (stats["std"] + 1e-8)
    print("\n=== Return by Buy-Signal Bucket (0=weakest, 4=strongest) ===")
    print(stats.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:   dict[str, list],
    save_path: str | None = None,
) -> None:
    """
    2×2 grid:
      Top-left  : train loss vs val loss + best-epoch marker
      Top-right : train accuracy vs val accuracy
      Bot-left  : per-regime val accuracy over epochs
      Bot-right : val loss gap (val - train) — overfitting indicator
    """
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Curves — Regime-Aware MoE", fontsize=13, fontweight="bold")

    REGIME_COLORS = {
        "QUIET_BEAR":    "#ef4444",
        "PANIC_BEAR":    "#7f1d1d",
        "QUIET_BULL":    "#22c55e",
        "VOLATILE_BULL": "#f59e0b",
    }

    # ── Train vs val loss ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train", color="#3b82f6", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="val",   color="#ef4444", linewidth=1.5, linestyle="--")
    best_ep = epochs[int(np.argmin(history["val_loss"]))]
    ax.axvline(best_ep, color="#94a3b8", linestyle=":", linewidth=1, label=f"best val (ep {best_ep})")
    ax.set_title("Task Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("CrossEntropy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Train vs val accuracy ─────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], label="train", color="#3b82f6", linewidth=1.5)
    ax.plot(epochs, history["val_acc"],   label="val",   color="#ef4444", linewidth=1.5, linestyle="--")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Per-regime val accuracy ───────────────────────────────────────────────
    ax = axes[1, 0]
    for r in range(N_REGIMES):
        key = f"val_acc_regime_{r}"
        if key in history:
            color = REGIME_COLORS.get(REGIME_NAME.get(r, ""), "#94a3b8")
            ax.plot(epochs, history[key], label=REGIME_NAME.get(r, f"Regime {r}"),
                    color=color, linewidth=1.3)
    ax.set_title("Val Accuracy per Regime"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Overfitting gap ───────────────────────────────────────────────────────
    ax = axes[1, 1]
    gap = [v - t for v, t in zip(history["val_loss"], history["train_loss"])]
    ax.plot(epochs, gap, color="#f59e0b", linewidth=1.5)
    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.fill_between(epochs, 0, gap, where=[g > 0 for g in gap], alpha=0.15, color="#ef4444",
                    label="overfitting region")
    ax.set_title("Overfitting Gap (val − train loss)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss gap")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Debug suggestions
# ─────────────────────────────────────────────────────────────────────────────

def print_debug_suggestions(
    history:   dict[str, list],
    val_result: dict,
    n_regimes: int = N_REGIMES,
) -> None:
    """
    Automated diagnostics. Prints targeted suggestions based on observed symptoms.
    """
    print("\n=== Debug Diagnostics ===")
    suggestions = []

    # Overfitting: val loss grew significantly above train loss
    best_val  = min(history["val_loss"])
    last_val  = history["val_loss"][-1]
    last_train = history["train_loss"][-1]
    gap = last_val - last_train
    if gap > 0.5:
        suggestions.append(
            f"  ⚠ Overfitting: val-train gap = {gap:.3f}. "
            "Try: increase DROPOUT, reduce HIDDEN_LSTM/HIDDEN_MLP, add WEIGHT_DECAY."
        )

    # Label collapse: any class never predicted
    preds  = val_result["preds"]
    labels = val_result["labels"]
    for c, name in enumerate(["sell", "hold", "buy"]):
        if (preds == c).sum() == 0:
            suggestions.append(
                f"  ⚠ Label collapse: model never predicts '{name}'. "
                "Try: increase class weight for this label or check SELL_THRESH/BUY_THRESH."
            )

    # Sparse regime: expert gets < 30 samples in test set
    regimes = val_result["regimes"]
    for r in range(n_regimes):
        n = (regimes == r).sum()
        if n < 30:
            suggestions.append(
                f"  ⚠ Regime {REGIME_NAME[r]} has only {n} test samples. "
                "Expert is undertrained. Consider merging rare regimes or using BLEND mode."
            )

    # Per-regime accuracy much worse than overall
    overall_acc = val_result["acc"]
    for r in range(n_regimes):
        mask = (regimes == r)
        if mask.sum() < 10:
            continue
        r_acc = (preds[mask] == labels[mask]).mean()
        if r_acc < overall_acc - 0.15:
            suggestions.append(
                f"  ⚠ Expert {r} ({REGIME_NAME[r]}) acc={r_acc:.3f} vs overall {overall_acc:.3f}. "
                "This expert is underperforming. Check feature coverage in REGIME_FEATURES."
            )

    # Early stopping fired very early
    best_ep = int(np.argmin(history["val_loss"])) + 1
    if best_ep <= 10:
        suggestions.append(
            f"  ⚠ Best val was at epoch {best_ep} — model diverged quickly. "
            "Try lower LR or check for NaN in features."
        )

    if suggestions:
        for s in suggestions:
            print(s)
    else:
        print("  ✓ No obvious issues detected.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_regime_moe() -> RegimeAwareMoE:
    print(f"Device: {DEVICE}  |  Routing: {ROUTING_MODE}  |  Shared residual: {USE_SHARED_RESIDUAL}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df, feature_cols = build_dataset(DATA_PATH)
    print_dataset_stats(df, feature_cols)

    split_date = df["TradingDate"].quantile(TRAIN_FRAC)
    train_df   = df[df["TradingDate"] <= split_date].reset_index(drop=True)
    test_df    = df[df["TradingDate"] >  split_date].reset_index(drop=True)
    print(f"\nTrain: {len(train_df)}  |  Test: {len(test_df)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test  = scaler.transform(test_df[feature_cols].values)
    y_train = train_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)
    r_train = train_df["regime"].values.astype(int)
    r_test  = test_df["regime"].values.astype(int)

    train_ds = RegimeDataset(X_train, y_train, r_train)
    test_ds  = RegimeDataset(X_test,  y_test,  r_test)

    # WeightedRandomSampler balances regime exposure per batch.
    # Without it, a batch might be 85% QUIET_BEAR and the rare experts
    # (QUIET_BULL, VOLATILE_BULL) get almost no gradient updates per step.
    regime_seq     = r_train[SEQ_LEN - 1:]
    regime_counts  = np.bincount(regime_seq, minlength=N_REGIMES).astype(float)
    sample_weights = 1.0 / (regime_counts[regime_seq] + 1e-8)
    sampler        = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float32),
        num_samples = len(train_ds),
        replacement = True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Class weights: inverse frequency to prevent label collapse ────────────
    label_seq     = y_train[SEQ_LEN - 1:]
    label_counts  = np.bincount(label_seq, minlength=3).astype(float)
    class_weights = torch.tensor(1.0 / (label_counts + 1e-8), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 3   # mean weight = 1
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    print(f"\nClass weights: sell={class_weights[0]:.3f}  hold={class_weights[1]:.3f}  buy={class_weights[2]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    n_features = len(feature_cols)
    model      = RegimeAwareMoE(n_features).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"\n{'Epoch':>5}  {'train_loss':>10}  {'val_loss':>8}  {'train_acc':>9}  {'val_acc':>7}")
    print("-" * 50)

    # ── Training loop with early stopping ─────────────────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        **{f"val_acc_regime_{r}": [] for r in range(N_REGIMES)},
    }

    for epoch in range(1, EPOCHS + 1):
        train_result = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_result   = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        # per-regime val accuracy for the plot
        regime_accs = {}
        for r in range(N_REGIMES):
            mask = (val_result["regimes"] == r)
            if mask.sum() > 0:
                regime_accs[r] = (
                    val_result["preds"][mask] == val_result["labels"][mask]
                ).mean()
            else:
                regime_accs[r] = float("nan")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_result["loss"])
        history["val_loss"].append(val_result["loss"])
        history["train_acc"].append(train_result["acc"])
        history["val_acc"].append(val_result["acc"])
        for r in range(N_REGIMES):
            history[f"val_acc_regime_{r}"].append(regime_accs[r])

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"{epoch:5d}  {train_result['loss']:10.4f}  {val_result['loss']:8.4f}"
                f"  {train_result['acc']:9.3f}  {val_result['acc']:7.3f}"
            )

        # Early stopping
        if val_result["loss"] < best_val_loss:
            best_val_loss  = val_result["loss"]
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)

    # ── Training curves ───────────────────────────────────────────────────────
    plot_training_curves(
        history,
        save_path=os.path.join(os.getcwd(), "data", "regime_moe_training_curves.png"),
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    final = evaluate(model, test_loader, criterion, DEVICE)

    print_per_regime_report(final["preds"], final["labels"], final["regimes"])
    print_expert_sanity_check(final["regimes"])

    # Align future returns with test dataset indices
    test_returns = test_df["future_return"].values[SEQ_LEN - 1:]
    test_returns = test_returns[:len(final["proba"])]
    print_return_bucket_analysis(final["proba"], test_returns)

    # ── Automated debug suggestions ───────────────────────────────────────────
    print_debug_suggestions(history, final)

    # ── Save model ────────────────────────────────────────────────────────────
    out_dir    = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(out_dir, "regime_moe_model.pt")
    torch.save({
        "model_state":  best_state,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "config": {
            "n_features":           n_features,
            "n_experts":            N_EXPERTS,
            "hidden_lstm":          HIDDEN_LSTM,
            "hidden_mlp":           HIDDEN_MLP,
            "routing_mode":         ROUTING_MODE,
            "use_shared_residual":  USE_SHARED_RESIDUAL,
            "seq_len":              SEQ_LEN,
            "horizon":              HORIZON,
        },
    }, model_path)
    print(f"\nModel saved → {model_path}")

    return model


if __name__ == "__main__":
    train_regime_moe()
