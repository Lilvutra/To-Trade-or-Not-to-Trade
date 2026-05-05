"""
train_regime_expert_.py
-----------------------
Regime-Aware MoE with OHLCV latent augmentation.

Identical to train_regime_expert.py in structure, with one addition:
  A pre-trained OHLCVEncoder (see ohlcv_encoder.py) extracts a latent
  vector from the raw OHLCV sequence for each row.  That latent is
  concatenated to `current_features` before being passed to each expert.

Flow
----
  Raw OHLCV [SEQ_LEN × 5]
       ↓  (frozen OHLCVEncoder)
  latent  [LATENT_DIM]           ← new
       ↓
  concat(current_features, latent)  →  augmented current [n_features + LATENT_DIM]
       ↓
  SharedLSTMEncoder(feature_seq)  →  lstm_hidden
       ↓
  concat(augmented_current, lstm_hidden)  →  ExpertMLP  →  logits

The OHLCV encoder is FROZEN during MoE training — it is a fixed feature
extractor, not jointly optimised.  Pre-train it first with ohlcv_encoder.py.

Quick start
-----------
  # Step 1: pre-train encoder (once)
  python ohlcv_encoder.py

  # Step 2: train augmented MoE
  python train_regime_expert_.py
"""

from __future__ import annotations

import os
import sys
import types


def _stub_transformers():
    root         = types.ModuleType("transformers")
    config_utils = types.ModuleType("transformers.configuration_utils")
    class PretrainedConfig: pass
    config_utils.PretrainedConfig = PretrainedConfig
    root.configuration_utils      = config_utils
    sys.modules["transformers"]                     = root
    sys.modules["transformers.configuration_utils"] = config_utils

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
from ohlcv_encoder import (
    OHLCVEncoder,
    load_ohlcv_encoder,
    extract_latents,
    LATENT_DIM,
    OHLCV_SEQ_LEN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config  (same as train_regime_expert.py; new entries marked with ★)
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH    = "./data/data-vn-20230228/stock-historical-data/VCB-VNINDEX-History.csv"
ENCODER_PATH = "./data/ohlcv_encoder.pt"   # ★ pre-trained OHLCV encoder checkpoint

HORIZON      = 10
SEQ_LEN      = 20
N_EXPERTS    = 7
N_REGIMES    = 4
HIDDEN_LSTM  = 64
HIDDEN_MLP   = 64
DROPOUT      = 0.1
EPOCHS       = 80
BATCH_SIZE   = 64
LR           = 1e-5
WEIGHT_DECAY = 1e-4
TRAIN_FRAC   = 0.70
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

ROUTING_MODE          = "soft"
CONFIDENCE_THRESHOLD  = 0.70
USE_SHARED_RESIDUAL   = True
SHARED_ALPHA          = 0.3
PATIENCE              = 15

SELL_THRESH = -0.03
BUY_THRESH  =  0.05


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_(
    data_path:    str = DATA_PATH,
    encoder_path: str = ENCODER_PATH,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Load CSV, run get_regime pipeline, engineer features, assign labels,
    then append OHLCV latent columns.

    Returns
    -------
    df           : full augmented DataFrame
    feature_cols : engineered feature column names (no latents)
    latent_cols  : latent column names  (lat_0 … lat_{LATENT_DIM-1})
    """
    raw = pd.read_csv(data_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])

    market_index = infer_market_index_from_filename(data_path)
    df = build_conditional_features(raw, market_index=market_index)

    regime_change        = (df["regime"] != df["regime"].shift()).cumsum()
    df["days_in_regime"] = df.groupby(regime_change).cumcount()

    df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
    df["label"] = 1
    df.loc[df["future_return"] < SELL_THRESH, "label"] = 0
    df.loc[df["future_return"] > BUY_THRESH,  "label"] = 2

    seen:    set[str]  = set()
    ordered: list[str] = []
    for rid in sorted(REGIME_FEATURES):
        for feat in REGIME_FEATURES[rid]:
            if feat not in seen:
                seen.add(feat); ordered.append(feat)
    feature_cols = [f for f in ordered if f in df.columns] + ["days_in_regime"]

    # ── OHLCV latents ────────────────────────────────────────────────────────
    latent_cols = [f"lat_{i}" for i in range(LATENT_DIM)]

    if os.path.exists(encoder_path):
        encoder, scaler = load_ohlcv_encoder(encoder_path, device=DEVICE)
        latents = extract_latents(df, encoder, scaler, seq_len=OHLCV_SEQ_LEN, device=DEVICE)
        for i, col in enumerate(latent_cols):
            df[col] = latents[:, i]
        print(f"OHLCV latents extracted  ({LATENT_DIM}d)  from {encoder_path}", flush=True)
    else:
        # Encoder not yet trained — fill with zeros and warn
        print(
            f"WARNING: encoder not found at {encoder_path}. "
            "Latents set to zero. Run `python ohlcv_encoder.py` first."
        )
        for col in latent_cols:
            df[col] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + latent_cols + ["label", "future_return"])
    df = df.reset_index(drop=True)

    return df, feature_cols, latent_cols


def print_dataset_stats(df: pd.DataFrame, feature_cols: list[str]) -> None:
    print(f"\nDataset: {len(df)} rows  |  {len(feature_cols)} features", flush=True)
    rc = df["regime_name"].value_counts()
    print("\nRegime distribution:", flush=True)
    for name, count in rc.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        warn = "  ⚠ sparse (<5%)" if pct < 5 else ""
        print(f"  {name:<18s}  {count:5d}  ({pct:5.1f}%)  {bar}{warn}", flush=True)

    lc = df["label"].value_counts().sort_index()
    print("\nLabel distribution (sell=0, hold=1, buy=2):", flush=True)
    for lbl, count in lc.items():
        print(f"  {['sell','hold','buy'][lbl]}: {count}  ({count/len(df)*100:.1f}%)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDataset_(Dataset):
    """
    Returns (feat_seq, ohlcv_latent, label, regime) per sample.

    feat_seq     : [SEQ_LEN, n_features]   — engineered feature window
    ohlcv_latent : [LATENT_DIM]            — frozen OHLCV latent at prediction day
    label        : int
    regime       : int
    """

    def __init__(
        self,
        features:    np.ndarray,   # [T, n_features]
        latents:     np.ndarray,   # [T, LATENT_DIM]
        labels:      np.ndarray,   # [T]
        regimes:     np.ndarray,   # [T]
        seq_len:     int = SEQ_LEN,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.latents  = torch.tensor(latents,  dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)
        self.regimes  = torch.tensor(regimes,  dtype=torch.long)
        self.seq_len  = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        feat_seq     = self.features[idx : idx + self.seq_len]        # [SEQ_LEN, n_feat]
        ohlcv_latent = self.latents [idx + self.seq_len - 1]          # [LATENT_DIM]
        label        = self.labels  [idx + self.seq_len - 1]
        regime       = self.regimes [idx + self.seq_len - 1]
        return feat_seq, ohlcv_latent, label, regime


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────────────────────────────────────

class SharedLSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        return h_n.squeeze(0)   # [B, hidden]


class ExpertMLP(nn.Module):
    def __init__(self, n_input: int, hidden: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedMLP(nn.Module):
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


class RegimeAwareMoE_(nn.Module):
    """
    Same as RegimeAwareMoE but the expert input includes OHLCV latents.

    expert_input_dim = n_features + HIDDEN_LSTM + LATENT_DIM
                       ^^^^^^^^^^   ^^^^^^^^^^^   ^^^^^^^^^^^
                       engineered   seq context   raw OHLCV
                       features     (LSTM h)      latent
    """

    def __init__(
        self,
        n_features:           int,
        latent_dim:           int   = LATENT_DIM,
        n_experts:            int   = N_EXPERTS,
        hidden_lstm:          int   = HIDDEN_LSTM,
        hidden_mlp:           int   = HIDDEN_MLP,
        n_classes:            int   = 3,
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

        self.encoder = SharedLSTMEncoder(n_features, hidden_lstm)

        # augmented current = current_features + ohlcv_latent
        augmented_current_dim = n_features + latent_dim
        expert_input_dim      = augmented_current_dim + hidden_lstm

        self.experts = nn.ModuleList([
            ExpertMLP(expert_input_dim, hidden_mlp, n_classes, dropout)
            for _ in range(n_experts)
        ])

        self.shared_model = (
            SharedMLP(expert_input_dim, hidden_mlp, n_classes, dropout)
            if use_shared_residual else None
        )

    def _encode(
        self,
        feat_seq:     torch.Tensor,   # [B, SEQ_LEN, n_features]
        current_feat: torch.Tensor,   # [B, n_features]
        ohlcv_latent: torch.Tensor,   # [B, LATENT_DIM]
    ) -> torch.Tensor:
        h   = self.encoder(feat_seq)                                  # [B, hidden_lstm]
        aug = torch.cat([current_feat, ohlcv_latent], dim=-1)        # [B, n_feat+lat]
        return torch.cat([aug, h], dim=-1)                            # [B, expert_input]

    def _hard_route(self, x: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.size(0), self.n_classes, device=x.device)
        for r in range(self.n_experts):
            mask = (regime == r)
            if mask.any():
                logits[mask] = self.experts[r](x[mask])
        return logits

    def _soft_route(self, x: torch.Tensor, regime_probs: torch.Tensor) -> torch.Tensor:
        all_logits = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, C]
        return (all_logits * regime_probs.unsqueeze(-1)).sum(dim=1)     # [B, C]

    def forward(
        self,
        feat_seq:     torch.Tensor,                     # [B, SEQ_LEN, n_features]
        current_feat: torch.Tensor,                     # [B, n_features]
        ohlcv_latent: torch.Tensor,                     # [B, LATENT_DIM]
        regime:       torch.Tensor,                     # [B] int
        regime_probs: torch.Tensor | None = None,       # [B, N_EXPERTS] optional
    ) -> torch.Tensor:
        x = self._encode(feat_seq, current_feat, ohlcv_latent)

        if self.routing_mode == "hard":
            logits = self._hard_route(x, regime)

        elif self.routing_mode == "soft":
            if regime_probs is None:
                regime_probs = torch.zeros(
                    x.size(0), self.n_experts, device=x.device
                ).scatter_(1, regime.unsqueeze(1), 1.0)
            logits = self._soft_route(x, regime_probs)

        else:  # blend
            if regime_probs is None:
                logits = self._hard_route(x, regime)
            else:
                confidence = regime_probs.max(dim=-1).values
                hard_mask  = confidence >= self.confidence_threshold
                logits = torch.zeros(x.size(0), self.n_classes, device=x.device)
                if hard_mask.any():
                    logits[hard_mask] = self._hard_route(x[hard_mask], regime[hard_mask])
                if (~hard_mask).any():
                    logits[~hard_mask] = self._soft_route(x[~hard_mask], regime_probs[~hard_mask])

        if self.shared_model is not None:
            logits = logits + self.shared_alpha * self.shared_model(x)

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_(
    model:     RegimeAwareMoE_,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    str,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_regimes = [], [], []

    for feat_seq, ohlcv_lat, label, regime in loader:
        feat_seq, ohlcv_lat, label, regime = (
            feat_seq.to(device), ohlcv_lat.to(device),
            label.to(device),    regime.to(device),
        )
        current = feat_seq[:, -1, :]         # last step = prediction day features
        optimizer.zero_grad()
        logits = model(feat_seq, current, ohlcv_lat, regime)
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
    return {
        "loss":    total_loss / len(loader.dataset),
        "acc":     (preds == labels).mean(),
        "preds":   preds, "labels": labels, "regimes": regimes,
    }


@torch.no_grad()
def evaluate_(
    model:     RegimeAwareMoE_,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_regimes, all_proba = [], [], [], []

    for feat_seq, ohlcv_lat, label, regime in loader:
        feat_seq, ohlcv_lat, label, regime = (
            feat_seq.to(device), ohlcv_lat.to(device),
            label.to(device),    regime.to(device),
        )
        current = feat_seq[:, -1, :]
        logits  = model(feat_seq, current, ohlcv_lat, regime)
        loss    = criterion(logits, label)
        total_loss += loss.item() * len(label)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_regimes.append(regime.cpu().numpy())
        all_proba.append(torch.softmax(logits, -1).cpu().numpy())

    preds   = np.concatenate(all_preds)
    labels  = np.concatenate(all_labels)
    regimes = np.concatenate(all_regimes)
    proba   = np.concatenate(all_proba)
    return {
        "loss":    total_loss / len(loader.dataset),
        "acc":     (preds == labels).mean(),
        "preds":   preds, "labels": labels,
        "regimes": regimes, "proba": proba,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation utilities  (same as train_regime_expert.py)
# ─────────────────────────────────────────────────────────────────────────────

def print_per_regime_report(
    preds: np.ndarray, labels: np.ndarray, regimes: np.ndarray,
    n_regimes: int = N_REGIMES,
) -> None:
    CLASS_NAMES = ["sell", "hold", "buy"]
    print("\n=== Overall Classification Report ===", flush=True)
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0), flush=True)
    print("=== Confusion Matrix ===", flush=True)
    print(confusion_matrix(labels, preds), flush=True)

    print("\n=== Per-Regime Accuracy ===", flush=True)
    print(f"  {'Regime':<18s}  {'n':>5}  {'acc':>6}  {'sell_r':>7}  {'hold_r':>7}  {'buy_r':>7}", flush=True)
    print("  " + "-" * 62, flush=True)
    for r in range(n_regimes):
        mask = (regimes == r)
        if not mask.any():
            print(f"  {REGIME_NAME[r]:<18s}  {'no data':>5}"); continue
        r_acc = (preds[mask] == labels[mask]).mean()
        recalls = []
        for c in range(3):
            cm = (labels[mask] == c)
            recalls.append((preds[mask][cm] == c).mean() if cm.any() else float("nan"))
        print(
            f"  {REGIME_NAME[r]:<18s}  {mask.sum():>5d}  {r_acc:>6.3f}"
            f"  {recalls[0]:>7.3f}  {recalls[1]:>7.3f}  {recalls[2]:>7.3f}"
        )


def print_return_bucket_analysis(
    proba: np.ndarray, future_returns: np.ndarray, n_buckets: int = 5,
) -> None:
    df = pd.DataFrame({"proba_buy": proba[:, 2], "ret": future_returns})
    df["bucket"] = pd.qcut(df["proba_buy"], n_buckets, labels=False, duplicates="drop")
    stats = df.groupby("bucket")["ret"].agg(["mean", "std", "count"])
    stats["sharpe"] = stats["mean"] / (stats["std"] + 1e-8)
    print("\n=== Return by Buy-Signal Bucket (0=weakest, 4=strongest) ===", flush=True)
    print(stats.round(4).to_string(), flush=True)


def plot_training_curves(history: dict, save_path: str | None = None) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Curves — Regime-Aware MoE + OHLCV Latents", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train", color="#3b82f6", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="val",   color="#ef4444", linewidth=1.5, linestyle="--")
    best_ep = epochs[int(np.argmin(history["val_loss"]))]
    ax.axvline(best_ep, color="#94a3b8", linestyle=":", linewidth=1, label=f"best val (ep {best_ep})")
    ax.set_title("Task Loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], label="train", color="#3b82f6", linewidth=1.5)
    ax.plot(epochs, history["val_acc"],   label="val",   color="#ef4444", linewidth=1.5, linestyle="--")
    ax.set_title("Accuracy"); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    REGIME_COLORS = {
        "QUIET_BEAR": "#ef4444", "PANIC_BEAR": "#7f1d1d",
        "QUIET_BULL": "#22c55e", "VOLATILE_BULL": "#f59e0b",
    }
    ax = axes[1, 0]
    for r in range(N_REGIMES):
        key = f"val_acc_regime_{r}"
        if key in history:
            ax.plot(epochs, history[key], label=REGIME_NAME.get(r, f"R{r}"),
                    color=REGIME_COLORS.get(REGIME_NAME.get(r, ""), "#94a3b8"), linewidth=1.3)
    ax.set_title("Val Acc per Regime"); ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    gap = [v - t for v, t in zip(history["val_loss"], history["train_loss"])]
    ax.plot(epochs, gap, color="#f59e0b", linewidth=1.5)
    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.fill_between(epochs, 0, gap, where=[g > 0 for g in gap], alpha=0.15, color="#ef4444")
    ax.set_title("Overfitting Gap (val − train loss)"); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {save_path}", flush=True)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_regime_moe_() -> RegimeAwareMoE_:
    print(f"Device: {DEVICE}  |  Routing: {ROUTING_MODE}  |  Latent dim: {LATENT_DIM}", flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    df, feature_cols, latent_cols = build_dataset_(DATA_PATH, ENCODER_PATH)
    print_dataset_stats(df, feature_cols)

    split_date = df["TradingDate"].quantile(TRAIN_FRAC)
    train_df   = df[df["TradingDate"] <= split_date].reset_index(drop=True)
    test_df    = df[df["TradingDate"] >  split_date].reset_index(drop=True)
    print(f"\nTrain: {len(train_df)}  |  Test: {len(test_df)}", flush=True)

    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(train_df[feature_cols].values)
    X_test  = feat_scaler.transform(test_df[feature_cols].values)

    # Latents are already standardised by the encoder's scaler; no re-scaling needed
    L_train = train_df[latent_cols].values.astype(np.float32)
    L_test  = test_df[latent_cols].values.astype(np.float32)

    y_train = train_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)
    r_train = train_df["regime"].values.astype(int)
    r_test  = test_df["regime"].values.astype(int)

    train_ds = RegimeDataset_(X_train, L_train, y_train, r_train)
    test_ds  = RegimeDataset_(X_test,  L_test,  y_test,  r_test)

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

    label_seq    = y_train[SEQ_LEN - 1:]
    label_counts = np.bincount(label_seq, minlength=3).astype(float)
    class_weights = torch.tensor(1.0 / (label_counts + 1e-8), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 3
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    print(f"\nClass weights: sell={class_weights[0]:.3f}  hold={class_weights[1]:.3f}  buy={class_weights[2]:.3f}", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_features = len(feature_cols)
    model      = RegimeAwareMoE_(n_features, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (feat={n_features}, lat={LATENT_DIM})", flush=True)
    print(f"\n{'Epoch':>5}  {'train_loss':>10}  {'val_loss':>8}  {'train_acc':>9}  {'val_acc':>7}", flush=True)
    print("-" * 50, flush=True)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        **{f"val_acc_regime_{r}": [] for r in range(N_REGIMES)},
    }

    for epoch in range(1, EPOCHS + 1):
        train_result = train_epoch_(model, train_loader, optimizer, criterion, DEVICE)
        val_result   = evaluate_(model, test_loader, criterion, DEVICE)
        scheduler.step()

        regime_accs = {}
        for r in range(N_REGIMES):
            mask = (val_result["regimes"] == r)
            regime_accs[r] = (
                (val_result["preds"][mask] == val_result["labels"][mask]).mean()
                if mask.any() else float("nan")
            )

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

        if val_result["loss"] < best_val_loss:
            best_val_loss  = val_result["loss"]
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)

    plot_training_curves(
        history,
        save_path=os.path.join(os.getcwd(), "data", "regime_moe_latent_training_curves.png"),
    )

    final = evaluate_(model, test_loader, criterion, DEVICE)
    print_per_regime_report(final["preds"], final["labels"], final["regimes"])

    test_returns = test_df["future_return"].values[SEQ_LEN - 1:]
    test_returns = test_returns[:len(final["proba"])]
    print_return_bucket_analysis(final["proba"], test_returns)

    out_dir    = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(out_dir, "regime_moe_latent_model.pt")
    torch.save({
        "model_state":  best_state,
        "feat_scaler":  feat_scaler,
        "feature_cols": feature_cols,
        "latent_cols":  latent_cols,
        "config": {
            "n_features":          n_features,
            "latent_dim":          LATENT_DIM,
            "n_experts":           N_EXPERTS,
            "hidden_lstm":         HIDDEN_LSTM,
            "hidden_mlp":          HIDDEN_MLP,
            "routing_mode":        ROUTING_MODE,
            "use_shared_residual": USE_SHARED_RESIDUAL,
            "seq_len":             SEQ_LEN,
            "horizon":             HORIZON,
        },
    }, model_path)
    print(f"\nModel saved → {model_path}", flush=True)

    return model


if __name__ == "__main__":
    train_regime_moe_()
