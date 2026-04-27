"""
train_dynamic_factor.py
-----------------------
Dynamic Factor Weighting model for regime-conditional stock return prediction.

Concept
-------
Instead of routing each sample to a separate expert (MoE), this model uses
ONE shared prediction network for all samples, but reweights the input features
differently for each regime.

In quantitative finance, this mirrors how factor models work:
  - A momentum factor matters a lot in VOLATILE_BULL, less in PANIC_BEAR
  - A mean-reversion factor dominates in QUIET_BEAR, is noise in QUIET_BULL
  - The same feature can be a buy signal in one regime and a sell signal in another

The model learns a [N_REGIMES × N_FEATURES] weight matrix where entry [r, i]
represents how much feature i matters when regime r is active.
Weights are initialized from REGIME_FEATURES (domain knowledge), then refined
by training — so the model starts with the right prior and adapts from there.

Architecture
------------

  Input sequence [T-SEQ_LEN .. T]
         │
         ▼
  SharedLSTM ──► hidden state h  [B, hidden_lstm]
         │
         │   regime r ──► FactorWeightMatrix[r]  ← [N_REGIMES, N_FEATURES] learned
         │                        │
         │           sigmoid(w_r) = soft mask ∈ (0,1) per feature
         │                        │
         │  current_features ────► element-wise ×  ← regime reweights each feature
         │                        │
         └──── concat[reweighted_features | h] ────► SharedMLP ──► logits
                                                                       │
                                                               CrossEntropy (class-weighted)
                                                               + L1 sparsity on factor weights

vs MoE
------
  MoE              : N separate networks, route samples to one network
  DynamicFactor    : 1 shared network, reweight features before it sees them

  MoE is more expressive (each expert can learn completely different functions).
  DynamicFactor is more interpretable: after training you can read the weight
  matrix and see exactly which features the model learned to use per regime.
  It also trains more stably because all weights get gradient signal from
  every sample (not just the samples routed to that expert).

Initialization from REGIME_FEATURES
------------------------------------
  factor_weights[r, i] = +2.0  if feature_i ∈ REGIME_FEATURES[r]
                        = -1.0  otherwise
  sigmoid(+2.0) ≈ 0.88  →  feature is active
  sigmoid(-1.0) ≈ 0.27  →  feature is suppressed but not zeroed

  This encodes the domain knowledge from get_regime.py as a prior.
  Training can move weights in either direction — the prior is soft, not fixed.
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
    root.configuration_utils     = config_utils
    sys.modules["transformers"]                     = root
    sys.modules["transformers.configuration_utils"] = config_utils

_stub_transformers()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

DATA_PATH    = "./data/data-vn-20230228/stock-historical-data/VCB-VNINDEX-History.csv"
HORIZON      = 10
SEQ_LEN      = 20
N_REGIMES    = 4
HIDDEN_LSTM  = 64
HIDDEN_MLP   = 64
DROPOUT      = 0.4
EPOCHS       = 80
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_FRAC   = 0.70
PATIENCE     = 15
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

# Factor weight regularization: L1 encourages the model to zero out
# irrelevant features rather than keeping them all at small values.
LAMBDA_L1 = 1e-3

# Label thresholds
SELL_THRESH = -0.03
BUY_THRESH  =  0.05

# Weight init values for factor matrix
ACTIVE_INIT   =  2.0   # sigmoid(2.0) ≈ 0.88 — feature is "on"
INACTIVE_INIT = -1.0   # sigmoid(-1.0) ≈ 0.27 — feature is suppressed


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, list[str]]:
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


class RegimeDataset(Dataset):
    def __init__(self, features, labels, regimes, seq_len=SEQ_LEN):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)
        self.regimes  = torch.tensor(regimes,  dtype=torch.long)
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        seq     = self.features[idx : idx + self.seq_len]
        current = self.features[idx + self.seq_len - 1]
        label   = self.labels  [idx + self.seq_len - 1]
        regime  = self.regimes [idx + self.seq_len - 1]
        return seq, current, label, regime


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────────────────────────────────────

class DynamicFactorModel(nn.Module):
    """
    Dynamic Factor Weighting model.

    Core idea: one shared LSTM + one shared MLP, but feature importance
    is regime-dependent. The factor_weights matrix is the heart of the model:

      factor_weights : nn.Parameter  [N_REGIMES, N_FEATURES]
        Initialized from REGIME_FEATURES (domain prior).
        sigmoid(factor_weights[r]) gives a per-feature soft mask ∈ (0,1)
        for regime r.  Features in REGIME_FEATURES[r] start at 0.88,
        others at 0.27.

    Interpretability
    ----------------
    After training, inspect factor_weights to understand what the model learned:
      - Did PANIC_BEAR's weight on volume_spike stay high? (expected)
      - Did QUIET_BULL's weight on momentum increase? (possible new finding)
      - Did any "inactive" feature get promoted? (data-driven discovery)
    """

    def __init__(
        self,
        n_features:   int,
        feature_cols: list[str],
        n_regimes:    int = N_REGIMES,
        hidden_lstm:  int = HIDDEN_LSTM,
        hidden_mlp:   int = HIDDEN_MLP,
        n_classes:    int = 3,
        dropout:      float = DROPOUT,
    ) -> None:
        super().__init__()
        self.n_features  = n_features
        self.n_regimes   = n_regimes
        self.feature_cols = feature_cols

        # Shared LSTM: one temporal encoder for all regimes
        self.lstm = nn.LSTM(n_features, hidden_lstm, batch_first=True)

        # ── Factor weight matrix ──────────────────────────────────────────────
        # Initialize from REGIME_FEATURES:
        #   features in REGIME_FEATURES[r] → ACTIVE_INIT  (sigmoid ≈ 0.88)
        #   all others                     → INACTIVE_INIT (sigmoid ≈ 0.27)
        # This encodes domain knowledge as a starting point.
        # Training adjusts from here — the prior is informative but not fixed.
        init = torch.full((n_regimes, n_features), INACTIVE_INIT)
        for r in range(n_regimes):
            active_feats = set(REGIME_FEATURES.get(r, []))
            for i, col in enumerate(feature_cols):
                if col in active_feats:
                    init[r, i] = ACTIVE_INIT
        self.factor_weights = nn.Parameter(init)

        # Shared prediction head — same network for all regimes
        # Input: reweighted features + LSTM hidden state
        self.head = nn.Sequential(
            nn.Linear(n_features + hidden_lstm, hidden_mlp),
            nn.LayerNorm(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, n_classes),
        )

    def forward(
        self,
        seq:    torch.Tensor,   # [B, SEQ_LEN, N_FEATURES]
        current: torch.Tensor,  # [B, N_FEATURES]
        regime: torch.Tensor,   # [B] int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ── LSTM encoding ─────────────────────────────────────────────────────
        _, (h_n, _) = self.lstm(seq)
        h = h_n.squeeze(0)                             # [B, hidden_lstm]

        # ── Regime-conditioned feature reweighting ────────────────────────────
        # factor_weights[regime] selects the weight row for each sample's regime
        w = torch.sigmoid(self.factor_weights[regime])  # [B, N_FEATURES]

        # Element-wise multiply: suppresses features irrelevant to this regime
        weighted = current * w                          # [B, N_FEATURES]

        # ── Shared head ───────────────────────────────────────────────────────
        x      = torch.cat([weighted, h], dim=-1)      # [B, N_FEATURES + hidden_lstm]
        logits = self.head(x)                          # [B, n_classes]

        return logits, w   # return weights for analysis


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     DynamicFactorModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    str,
) -> dict:
    model.train()
    total_loss = total_task = total_l1 = 0.0
    all_preds, all_labels, all_regimes = [], [], []

    for seq, current, label, regime in loader:
        seq, current, label, regime = (
            seq.to(device), current.to(device), label.to(device), regime.to(device)
        )
        optimizer.zero_grad()
        logits, _ = model(seq, current, regime)

        task_loss = criterion(logits, label)

        # L1 on sigmoid(factor_weights): pushes small weights toward zero.
        # Encourages the model to truly ignore irrelevant features rather than
        # keeping them at a small non-zero value that still affects predictions.
        l1_loss = torch.sigmoid(model.factor_weights).abs().mean()

        loss = task_loss + LAMBDA_L1 * l1_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = len(label)
        total_task += task_loss.item() * n
        total_l1   += l1_loss.item()   * n
        total_loss += loss.item()      * n
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_regimes.append(regime.cpu().numpy())

    N = len(loader.dataset)
    return {
        "loss":      total_loss / N,
        "task_loss": total_task / N,
        "l1_loss":   total_l1  / N,
        "acc":       (np.concatenate(all_preds) == np.concatenate(all_labels)).mean(),
        "preds":     np.concatenate(all_preds),
        "labels":    np.concatenate(all_labels),
        "regimes":   np.concatenate(all_regimes),
    }


@torch.no_grad()
def evaluate(
    model:     DynamicFactorModel,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_regimes, all_proba = [], [], [], []

    for seq, current, label, regime in loader:
        seq, current, label, regime = (
            seq.to(device), current.to(device), label.to(device), regime.to(device)
        )
        logits, _ = model(seq, current, regime)
        total_loss += criterion(logits, label).item() * len(label)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_regimes.append(regime.cpu().numpy())
        all_proba.append(torch.softmax(logits, -1).cpu().numpy())

    preds   = np.concatenate(all_preds)
    labels  = np.concatenate(all_labels)
    regimes = np.concatenate(all_regimes)
    return {
        "loss":    total_loss / len(loader.dataset),
        "acc":     (preds == labels).mean(),
        "preds":   preds,
        "labels":  labels,
        "regimes": regimes,
        "proba":   np.concatenate(all_proba),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation & interpretability
# ─────────────────────────────────────────────────────────────────────────────

def print_per_regime_report(preds, labels, regimes) -> None:
    CLASS_NAMES = ["sell", "hold", "buy"]
    print("\n=== Overall Classification Report ===")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(labels, preds))

    print(f"\n=== Per-Regime Accuracy ===")
    print(f"  {'Regime':<18}  {'n':>5}  {'acc':>6}  {'sell_r':>7}  {'hold_r':>7}  {'buy_r':>7}")
    print(f"  {'─'*58}")
    for r in range(N_REGIMES):
        mask = regimes == r
        if mask.sum() == 0:
            continue
        r_acc = (preds[mask] == labels[mask]).mean()
        recalls = []
        for c in range(3):
            cm = labels[mask] == c
            recalls.append((preds[mask][cm] == c).mean() if cm.sum() > 0 else float("nan"))
        print(
            f"  {REGIME_NAME[r]:<18}  {mask.sum():>5d}  {r_acc:>6.3f}"
            f"  {recalls[0]:>7.3f}  {recalls[1]:>7.3f}  {recalls[2]:>7.3f}"
        )


def print_factor_weights(model: DynamicFactorModel, top_k: int = 6) -> None:
    """
    Print the learned factor weights per regime.
    High weight (→1.0) = feature is important in this regime.
    Low weight (→0.0)  = feature is suppressed.

    This is the key interpretability output — compare against REGIME_FEATURES
    to see whether the model confirmed domain expectations or discovered
    something unexpected.
    """
    weights = torch.sigmoid(model.factor_weights).detach().cpu().numpy()
    feature_cols = model.feature_cols

    print(f"\n=== Learned Factor Weights (sigmoid) — top {top_k} per regime ===")
    print(f"  Init: active features ≈ {torch.sigmoid(torch.tensor(ACTIVE_INIT)):.2f}, "
          f"inactive ≈ {torch.sigmoid(torch.tensor(INACTIVE_INIT)):.2f}")

    for r in range(N_REGIMES):
        w = weights[r]
        top_idx = w.argsort()[::-1][:top_k]
        top     = [(feature_cols[i], round(float(w[i]), 3)) for i in top_idx]
        domain  = set(REGIME_FEATURES.get(r, []))

        print(f"\n  Regime {r} — {REGIME_NAME[r]}")
        for feat, val in top:
            tag = "✓" if feat in domain else "★ unexpected"
            bar = "█" * int(val * 20)
            print(f"    {feat:<28s}  {val:.3f}  {bar:<20}  {tag}")


def plot_factor_heatmap(
    model:     DynamicFactorModel,
    save_path: str | None = None,
) -> None:
    """
    Heatmap of [N_REGIMES × N_FEATURES] factor weight matrix.
    Rows = regimes, columns = features.
    Colour: dark green = high weight (feature matters), white = suppressed.
    Columns with a dot (·) are in REGIME_FEATURES for that regime (domain prior).
    """
    weights = torch.sigmoid(model.factor_weights).detach().cpu().numpy()
    feature_cols = model.feature_cols
    regime_labels = [REGIME_NAME[r] for r in range(N_REGIMES)]

    fig, ax = plt.subplots(figsize=(max(14, len(feature_cols) * 0.45), 4))
    im = ax.imshow(weights, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(N_REGIMES))
    ax.set_yticklabels(regime_labels, fontsize=9)

    # Mark cells where the domain prior says this feature should be active
    for r in range(N_REGIMES):
        domain = set(REGIME_FEATURES.get(r, []))
        for i, col in enumerate(feature_cols):
            val = weights[r, i]
            # Annotate with weight value
            ax.text(i, r, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if val > 0.65 else "#1e293b")
            # Small dot in corner for domain-expected features
            if col in domain:
                ax.text(i + 0.38, r - 0.35, "·", ha="center", va="center",
                        fontsize=12, color="#7f1d1d", fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Factor weight (sigmoid)")
    ax.set_title(
        "Learned Factor Weights by Regime  (· = expected by REGIME_FEATURES domain prior)",
        fontsize=10, fontweight="bold", pad=10,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Factor heatmap saved → {save_path}")
    else:
        plt.show()


def plot_training_curves(history: dict, save_path: str | None = None) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Curves — Dynamic Factor Model", fontsize=12, fontweight="bold")

    # Task loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train", color="#3b82f6")
    ax.plot(epochs, history["val_loss"],   label="val",   color="#ef4444", linestyle="--")
    best_ep = epochs[int(np.argmin(history["val_loss"]))]
    ax.axvline(best_ep, color="#94a3b8", linestyle=":", label=f"best ep={best_ep}")
    ax.set_title("Task Loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="train", color="#3b82f6")
    ax.plot(epochs, history["val_acc"],   label="val",   color="#ef4444", linestyle="--")
    ax.set_title("Accuracy"); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Per-regime val accuracy
    ax = axes[2]
    COLORS = {"QUIET_BEAR": "#ef4444", "PANIC_BEAR": "#7f1d1d",
              "QUIET_BULL": "#22c55e", "VOLATILE_BULL": "#f59e0b"}
    for r in range(N_REGIMES):
        key = f"val_acc_r{r}"
        if key in history:
            ax.plot(epochs, history[key], label=REGIME_NAME[r],
                    color=COLORS.get(REGIME_NAME[r], "#94a3b8"), linewidth=1.3)
    ax.set_title("Val Accuracy per Regime"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_dynamic_factor() -> DynamicFactorModel:
    print(f"Device: {DEVICE}")

    df, feature_cols = build_dataset(DATA_PATH)
    n_features = len(feature_cols)
    print(f"Rows: {len(df)}  |  Features: {n_features}")
    print(f"Regime distribution:\n{df['regime_name'].value_counts()}\n")

    split_date = df["TradingDate"].quantile(TRAIN_FRAC)
    train_df   = df[df["TradingDate"] <= split_date].reset_index(drop=True)
    test_df    = df[df["TradingDate"] >  split_date].reset_index(drop=True)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test  = scaler.transform(test_df[feature_cols].values)
    y_train = train_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)
    r_train = train_df["regime"].values.astype(int)
    r_test  = test_df["regime"].values.astype(int)

    train_ds = RegimeDataset(X_train, y_train, r_train)
    test_ds  = RegimeDataset(X_test,  y_test,  r_test)

    # Oversample minority regimes so every expert gets equal gradient signal
    regime_seq     = r_train[SEQ_LEN - 1:]
    regime_counts  = np.bincount(regime_seq, minlength=N_REGIMES).astype(float)
    sample_weights = 1.0 / (regime_counts[regime_seq] + 1e-8)
    sampler        = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.float32), len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Class weights to prevent label collapse
    label_seq     = y_train[SEQ_LEN - 1:]
    label_counts  = np.bincount(label_seq, minlength=3).astype(float)
    class_weights = torch.tensor(1.0 / (label_counts + 1e-8), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 3
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    print(f"Class weights: sell={class_weights[0]:.3f}  hold={class_weights[1]:.3f}  buy={class_weights[2]:.3f}")

    model     = DynamicFactorModel(n_features, feature_cols).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (factor matrix: {N_REGIMES}×{n_features}={N_REGIMES*n_features})")
    print(f"\n{'Epoch':>5}  {'train':>8}  {'val':>8}  {'tr_acc':>7}  {'v_acc':>7}  {'l1':>7}")
    print("─" * 52)

    best_val   = float("inf")
    best_state = None
    patience_n = 0
    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        **{f"val_acc_r{r}": [] for r in range(N_REGIMES)},
    }

    for epoch in range(1, EPOCHS + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_acc"].append(vl["acc"])
        for r in range(N_REGIMES):
            mask = vl["regimes"] == r
            acc  = (vl["preds"][mask] == vl["labels"][mask]).mean() if mask.sum() > 0 else float("nan")
            history[f"val_acc_r{r}"].append(acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:5d}  {tr['task_loss']:8.4f}  {vl['loss']:8.4f}"
                  f"  {tr['acc']:7.3f}  {vl['acc']:7.3f}  {tr['l1_loss']:7.4f}")

        if vl["loss"] < best_val:
            best_val   = vl["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_n = 0
        else:
            patience_n += 1
            if patience_n >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # ── Plots ─────────────────────────────────────────────────────────────────
    out = os.path.join(os.getcwd(), "data")
    plot_training_curves(history, save_path=os.path.join(out, "dynamic_factor_curves.png"))
    plot_factor_heatmap(model,    save_path=os.path.join(out, "dynamic_factor_heatmap.png"))

    # ── Evaluation ────────────────────────────────────────────────────────────
    final = evaluate(model, test_loader, criterion, DEVICE)
    print_per_regime_report(final["preds"], final["labels"], final["regimes"])
    print_factor_weights(model, top_k=6)

    # Return bucket analysis
    test_rets = test_df["future_return"].values[SEQ_LEN - 1 : SEQ_LEN - 1 + len(final["proba"])]
    df_buck   = pd.DataFrame({"proba_buy": final["proba"][:, 2], "ret": test_rets})
    df_buck["bucket"] = pd.qcut(df_buck["proba_buy"], 5, labels=False, duplicates="drop")
    stats = df_buck.groupby("bucket")["ret"].agg(["mean", "std", "count"])
    stats["sharpe"] = stats["mean"] / (stats["std"] + 1e-8)
    print("\n=== Return by Buy-Signal Bucket (0=weakest, 4=strongest) ===")
    print(stats.round(4).to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = os.path.join(out, "dynamic_factor_model.pt")
    torch.save({
        "model_state":  best_state,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "config": {
            "n_features":  n_features,
            "hidden_lstm": HIDDEN_LSTM,
            "hidden_mlp":  HIDDEN_MLP,
            "seq_len":     SEQ_LEN,
            "horizon":     HORIZON,
        },
    }, model_path)
    print(f"\nModel saved → {model_path}")
    return model


if __name__ == "__main__":
    train_dynamic_factor()
