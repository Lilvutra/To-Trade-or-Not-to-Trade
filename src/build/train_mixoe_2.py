"""
train_mixoe_2.py
----------------
Mixture of Experts (MoE) with LSTM gating — regime-blind variant.

Key difference from train_mixoe.py
------------------------------------
Regime labels (one-hot dummies, days_in_regime) are NOT fed to the model.
The gate and experts see only raw market-derived features (momentum, volume,
volatility, limit flags, etc.).  The model must discover regime-like routing
entirely from price/volume dynamics in the sequence.

This tests whether the MoE can self-organise its experts around market
conditions without being told what those conditions are.  The post-training
analysis (gate weights by regime_name) reveals whether the learned routing
correlates with the hand-crafted regime labels — if it does, the model has
rediscovered the regime structure unsupervised.

Architecture (unchanged)
-------------------------
  Input sequence  [T-SEQ_LEN .. T]  (n_market_features per day)
       │
       ▼
  LSTM gating network
       │
       ▼  soft weights  [w_0 .. w_{N_EXPERTS-1}]

  Current features at T
       │
       ├─► Expert_0 (MLP) ─► logits_0
       ├─► Expert_1 (MLP) ─► logits_1
       ├─► Expert_2 (MLP) ─► logits_2
       └─► Expert_3 (MLP) ─► logits_3

  Weighted sum:  out = Σ w_i * logits_i
       │
       ▼
  Loss = CrossEntropy(out, label)
"""

from __future__ import annotations

import os
import sys
import types


def _stub_transformers():
    root = types.ModuleType("transformers")
    config_utils = types.ModuleType("transformers.configuration_utils")
    class PretrainedConfig:
        pass
    config_utils.PretrainedConfig = PretrainedConfig
    root.configuration_utils      = config_utils
    sys.modules["transformers"]                     = root
    sys.modules["transformers.configuration_utils"] = config_utils

_stub_transformers()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from robust_features import build_robust_features
from get_regime import build_conditional_features, detect_regime, infer_market_index_from_filename

_LIMIT_MAP = {"VNINDEX": 0.07, "HNXINDEX": 0.10, "HNX": 0.10, "UPCOMINDEX": 0.15, "UPCOM": 0.15}

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "./data/data-vn-20230228/stock-historical-data/VHM-VNINDEX-History.csv"
HORIZON     = 5
SEQ_LEN     = 20
N_EXPERTS   = 4
HIDDEN_LSTM = 128
HIDDEN_MLP  = 128
EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
TRAIN_FRAC  = 0.70
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data preparation
# ─────────────────────────────────────────────────────────────────────────────

# Columns that are never market features — excluded from model input
_NON_FEATURES = {
    "Open", "High", "Low", "Close", "Volume", "TradingDate",
    "regime", "regime_name", "future_return", "label", "Unnamed: 0",
}

# Regime-derived columns — excluded so the model cannot see the regime label
_REGIME_COLS = {"days_in_regime"}   # regime dummies are never added in this variant


def build_dataset(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, list[str]]:
    """
    Load raw OHLCV, run the regime feature pipeline, build tertile labels,
    and return a DataFrame whose feature_cols contain NO regime information.

    regime and regime_name are kept in the DataFrame only for post-hoc analysis
    (gate weight breakdown by regime) — they are never in feature_cols.
    """
    raw = pd.read_csv(data_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])

    market_index  = infer_market_index_from_filename(data_path)
    limit_thresh  = _LIMIT_MAP.get(market_index.upper(), 0.07)

    # Build robust percentile-rank features (no regime labels in feature set)
    df = build_robust_features(raw, limit_thresh=limit_thresh) # can use build_conditional_features

    # Add regime labels for post-hoc analysis only — never in feature_cols
    df["regime"]      = detect_regime(df)
    df["regime_name"] = df["regime"].map({0: "QUIET_BEAR", 1: "PANIC_BEAR", 2: "QUIET_BULL", 3: "VOLATILE_BULL"})

    # Forward return + asymmetric label
    # 0 = sell/avoid  (return < -2%)
    # 1 = hold        (-2% <= return <= +3%)
    # 2 = buy         (return > +3%)
    df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
    df["label"] = 1  # default: hold
    df.loc[df["future_return"] >  0.03, "label"] = 2
    df.loc[df["future_return"] < -0.02, "label"] = 0

    # Feature columns: everything except identity, target, and regime labels
    feature_cols = [
        c for c in df.columns
        if c not in _NON_FEATURES
        and c not in _REGIME_COLS
        and not c.startswith("regime_")   # exclude any stray regime dummies
    ]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["label", "future_return"])
    df = df.reset_index(drop=True)

    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RegimeSequenceDataset(Dataset):
    """
    Sliding-window dataset.  Identical to train_mixoe.py — the only difference
    is that the feature matrix contains no regime columns.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels:   np.ndarray,
        seq_len:  int = SEQ_LEN,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)
        self.seq_len  = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        seq     = self.features[idx : idx + self.seq_len]
        current = self.features[idx + self.seq_len - 1]
        label   = self.labels[idx + self.seq_len - 1]
        return seq, current, label


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model (identical architecture — different input)
# ─────────────────────────────────────────────────────────────────────────────

class ExpertMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMGate(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_experts: int) -> None:
        super().__init__()
        self.lstm      = nn.LSTM(n_features, hidden, batch_first=True)
        self.gate_head = nn.Linear(hidden, n_experts)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        h = h_n.squeeze(0)
        return torch.softmax(self.gate_head(h), dim=-1)


class MoEModel(nn.Module):
    def __init__(
        self,
        n_features:  int,
        n_experts:   int = N_EXPERTS,
        hidden_lstm: int = HIDDEN_LSTM,
        hidden_mlp:  int = HIDDEN_MLP,
        n_classes:   int = 3,
    ) -> None:
        super().__init__()
        self.gate    = LSTMGate(n_features, hidden_lstm, n_experts)
        self.experts = nn.ModuleList([
            ExpertMLP(n_features, hidden_mlp, n_classes)
            for _ in range(n_experts)
        ])

    def forward(
        self, seq: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights       = self.gate(seq)
        expert_logits = torch.stack(
            [expert(current) for expert in self.experts], dim=1
        )
        out = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        return out, weights


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training helpers
# ─────────────────────────────────────────────────────────────────────────────

AUX_LOSS_COEF = 0.01   # scale of load-balancing penalty vs task loss


def load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation from uniform expert utilization.
    weights: [B, N_EXPERTS] — gate softmax output.

    If one expert always gets weight ~1.0 and others ~0.0, mean_weight will
    be far from [1/N, ..., 1/N] and this loss will push the gate back toward
    balanced routing, forcing all experts to receive gradient and specialise.
    """
    mean_weight = weights.mean(dim=0)                        # [N_EXPERTS]
    target      = torch.ones_like(mean_weight) / weights.shape[1]
    return ((mean_weight - target) ** 2).sum()


def train_epoch(
    model:     MoEModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    str,
) -> tuple[float, float]:
    model.train()
    total_task_loss = 0.0
    total_aux_loss  = 0.0
    for seq, current, label in loader:
        seq, current, label = seq.to(device), current.to(device), label.to(device)
        optimizer.zero_grad()
        logits, weights = model(seq, current)
        task_loss = criterion(logits, label)
        aux_loss  = load_balance_loss(weights)
        loss      = task_loss + AUX_LOSS_COEF * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_task_loss += task_loss.item() * len(label)
        total_aux_loss  += aux_loss.item()  * len(label)
    n = len(loader.dataset)
    return total_task_loss / n, total_aux_loss / n


@torch.no_grad()
def evaluate(
    model:     MoEModel,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_weights = [], [], []
    for seq, current, label in loader:
        seq, current, label = seq.to(device), current.to(device), label.to(device)
        logits, weights = model(seq, current)
        loss = criterion(logits, label)
        total_loss += loss.item() * len(label)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_weights.append(weights.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_weights),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_moe() -> MoEModel:
    print(f"Device: {DEVICE}")
    print("Regime-blind variant: regime labels excluded from model input.\n")

    df, feature_cols = build_dataset(DATA_PATH)
    n_features = len(feature_cols)
    print(f"Rows: {len(df)}  |  Features: {n_features}")
    print(f"Regime distribution (for analysis only, not fed to model):")
    print(df["regime_name"].value_counts(), "\n")

    # ── Time-based split ──────────────────────────────────────────────────────
    split_date = df["TradingDate"].quantile(TRAIN_FRAC)
    train_df   = df[df["TradingDate"] <= split_date].reset_index(drop=True)
    test_df    = df[df["TradingDate"] >  split_date].reset_index(drop=True)
    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test  = scaler.transform(test_df[feature_cols].values)
    y_train = train_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)

    # ── Datasets + loaders ────────────────────────────────────────────────────
    train_ds     = RegimeSequenceDataset(X_train, y_train)
    test_ds      = RegimeSequenceDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Class weights — inverse frequency so sell/buy are not drowned by hold ──
    counts      = np.bincount(y_train, minlength=3).astype(float)
    class_weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32).to(DEVICE)
    print(f"Label distribution (train): sell={int(counts[0])}  hold={int(counts[1])}  buy={int(counts[2])}")
    print(f"Class weights:              sell={class_weights[0]:.2f}  hold={class_weights[1]:.2f}  buy={class_weights[2]:.2f}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = MoEModel(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {EPOCHS} epochs...\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, aux_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_preds, val_labels, _ = evaluate(
            model, test_loader, criterion, DEVICE
        )
        scheduler.step(val_loss)
        val_acc = (val_preds == val_labels).mean()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  "
                  f"aux_loss={aux_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────────────────────
    _, final_preds, final_labels, gate_weights = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print("\n=== Classification Report ===")
    print(classification_report(final_labels, final_preds,
                                target_names=["sell", "hold", "buy"]))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(final_labels, final_preds))

    # ── Gate weight analysis ──────────────────────────────────────────────────
    print("\n=== Mean Gate Weights per Expert ===")
    for i, w in enumerate(gate_weights.mean(axis=0)):
        print(f"  Expert {i}: {w:.3f}")

    # Align gate weights with test rows and the hand-crafted regime labels.
    # This reveals whether the model self-organised its routing to match
    # regime structure — even though it never saw the regime labels.
    test_aligned = test_df.iloc[SEQ_LEN - 1:].copy().reset_index(drop=True)
    test_aligned = test_aligned.iloc[:len(gate_weights)].copy()
    for i in range(N_EXPERTS):
        test_aligned[f"expert_{i}_weight"] = gate_weights[:, i]

    print("\n=== Mean Gate Weights by Regime (post-hoc — regime was NOT a model input) ===")
    expert_cols = [f"expert_{i}_weight" for i in range(N_EXPERTS)]
    print(
        test_aligned.groupby("regime_name")[expert_cols]
        .mean()
        .round(3)
        .to_string()
    )
    print("\nIf weights vary across regimes, the model rediscovered regime structure")
    print("from price/volume features alone.  Flat weights = no self-organisation.")

    # ── Expert class bias ─────────────────────────────────────────────────────
    model.eval()
    expert_class_probs = [[] for _ in range(N_EXPERTS)]
    with torch.no_grad():
        for _, current, _ in test_loader:
            current = current.to(DEVICE)
            for i, expert in enumerate(model.experts):
                probs = torch.softmax(expert(current), dim=-1).cpu().numpy()
                expert_class_probs[i].append(probs)

    print("\n=== Expert Class Bias (avg softmax when used alone) ===")
    print(f"{'':12s}  {'sell':>8}  {'hold':>8}  {'buy':>8}  → bias")
    for i, chunks in enumerate(expert_class_probs):
        avg  = np.concatenate(chunks).mean(axis=0)
        bias = ["sell", "hold", "buy"][avg.argmax()]
        print(f"  Expert {i}:   {avg[0]:8.3f}  {avg[1]:8.3f}  {avg[2]:8.3f}  → {bias}")

    # ── Top-5 features per expert (weight norm) ──────────────────────────────
    print("\n=== Top-5 Features per Expert (first-layer weight norm) ===")
    for i, expert in enumerate(model.experts):
        w          = expert.net[0].weight.detach().cpu().numpy()
        importance = np.linalg.norm(w, axis=0)
        top5_idx   = importance.argsort()[::-1][:5]
        top5       = [(feature_cols[j], round(float(importance[j]), 4)) for j in top5_idx]
        print(f"  Expert {i}: {top5}")

    # ── Permutation importance per expert ─────────────────────────────────────
    print("\n=== Top-5 Features per Expert (permutation importance) ===")

    def _expert_accuracy(expert_idx: int, shuffle_idx: int | None = None) -> float:
        correct, total = 0, 0
        with torch.no_grad():
            for seq, current, label in test_loader:
                current = current.to(DEVICE)
                if shuffle_idx is not None:
                    perm = torch.randperm(current.shape[0])
                    current = current.clone()
                    current[:, shuffle_idx] = current[perm, shuffle_idx]
                logits = model.experts[expert_idx](current)
                preds  = logits.argmax(dim=-1).cpu()
                correct += (preds == label).sum().item()
                total   += len(label)
        return correct / total

    N_REPEATS = 5
    model.eval()
    for i in range(N_EXPERTS):
        base_acc = _expert_accuracy(i)
        drops = {}
        for j, feat in enumerate(feature_cols):
            delta = np.mean([base_acc - _expert_accuracy(i, shuffle_idx=j)
                             for _ in range(N_REPEATS)])
            drops[feat] = round(delta, 4)
        top5 = sorted(drops.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Expert {i} (base acc={base_acc:.3f}): {top5}")

    # ── Signal bucket analysis ────────────────────────────────────────────────
    model.eval()
    all_proba_top = []
    with torch.no_grad():
        for seq, current, _ in test_loader:
            seq, current = seq.to(DEVICE), current.to(DEVICE)
            logits, _ = model(seq, current)
            all_proba_top.append(torch.softmax(logits, dim=-1)[:, 2].cpu().numpy())

    test_aligned["proba_top"] = np.concatenate(all_proba_top)
    test_aligned["bucket"]    = pd.qcut(
        test_aligned["proba_top"], 5, labels=False, duplicates="drop"
    )

    EPS = 1e-8
    bucket_stats = test_aligned.groupby("bucket")["future_return"].agg(["mean", "std", "count"])
    bucket_stats["sharpe_proxy"] = bucket_stats["mean"] / (bucket_stats["std"] + EPS)
    print("\n=== Return by Signal Bucket (0=weakest, 4=strongest) ===")
    print(bucket_stats.to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir    = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(out_dir, "moe_model_v2.pt")
    torch.save({
        "model_state":  best_state,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "config": {
            "n_features":  n_features,
            "n_experts":   N_EXPERTS,
            "hidden_lstm": HIDDEN_LSTM,
            "hidden_mlp":  HIDDEN_MLP,
            "seq_len":     SEQ_LEN,
            "horizon":     HORIZON,
            "regime_blind": True,
        },
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    return model


if __name__ == "__main__":
    train_moe()
