"""
train_moe.py
------------
Time-series Mixture of Experts (MoE) with LSTM gating.

Architecture
------------
Each "expert" is a small MLP that learns to predict the tertile label
(0=bottom, 1=middle, 2=top 5-day return) from the current day's features.
The gating network is an LSTM that reads the last SEQ_LEN days of features
and outputs a softmax distribution over experts — so the gate learns which
expert to trust based on the recent sequence of market behaviour.

  Input sequence  [T-SEQ_LEN .. T]  (n_features per day)
       │
       ▼
  LSTM gating network
       │
       ▼  soft weights  [w_0 .. w_{N_EXPERTS-1}]

  Current features at T
       │
       ├─► Expert_0 (MLP) ─► logits_0 - this expert might specialise in PANIC_BEAR regimes, learning patterns that predict strong rebounds after sharp sell-offs
       ├─► Expert_1 (MLP) ─► logits_1
       ├─► Expert_2 (MLP) ─► logits_2
       └─► Expert_3 (MLP) ─► logits_3

  Weighted sum:  out = Σ w_i * logits_i
       │
       ▼
  Loss = CrossEntropy(out, label)

Why LSTM gating?
----------------
A feedforward gate sees only today's features.  An LSTM gate sees the
full sequence leading up to today, so it learns temporal patterns like:
  "3 consecutive PANIC_BEAR days with rising volume  → transition near"
  "QUIET_BULL for 20+ days with fading volume        → momentum fading"
These patterns inform WHICH expert to trust at each point in time —
something a static gate cannot capture.

Why Mixture of Experts?
-----------------------
Different regimes have structurally different return dynamics.  Rather
than forcing one network to handle all of them, each expert specialises;
the gate decides the blend.  At regime transitions the gate naturally
produces a mixed weight (e.g. 0.6 PANIC_BEAR + 0.4 QUIET_BULL) that
interpolates between expert predictions.

Run
---
  cd /path/to/vnstock_trade
  python -m vnstock_trade.training.train_moe
"""

from __future__ import annotations

import os
import sys
import types

# torch._dynamo and torch.onnx both import transformers, but the installed
# version is incompatible with PyTorch 2.1.  Pre-load a minimal stub so that
# dynamo can patch it without crashing.
def _stub_transformers():
    """ This is a workaround to prevent import errors related to the transformers library when using PyTorch's dynamo or ONNX export features. 
    Since the installed version of transformers is incompatible with PyTorch 2.1, we create a minimal stub module that satisfies the import 
    requirements without actually providing the full functionality of transformers. This allows the code to run without crashing due to import errors, 
    while still allowing dynamo and ONNX to patch the necessary components."""
    
    
    root = types.ModuleType("transformers")

    # torch._dynamo.variables.torch patches PretrainedConfig.__eq__
    config_utils = types.ModuleType("transformers.configuration_utils") 
    class PretrainedConfig:          # noqa: N801
        pass
    config_utils.PretrainedConfig = PretrainedConfig
    root.configuration_utils       = config_utils

    sys.modules["transformers"]                        = root
    sys.modules["transformers.configuration_utils"]    = config_utils

_stub_transformers()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from vnstock_trade.features.detect_regime import build_conditional_features, infer_market_index_from_filename

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "./data/data-vn-20230228/stock-historical-data/VCB-VNINDEX-History.csv"
HORIZON     = 5       # forward return horizon in trading days
SEQ_LEN     = 20      # days of history the LSTM gate sees
N_EXPERTS   = 4       # one per regime, but the model is free to use them differently
HIDDEN_LSTM = 64      # LSTM hidden size for the gate
HIDDEN_MLP  = 64      # hidden units inside each expert
EPOCHS      = 50
BATCH_SIZE  = 64      
LR          = 1e-3
TRAIN_FRAC  = 0.70    
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu" 

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, list[str]]:
    """
    Load raw OHLCV, run the full regime feature pipeline, add days_in_regime,
    one-hot encode regime, build tertile labels, and return a clean DataFrame.
    """
    
    raw = pd.read_csv(data_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])

    market_index = infer_market_index_from_filename(data_path)
    df = build_conditional_features(raw, market_index=market_index)

    # days_in_regime — resets to 0 at the start of each episode
    regime_change        = (df["regime"] != df["regime"].shift()).cumsum()
    df["days_in_regime"] = df.groupby(regime_change).cumcount()
    
    # we are trying to infer regime from the sequence, so we don't want to feed it directly into the model as an ordinal feature, by one-hot encoding the regime, we allow the model to learn separate patterns for each regime without assuming any ordinal relationship between them, this is important because regimes are categorical and do not have a natural order (e.g., PANIC_BEAR is not "less than" QUIET_BULL), by creating binary features for each regime, we enable the model to capture the unique characteristics of each regime and how they relate to the target variable (future return tertile) without introducing misleading assumptions about their relationships.
    # One-hot encode regime — no false ordinal relationship
    regime_dummies = pd.get_dummies(df["regime"], prefix="regime")
    df = pd.concat([df, regime_dummies], axis=1)

    # Tertile label: 0=bottom third, 1=middle, 2=top third of 5-day returns
    df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
    df["label"]         = pd.qcut(df["future_return"], q=3, labels=False)

    # Drop non-feature columns
    non_features = {
        "Open", "High", "Low", "Close", "Volume", "TradingDate",
        "regime", "regime_name", "future_return", "label", "Unnamed: 0",
    }
    regime_dummy_cols = list(regime_dummies.columns) # these are the one-hot encoded regime columns, which are also features, so we need to include them in the feature_cols list, by extracting the column names from the regime_dummies DataFrame, we ensure that all the one-hot encoded regime features are included in the model training, allowing the model to learn different patterns for each regime without assuming any ordinal relationship between them.
    # regime_dummy_cols is a list of the column names corresponding to the one-hot encoded regime features, which are generated by pd.get_dummies and prefixed with 'regime_', for example, if we have 4 regimes, we might get columns like 'regime_0', 'regime_1', 'regime_2', 'regime_3', and these columns will be included in the feature_cols list that is used for training the model, ensuring that the model can learn to differentiate between regimes based on these features.
    feature_cols = (
        regime_dummy_cols
        + ["days_in_regime"]
        + [c for c in df.columns
           if c not in non_features
           and c not in regime_dummy_cols
           and c != "days_in_regime"]
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["label", "future_return"])
    df = df.reset_index(drop=True)

    return df, feature_cols


class RegimeSequenceDataset(Dataset):
    """
    Sliding-window dataset for the MoE model.

    Each sample is:
      sequence : float32 tensor  [SEQ_LEN, n_features]  — LSTM gate input
      current  : float32 tensor  [n_features]            — current day features
      label    : long tensor     []                       — tertile class 0/1/2

    The sequence ends at the SAME day as `current` (day T).  The LSTM gate
    therefore sees days [T-SEQ_LEN .. T] and outputs weights for the experts,
    which predict T+HORIZON return using the day-T feature snapshot.

    Temporal integrity: we only build windows where ALL SEQ_LEN+1 rows lie
    within the same split partition (train or test), so no future data leaks
    into the sequence.
    """

    def __init__(
        self,
        features: np.ndarray,   # shape [N, n_features], scaled
        labels:   np.ndarray,   # shape [N], int
        seq_len:  int = SEQ_LEN,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)
        self.seq_len  = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        seq     = self.features[idx : idx + self.seq_len]          # [SEQ_LEN, F]
        current = self.features[idx + self.seq_len - 1]            # [F] — last in seq
        label   = self.labels[idx + self.seq_len - 1]              # scalar
        return seq, current, label


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────────────────────────────────────

class ExpertMLP(nn.Module):
    """
    One expert: a 2-layer MLP that maps current day features → class logits.
    Dropout for regularisation (experts overfit to their regime if unchecked).
    """

    def __init__(self, n_features: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, n_classes]


class LSTMGate(nn.Module):
    """
    LSTM gating network.

    Reads the SEQ_LEN-day history and outputs a softmax distribution over
    N_EXPERTS experts.  The final hidden state captures the recent trajectory
    of market conditions — which is exactly the information needed to decide
    which expert should dominate.
    """

    def __init__(
        # initialize the LSTM gate with the number of input features, hidden size, and number of experts, the LSTM will process sequences of shape [B, SEQ_LEN, n_features] and output a hidden state of size [B, hidden], which is then passed through a linear layer to produce logits for each expert, and finally a softmax to get the weights for each expert.
        self, n_features: int, hidden: int, n_experts: int
    ) -> None:
        super().__init__()
        self.lstm       = nn.LSTM(n_features, hidden, batch_first=True)
        self.gate_head  = nn.Linear(hidden, n_experts)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, SEQ_LEN, n_features]
        # the LSTM processes the input sequence and returns the output for all time steps and the final hidden state, we only care about the final hidden state (h_n) which captures the information from the entire sequence, we then pass this hidden
        _, (h_n, _) = self.lstm(seq)   # h_n: [1, B, hidden]
        h = h_n.squeeze(0)             # [B, hidden]
        return torch.softmax(self.gate_head(h), dim=-1)  # [B, n_experts] is the softmax distribution over experts, where each value represents the weight assigned to the corresponding expert based on the recent sequence of market conditions captured by the LSTM hidden state.


class MoEModel(nn.Module):
    """
    Full Mixture of Experts model.

    Forward pass
    ------------
    1. Gate reads the sequence → soft weights over experts  [B, N_EXPERTS]
    2. Each expert processes current features               [B, n_classes] × N_EXPERTS
    3. Weighted sum of expert logits                        [B, n_classes]
    4. CrossEntropy applied outside (raw logits returned)
    """

    def __init__(
        self,
        n_features: int,
        n_experts:  int  = N_EXPERTS,
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
        weights = self.gate(seq)                          # [B, N_EXPERTS]

        # Stack expert outputs: [B, N_EXPERTS, n_classes]
        expert_logits = torch.stack(
            [expert(current) for expert in self.experts], dim=1
        )

        # Weighted sum: [B, n_classes]
        # weights[:, :, None] broadcasts to [B, N_EXPERTS, 1]
        out = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)

        return out, weights   # return weights for inspection


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:      MoEModel,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     str,
) -> float:
    model.train()
    total_loss = 0.0

    for seq, current, label in loader:
        seq, current, label = seq.to(device), current.to(device), label.to(device)
        optimizer.zero_grad()
        logits, _ = model(seq, current)
        loss = criterion(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(label)

    return total_loss / len(loader.dataset)


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

        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_weights.append(weights.cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_weights),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_moe() -> MoEModel:
    print(f"Device: {DEVICE}")

    # ── Build features + labels ───────────────────────────────────────────────
    df, feature_cols = build_dataset(DATA_PATH)
    n_features = len(feature_cols)
    print(f"Rows: {len(df)}  |  Features: {n_features}")
    print(f"Regime distribution:\n{df['regime_name'].value_counts()}\n")

    # ── Time-based split ──────────────────────────────────────────────────────
    split_date  = df["TradingDate"].quantile(TRAIN_FRAC)
    train_df    = df[df["TradingDate"] <= split_date].reset_index(drop=True)
    test_df     = df[df["TradingDate"] >  split_date].reset_index(drop=True)
    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

    # ── Scale on train, apply to test ─────────────────────────────────────────
    scaler      = StandardScaler()
    X_train     = scaler.fit_transform(train_df[feature_cols].values)
    X_test      = scaler.transform(test_df[feature_cols].values)
    y_train     = train_df["label"].values.astype(int)
    y_test      = test_df["label"].values.astype(int)

    # ── Datasets + loaders ────────────────────────────────────────────────────
    train_ds = RegimeSequenceDataset(X_train, y_train)
    test_ds  = RegimeSequenceDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = MoEModel(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {EPOCHS} epochs...\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_preds, val_labels, _ = evaluate(
            model, test_loader, criterion, DEVICE
        )
        scheduler.step(val_loss)

        val_acc = (val_preds == val_labels).mean()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Restore best checkpoint ───────────────────────────────────────────────
    model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────────────────────
    _, final_preds, final_labels, gate_weights = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print("\n=== Classification Report ===")
    print(classification_report(final_labels, final_preds,
                                target_names=["bottom", "middle", "top"]))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(final_labels, final_preds))

    # ── Gate weight analysis — which expert does the model rely on? ───────────
    print("\n=== Mean Gate Weights per Expert ===")
    mean_weights = gate_weights.mean(axis=0)
    for i, w in enumerate(mean_weights):
        print(f"  Expert {i}: {w:.3f}")

    # Gate weights by actual label — do different experts fire for different
    # market outcomes?
    test_aligned = test_df.iloc[SEQ_LEN - 1:].copy().reset_index(drop=True)
    test_aligned = test_aligned.iloc[:len(gate_weights)].copy()
    for i in range(N_EXPERTS):
        test_aligned[f"expert_{i}_weight"] = gate_weights[:, i]

    print("\n=== Mean Gate Weights by Regime (test set) ===")
    expert_cols = [f"expert_{i}_weight" for i in range(N_EXPERTS)]
    print(
        test_aligned.groupby("regime_name")[expert_cols]
        .mean()
        .round(3)
        .to_string()
    )

    # ── Signal bucket analysis (top-class probability as signal) ──────────────
    model.eval()
    all_proba_top = []
    with torch.no_grad():
        for seq, current, _ in test_loader:
            seq, current = seq.to(DEVICE), current.to(DEVICE)
            logits, _ = model(seq, current)
            proba = torch.softmax(logits, dim=-1)
            all_proba_top.append(proba[:, 2].cpu().numpy())  # P(top tertile)

    test_aligned["proba_top"] = np.concatenate(all_proba_top)
    test_aligned["bucket"] = pd.qcut(
        test_aligned["proba_top"], 5, labels=False, duplicates="drop"
    )

    EPS = 1e-8
    bucket_stats = (
        test_aligned.groupby("bucket")["future_return"]
        .agg(["mean", "std", "count"])
    )
    bucket_stats["sharpe_proxy"] = (
        bucket_stats["mean"] / (bucket_stats["std"] + EPS)
    )
    print("\n=== Return by Signal Bucket (0=weakest, 4=strongest) ===")
    print(bucket_stats.to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir   = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(out_dir, "moe_model.pt")
    torch.save({
        "model_state": best_state,
        "scaler":      scaler,
        "feature_cols": feature_cols,
        "config": {
            "n_features":  n_features,
            "n_experts":   N_EXPERTS,
            "hidden_lstm": HIDDEN_LSTM,
            "hidden_mlp":  HIDDEN_MLP,
            "seq_len":     SEQ_LEN,
            "horizon":     HORIZON,
        },
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    return model


if __name__ == "__main__":
    train_moe()
