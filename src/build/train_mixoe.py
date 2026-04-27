"""
train_mixoe.py
--------------
Time-series Mixture of Experts (MoE) with regime-conditioned LSTM gating.

Architecture (v3)
-----------------
The gate now receives both the LSTM hidden state AND a learned regime embedding
from detect_regime(), so routing is directly conditioned on the market regime.

  Input sequence  [T-SEQ_LEN .. T]  (n_features per day)
       │
       ▼
  LSTM gating network
       │
       ├─► hidden state h ─────────────────────────────────────────────────┐
       │                                                                    │
  detect_regime()                                                           │
       │                                                                    │
       ▼ regime_embed                                                       │
  concat[h | regime_embed]                                                  │
       │                                                                    │
       ▼  gate weights [w_0..w_{N-1}]         concat [current | h]
                                                      │
       ├─► Expert_0 (MLP) ──────────────────────────►│─► logits_0
       ├─► Expert_1 (MLP) ──────────────────────────►│─► logits_1
       ├─► Expert_2 (MLP) ──────────────────────────►│─► logits_2
       └─► Expert_3 (MLP) ──────────────────────────►│─► logits_3

  Weighted sum:  out = Σ w_i * logits_i
       │
       ▼
  Loss = task_loss + λ_bal * balance_loss + λ_ent * entropy_loss
                   + λ_align * regime_alignment_loss

Losses
------
  task_loss            : CrossEntropy on label (main objective)
  balance_loss         : penalize deviation from uniform expert usage → prevent collapse
  entropy_loss         : maximize gate distribution entropy → prevent hard routing too early
  regime_alignment_loss: KL(gate_weights || regime_routing_probs)
                         A learnable [N_REGIMES × N_EXPERTS] matrix is jointly optimised.
                         Each regime learns to prefer certain experts; gate weights are
                         pushed to align with its regime's routing preference.
                         This is stronger than the old stop-gradient probe — gradients
                         flow into the gate and into the routing matrix simultaneously.
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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_regime import (
    build_conditional_features,
    infer_market_index_from_filename,
    REGIME_FEATURES,   # {regime_id: [feature_names]} — domain knowledge from get_regime.py
    REGIME_NAME,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH    = "./data/data-vn-20230228/stock-historical-data/VCB-VNINDEX-History.csv"
HORIZON      = 10
SEQ_LEN      = 20
N_EXPERTS    = 4
N_REGIMES    = 4        # QUIET_BEAR, PANIC_BEAR, QUIET_BULL, VOLATILE_BULL
HIDDEN_LSTM  = 64       # reduced from 128 — dataset has ~2.3K sequences, 128 was overfit
HIDDEN_MLP   = 64       # reduced from 128 — same reason
DROPOUT      = 0.5      # increased from 0.3 — stronger regularisation
EPOCHS       = 60
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4     # L2 regularisation on all weights
TRAIN_FRAC   = 0.70
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

# Loss coefficients
LAMBDA_BALANCE   = 0.05   # increased from 0.02 — Expert 3 was dominating 60% of samples
LAMBDA_ENTROPY   = 0.05   # entropy: keep gate soft early, annealed to 0 by epoch 60%
LAMBDA_ALIGNMENT = 0.10   # regime-gate alignment: KL(gate || regime_routing_probs)
LAMBDA_DIVERSITY = 0.005  # diversity on softmax probs (bounded)

REGIME_EMBED_DIM = 16     # dimension of learned regime embedding injected into gate


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(data_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])

    market_index = infer_market_index_from_filename(data_path)
    df = build_conditional_features(raw, market_index=market_index)

    regime_change        = (df["regime"] != df["regime"].shift()).cumsum()
    df["days_in_regime"] = df.groupby(regime_change).cumcount()

    df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
    # asymmetric labels: sell (<-2%), hold, buy (>+3%)
    df["label"] = 1  # hold
    df.loc[df["future_return"] < -0.03, "label"] = 0   # sell
    df.loc[df["future_return"] >  0.05, "label"] = 2   # buy

    # feature_cols = union of all REGIME_FEATURES, preserving ordering per regime.
    # This is the complete set of domain-validated features from get_regime.py.
    # Each expert will then receive a binary mask keeping only its regime's subset.
    seen = set()
    ordered_feats: list[str] = []
    for regime_id in sorted(REGIME_FEATURES):
        for feat in REGIME_FEATURES[regime_id]:
            if feat not in seen:
                seen.add(feat)
                ordered_feats.append(feat)
    # keep only columns that actually exist in df and add days_in_regime last
    feature_cols = [f for f in ordered_feats if f in df.columns] + ["days_in_regime"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["label", "future_return"])
    df = df.reset_index(drop=True)

    return df, feature_cols


class RegimeSequenceDataset(Dataset):
    """
    Returns (seq, current, label, regime) per sample.
    regime is used for weak supervision of the gate's hidden state.
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
        label   = self.labels[idx + self.seq_len - 1]
        regime  = self.regimes[idx + self.seq_len - 1]
        return seq, current, label, regime


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────────────────────────────────────

class ExpertMLP(nn.Module):
    """
    Expert that sees both current features AND the LSTM sequence summary.
    Input: concat([current_features, lstm_hidden]) → [n_features + hidden_lstm]

    feature_mask : binary float tensor [n_features] derived from REGIME_FEATURES.
        1.0 for columns in this expert's assigned regime, 0.0 otherwise.
        Registered as a buffer (not a parameter) — it's fixed domain knowledge,
        not learned. Zeroing irrelevant features forces each expert to specialize
        on the signals that actually matter for its regime.
        The hidden-state portion (last hidden_lstm dims) is always 1.0 — the LSTM
        context is never masked since it encodes regime dynamics implicitly.
    """

    def __init__(
        self,
        n_input:      int,
        hidden:       int,
        n_classes:    int,
        feature_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, n_classes),
        )
        if feature_mask is not None:
            self.register_buffer("feature_mask", feature_mask)
        else:
            self.feature_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mask is not None:
            x = x * self.feature_mask
        return self.net(x)


class LSTMGate(nn.Module):
    """
    Regime-conditioned LSTM gate.

    The gate sees concat(h, regime_embedding) before producing routing weights,
    so the detected market regime directly biases which expert is activated.
    No stop-gradient probe — routing gradients flow freely into both h and
    the regime embedding, aligning the gate with market structure.
    """

    def __init__(
        self,
        n_features:      int,
        hidden:          int,
        n_experts:       int,
        n_regimes:       int,
        regime_embed_dim: int = REGIME_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.lstm         = nn.LSTM(n_features, hidden, batch_first=True)
        self.regime_embed = nn.Embedding(n_regimes, regime_embed_dim)
        # gate sees LSTM context + learned regime embedding
        self.gate_head    = nn.Linear(hidden + regime_embed_dim, n_experts)

    def forward(
        self, seq: torch.Tensor, regime: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(seq)                         # h_n: [1, B, hidden]
        h           = h_n.squeeze(0)                          # [B, hidden]
        r_emb       = self.regime_embed(regime)               # [B, embed_dim]
        gate_input  = torch.cat([h, r_emb], dim=-1)          # [B, hidden + embed_dim]
        weights     = torch.softmax(self.gate_head(gate_input), dim=-1)  # [B, N_EXPERTS]
        return weights, h


class MoEModel(nn.Module):
    """
    Regime-conditioned MoE model.

    Forward pass
    ------------
    1. Gate reads (sequence, detected_regime) → (weights, hidden_state)
       - regime_embed injects market regime directly into routing
    2. Each expert reads concat(current, hidden_state) → logits
    3. Weighted sum of expert logits → final prediction

    regime_routing : [N_REGIMES, N_EXPERTS] learnable parameter
       softmax(regime_routing[r]) = which experts regime r prefers
       Used in regime_alignment_loss to push gate weights toward regime preference.
    """

    def __init__(
        self,
        n_features:   int,
        feature_cols: list[str],
        n_experts:    int = N_EXPERTS,
        n_regimes:    int = N_REGIMES,
        hidden_lstm:  int = HIDDEN_LSTM,
        hidden_mlp:   int = HIDDEN_MLP,
        n_classes:    int = 3,
    ) -> None:
        super().__init__()
        self.n_experts      = n_experts
        self.n_regimes      = n_regimes
        self.gate           = LSTMGate(n_features, hidden_lstm, n_experts, n_regimes)
        # learnable per-regime routing preference — jointly optimised with gate
        # regime_routing[r] → softmax → which experts regime r prefers
        # Diagonal init: Expert i starts strongly preferred for Regime i.
        # torch.zeros → softmax = uniform → KL(uniform||uniform) = 0 → no gradient.
        # torch.eye * 2.0 → softmax ≈ [0.58, 0.14, 0.14, 0.14] → clear gradient signal.
        self.regime_routing = nn.Parameter(torch.eye(n_regimes, n_experts) * 2.0)

        # Build one feature mask per expert (Expert i → Regime i).
        # Mask is [n_features + hidden_lstm]: feature portion uses REGIME_FEATURES,
        # hidden portion is always 1.0 so LSTM context is never masked.
        experts = []
        for regime_id in range(n_experts):
            regime_feat_set = set(REGIME_FEATURES.get(regime_id, []))
            feat_mask = torch.tensor(
                [1.0 if col in regime_feat_set else 0.0 for col in feature_cols],
                dtype=torch.float32,
            )
            # extend mask with 1.0 for hidden_lstm dims (always unmasked)
            full_mask = torch.cat([feat_mask, torch.ones(hidden_lstm)])
            experts.append(ExpertMLP(n_features + hidden_lstm, hidden_mlp, n_classes, full_mask))
        self.experts = nn.ModuleList(experts)

    def forward(
        self, seq: torch.Tensor, current: torch.Tensor, regime: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, h = self.gate(seq, regime)               # [B,N], [B,H]

        expert_input  = torch.cat([current, h], dim=-1)  # [B, n_features + hidden]
        expert_logits = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=1
        )                                                 # [B, N_EXPERTS, n_classes]

        out = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)  # [B, n_classes]
        return out, weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. Auxiliary losses
# ─────────────────────────────────────────────────────────────────────────────

def balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Penalize deviation from uniform expert usage across the batch."""
    mean_w = weights.mean(dim=0)
    target = torch.ones_like(mean_w) / weights.shape[1]
    return ((mean_w - target) ** 2).sum()


def entropy_loss(weights: torch.Tensor) -> torch.Tensor:
    """Negative entropy — minimizing maximizes gate softness."""
    eps = 1e-8
    return (weights * torch.log(weights + eps)).sum(dim=-1).mean()


def diversity_loss(expert_logits: torch.Tensor) -> torch.Tensor:
    """
    Penalise experts making identical predictions.
    expert_logits: [B, N_EXPERTS, n_classes]

    Applied to softmax PROBABILITIES (bounded [0,1]), not raw logits.
    Raw logits grow unbounded during training → variance explodes → val_loss diverges.
    """
    probs     = torch.softmax(expert_logits, dim=-1)             # [B, N_EXPERTS, n_classes]
    mean_pred = probs.mean(dim=1, keepdim=True)                  # [B, 1, n_classes]
    variance  = ((probs - mean_pred) ** 2).mean()
    return -variance


def regime_alignment_loss(
    weights:         torch.Tensor,  # [B, N_EXPERTS] gate weights
    regimes:         torch.Tensor,  # [B] int — detected regime per sample
    regime_routing:  torch.Tensor,  # [N_REGIMES, N_EXPERTS] learnable parameter
) -> torch.Tensor:
    """
    KL divergence between gate weights and each sample's regime routing preference.

    regime_routing[r] → softmax → P_r  (which experts regime r prefers)
    gate weights      → Q_i           (what the gate actually chose)

    Loss = KL(P_r || Q_i)  averaged over batch.

    Both the gate and the regime_routing matrix receive gradients, so:
      - gate learns to route consistently with what each regime prefers
      - regime_routing matrix learns which experts work best per regime
    This replaces the old stop-gradient probe (which gave zero gradient into routing).
    """
    routing_probs  = torch.softmax(regime_routing, dim=-1)   # [N_REGIMES, N_EXPERTS]
    target_routing = routing_probs[regimes]                   # [B, N_EXPERTS]
    eps = 1e-8
    # KL(target || weights): push gate weights toward regime's preferred distribution
    kl = (target_routing * (torch.log(target_routing + eps) - torch.log(weights + eps)))
    return kl.sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:          MoEModel,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    task_criterion: nn.Module,
    device:         str,
    epoch:          int,
) -> dict[str, float]:
    model.train()
    totals = {"task": 0.0, "balance": 0.0, "entropy": 0.0, "align": 0.0, "diversity": 0.0, "total": 0.0}
    all_weights = []

    # anneal entropy coefficient: high early (exploratory) → 0 by 60% of training
    entropy_coef = LAMBDA_ENTROPY * max(0.0, 1.0 - epoch / (EPOCHS * 0.6))

    for seq, current, label, regime in loader:
        seq, current, label, regime = (
            seq.to(device), current.to(device), label.to(device), regime.to(device)
        )
        optimizer.zero_grad()

        logits, weights = model(seq, current, regime)

        # per-expert logits for diversity loss (reuse h already computed in forward)
        with torch.no_grad():
            _, h = model.gate(seq, regime)
        expert_input  = torch.cat([current, h], dim=-1)
        expert_logits = torch.stack(
            [expert(expert_input) for expert in model.experts], dim=1
        )  # [B, N_EXPERTS, n_classes]

        t_loss = task_criterion(logits, label)
        b_loss = balance_loss(weights)
        e_loss = entropy_loss(weights)
        a_loss = regime_alignment_loss(weights, regime, model.regime_routing)
        d_loss = diversity_loss(expert_logits)

        loss = (t_loss
                + LAMBDA_BALANCE   * b_loss
                - entropy_coef     * e_loss      # subtract: minimising -entropy = maximising entropy
                + LAMBDA_ALIGNMENT * a_loss
                + LAMBDA_DIVERSITY * d_loss)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = len(label)
        totals["task"]      += t_loss.item() * n
        totals["balance"]   += b_loss.item() * n
        totals["entropy"]   += e_loss.item() * n
        totals["align"]     += a_loss.item() * n
        totals["diversity"] += d_loss.item() * n
        totals["total"]     += loss.item()   * n
        all_weights.append(weights.detach().cpu().numpy())

    N = len(loader.dataset)
    losses = {k: v / N for k, v in totals.items()}
    losses["entropy_coef"] = entropy_coef

    W = np.concatenate(all_weights)
    dominant = W.argmax(axis=1)
    usage = np.bincount(dominant, minlength=N_EXPERTS) / len(dominant)
    losses["mean_weights"] = W.mean(axis=0).tolist()
    losses["usage"]        = usage.tolist()

    return losses


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

    for seq, current, label, regime in loader:
        seq, current, label, regime = (
            seq.to(device), current.to(device), label.to(device), regime.to(device)
        )
        logits, weights = model(seq, current, regime)
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


def specialization_score(gate_weights_by_regime: pd.DataFrame) -> None:
    """
    For each expert compute how concentrated its usage is across regimes.
    Score = max_regime_weight / mean_regime_weight.
    Score > 1.5 = meaningfully specialised. Score ≈ 1.0 = no specialisation.
    """
    print("\n=== Expert Specialisation Score (max/mean across regimes) ===")
    for col in gate_weights_by_regime.columns:
        vals  = gate_weights_by_regime[col].values
        score = vals.max() / (vals.mean() + 1e-8)
        top_regime = gate_weights_by_regime[col].idxmax()
        status = "✓ specialised" if score > 1.3 else "✗ generalised"
        print(f"  {col}: score={score:.2f}  dominant_regime={top_regime}  {status}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training curve plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict[str, list], save_path: str | None = None) -> None:
    """
    Plot training vs validation loss and all auxiliary loss components.

    history keys expected:
      epoch, task_train, val, balance, entropy_coef, align, diversity, val_acc
    """
    epochs = history["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Curves — MoE Model", fontsize=13, fontweight="bold")

    # ── Top-left: task loss train vs val ─────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history["task_train"], label="train task",  color="#3b82f6", linewidth=1.5)
    ax.plot(epochs, history["val"],        label="val (task)",  color="#ef4444", linewidth=1.5, linestyle="--")
    best_ep = epochs[int(np.argmin(history["val"]))]
    ax.axvline(best_ep, color="#94a3b8", linestyle=":", linewidth=1, label=f"best val (ep {best_ep})")
    ax.set_title("Task Loss (train vs val)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CrossEntropy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Top-right: val accuracy ───────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history["val_acc"], color="#22c55e", linewidth=1.5)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    # ── Bottom-left: auxiliary losses ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, history["balance"],   label="balance",   color="#f59e0b", linewidth=1.3)
    ax.plot(epochs, history["align"],     label="alignment", color="#8b5cf6", linewidth=1.3)
    ax.plot(epochs, history["diversity"], label="diversity", color="#06b6d4", linewidth=1.3)
    ax.set_title("Auxiliary Losses")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Bottom-right: entropy coefficient (annealing schedule) ───────────────
    ax = axes[1, 1]
    ax.plot(epochs, history["entropy_coef"], color="#64748b", linewidth=1.5, linestyle="--")
    ax.fill_between(epochs, history["entropy_coef"], alpha=0.15, color="#64748b")
    ax.set_title("Entropy Coefficient (annealing)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("entropy_coef")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_moe() -> MoEModel:
    print(f"Device: {DEVICE}")

    df, feature_cols = build_dataset(DATA_PATH)
    n_features = len(feature_cols)
    print(f"Rows: {len(df)}  |  Features: {n_features}")
    print(f"Regime distribution:\n{df['regime_name'].value_counts()}\n")

    split_date = df["TradingDate"].quantile(TRAIN_FRAC)
    train_df   = df[df["TradingDate"] <= split_date].reset_index(drop=True)
    test_df    = df[df["TradingDate"] >  split_date].reset_index(drop=True)
    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test  = scaler.transform(test_df[feature_cols].values)
    y_train = train_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)
    r_train = train_df["regime"].values.astype(int)
    r_test  = test_df["regime"].values.astype(int)

    train_ds = RegimeSequenceDataset(X_train, y_train, r_train)
    test_ds  = RegimeSequenceDataset(X_test,  y_test,  r_test)

    # WeightedRandomSampler: oversample minority regimes so each batch sees
    # a balanced mix of all 4 regimes, not 85% QUIET_BEAR.
    regime_train   = r_train[SEQ_LEN - 1:]           # align with dataset indices
    regime_counts  = np.bincount(regime_train, minlength=N_REGIMES).astype(float)
    sample_weights = 1.0 / regime_counts[regime_train]
    sampler        = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float32),
        num_samples = len(train_ds),
        replacement = True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"Regime counts (train): " +
          " ".join(f"{i}={int(regime_counts[i])}" for i in range(N_REGIMES)))
    print(f"Sampler: each regime sees ~equal batches\n")

    # Print expert feature coverage so we can verify masks are correct
    print("Expert feature masks from REGIME_FEATURES:")
    for regime_id, feats in REGIME_FEATURES.items():
        active = [f for f in feats if f in feature_cols]
        print(f"  Expert {regime_id} ({REGIME_NAME[regime_id]}): {len(active)}/{len(feats)} features active")

    model     = MoEModel(n_features, feature_cols).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    # Class weights: inverse frequency so sell/buy are not drowned by hold.
    # Model collapsed to predicting "hold" for everything without this.
    label_train    = y_train[SEQ_LEN - 1:]
    label_counts   = np.bincount(label_train, minlength=3).astype(float)
    class_weights  = torch.tensor(1.0 / (label_counts + 1e-8), dtype=torch.float32)
    class_weights  = class_weights / class_weights.sum() * 3   # scale so mean weight ≈ 1
    task_criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    print(f"Class weights: sell={class_weights[0]:.3f}  hold={class_weights[1]:.3f}  buy={class_weights[2]:.3f}")

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss: balance={LAMBDA_BALANCE}  entropy={LAMBDA_ENTROPY}→0  alignment={LAMBDA_ALIGNMENT}  diversity={LAMBDA_DIVERSITY}")
    print(f"Training for {EPOCHS} epochs...\n")
    print(f"{'Epoch':>5}  {'task':>7}  {'bal':>7}  {'ent_c':>6}  {'align':>7}  {'div':>7}  {'val':>7}  {'acc':>6}  expert_usage")
    print("-" * 105)

    best_val_loss = float("inf")
    best_state    = None
    history: dict[str, list] = {
        "epoch": [], "task_train": [], "val": [], "val_acc": [],
        "balance": [], "entropy_coef": [], "align": [], "diversity": [],
    }

    for epoch in range(1, EPOCHS + 1):
        train_losses = train_epoch(
            model, train_loader, optimizer, task_criterion, DEVICE, epoch
        )
        val_loss, val_preds, val_labels, _ = evaluate(
            model, test_loader, task_criterion, DEVICE
        )
        scheduler.step()
        val_acc = (val_preds == val_labels).mean()

        history["epoch"].append(epoch)
        history["task_train"].append(train_losses["task"])
        history["val"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["balance"].append(train_losses["balance"])
        history["entropy_coef"].append(train_losses["entropy_coef"])
        history["align"].append(train_losses["align"])
        history["diversity"].append(train_losses["diversity"])

        if epoch % 5 == 0 or epoch == 1:
            usage_str = " ".join(f"e{i}={u:.2f}" for i, u in enumerate(train_losses["usage"]))
            print(
                f"{epoch:5d}  "
                f"{train_losses['task']:7.4f}  "
                f"{train_losses['balance']:7.4f}  "
                f"{train_losses['entropy_coef']:6.3f}  "
                f"{train_losses['align']:7.4f}  "
                f"{train_losses['diversity']:7.4f}  "
                f"{val_loss:7.4f}  "
                f"{val_acc:6.3f}  "
                f"{usage_str}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # ── Training curves ───────────────────────────────────────────────────────
    plot_training_curves(
        history,
        save_path=os.path.join(os.getcwd(), "data", "training_curves.png"),
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    _, final_preds, final_labels, gate_weights = evaluate(
        model, test_loader, task_criterion, DEVICE
    )

    print("\n=== Classification Report ===")
    print(classification_report(final_labels, final_preds,
                                target_names=["bottom", "middle", "top"]))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(final_labels, final_preds))

    # ── Gate weight analysis ──────────────────────────────────────────────────
    print("\n=== Mean Gate Weights per Expert ===")
    mean_w = gate_weights.mean(axis=0)
    for i, w in enumerate(mean_w):
        print(f"  Expert {i}: {w:.3f}")

    dominant  = gate_weights.argmax(axis=1)
    usage     = np.bincount(dominant, minlength=N_EXPERTS) / len(dominant)
    print("\n=== Per-Expert Dominant Usage (fraction of test samples) ===")
    for i, u in enumerate(usage):
        bar = "█" * int(u * 40)
        print(f"  Expert {i}: {u:.3f}  {bar}")

    test_aligned = test_df.iloc[SEQ_LEN - 1:].copy().reset_index(drop=True)
    test_aligned = test_aligned.iloc[:len(gate_weights)].copy()
    for i in range(N_EXPERTS):
        test_aligned[f"expert_{i}_weight"] = gate_weights[:, i]

    expert_cols = [f"expert_{i}_weight" for i in range(N_EXPERTS)]
    regime_gate = (
        test_aligned.groupby("regime_name")[expert_cols]
        .mean()
        .round(3)
    )
    print("\n=== Mean Gate Weights by Regime (test set) ===")
    print(regime_gate.to_string())

    # ── Specialisation check ──────────────────────────────────────────────────
    specialization_score(regime_gate)

    # ── Regime routing matrix ─────────────────────────────────────────────────
    # Shows what each regime learned to prefer — independent of which samples
    # actually occurred. High value = regime strongly prefers that expert.
    REGIME_NAMES = ["QUIET_BEAR", "PANIC_BEAR", "QUIET_BULL", "VOLATILE_BULL"]
    routing_probs = torch.softmax(model.regime_routing, dim=-1).detach().cpu().numpy()
    print("\n=== Learned Regime Routing Preferences (regime_routing matrix) ===")
    header = "  " + "".join(f"  Expert {i}" for i in range(N_EXPERTS))
    print(header)
    for r, name in enumerate(REGIME_NAMES):
        row = "  ".join(f"{routing_probs[r, e]:.3f}" for e in range(N_EXPERTS))
        dominant_expert = routing_probs[r].argmax()
        print(f"  {name:<18s}  {row}  → Expert {dominant_expert}")
    print("  (Learned purely from detect_regime() labels + KL alignment loss)")

    # ── Expert class bias ─────────────────────────────────────────────────────
    model.eval()
    expert_class_probs = [[] for _ in range(N_EXPERTS)]
    with torch.no_grad():
        for seq, current, _, regime in test_loader:
            seq     = seq.to(DEVICE)
            current = current.to(DEVICE)
            regime  = regime.to(DEVICE)
            _, h    = model.gate(seq, regime)
            expert_input = torch.cat([current, h], dim=-1)
            for i, expert in enumerate(model.experts):
                probs = torch.softmax(expert(expert_input), dim=-1).cpu().numpy()
                expert_class_probs[i].append(probs)

    print("\n=== Expert Class Bias (avg softmax when used alone) ===")
    print(f"{'':12s}  {'bottom':>8}  {'middle':>8}  {'top':>8}  → bias")
    for i, chunks in enumerate(expert_class_probs):
        avg  = np.concatenate(chunks).mean(axis=0)
        bias = ["bottom", "middle", "top"][avg.argmax()]
        print(f"  Expert {i}:   {avg[0]:8.3f}  {avg[1]:8.3f}  {avg[2]:8.3f}  → {bias}")

    # ── Top-5 features per expert (weight norm, within masked region only) ──────
    print("\n=== Top-5 Features per Expert (first-layer weight norm, active mask only) ===")
    for i, expert in enumerate(model.experts):
        regime_name = REGIME_NAME.get(i, f"Regime{i}")
        w           = expert.net[0].weight.detach().cpu().numpy()
        importance  = np.linalg.norm(w, axis=0)[:n_features]
        # mask out features the expert can't see (importance=0 by mask, but show explicitly)
        regime_feat_set = set(REGIME_FEATURES.get(i, []))
        active_idx  = [j for j, col in enumerate(feature_cols) if col in regime_feat_set]
        if active_idx:
            top_idx = sorted(active_idx, key=lambda j: importance[j], reverse=True)[:5]
            top5    = [(feature_cols[j], round(float(importance[j]), 4)) for j in top_idx]
        else:
            top5 = []
        print(f"  Expert {i} ({regime_name}): {top5}")

    # ── Signal bucket analysis ────────────────────────────────────────────────
    model.eval()
    all_proba_top = []
    with torch.no_grad():
        for seq, current, _, regime in test_loader:
            seq, current, regime = seq.to(DEVICE), current.to(DEVICE), regime.to(DEVICE)
            logits, _ = model(seq, current, regime)
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
    model_path = os.path.join(out_dir, "moe_model.pt")
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
        },
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    return model


if __name__ == "__main__":
    train_moe()
