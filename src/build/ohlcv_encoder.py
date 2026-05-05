"""
ohlcv_encoder.py
----------------
LSTM autoencoder for raw OHLCV sequences.

Pre-trains on reconstruction loss. The encoder half is frozen
and used to produce per-row latent vectors for train_regime_expert_.py.

Input normalization (applied before encoding, makes output scale-invariant):
  open_r, high_r, low_r, close_r  →  pct-change relative to previous close
  vol_r                            →  log(vol / rolling_20_vol_mean + EPS)

Architecture:
  Encoder : LSTM(5, HIDDEN_ENC) → last hidden → Linear → latent [LATENT_DIM]
  Decoder : Linear → unsqueeze → LSTM(LATENT_DIM, HIDDEN_ENC) → Linear(5)
              → reconstruct [SEQ_LEN, 5]
  Loss    : MSE between normalized input and reconstruction

Usage
-----
  # pre-train on a list of CSV files
  encoder, scaler = train_ohlcv_encoder(csv_files, save_path="data/ohlcv_encoder.pt")

  # extract latents for one stock DataFrame
  latent_cols = [f"lat_{i}" for i in range(LATENT_DIM)]
  df[latent_cols] = extract_latents(df, encoder, scaler)
"""

from __future__ import annotations

import os
import sys
import glob
import types


# torch._dynamo → torch.onnx → import transformers (real pkg) → version crash.
# Stub must be in sys.modules before the first Adam/optimizer instantiation.
def _stub_transformers() -> None:
    if "transformers" in sys.modules:
        return
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OHLCV_SEQ_LEN = 20          # window of past days fed to the encoder
LATENT_DIM    = 32          # dimensionality of the latent vector
HIDDEN_ENC    = 64          # LSTM hidden size
EPOCHS_ENC    = 50
BATCH_SIZE    = 256
LR_ENC        = 1e-3
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"
EPS           = 1e-8

OHLCV_COLS    = ["Open", "High", "Low", "Close", "Volume"]
N_OHLCV       = 5          # number of normalized features per step


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw OHLCV columns to scale-invariant, stationary features.

    Returns a DataFrame with columns: open_r, high_r, low_r, close_r, vol_r
    The first row will be NaN (pct-change) and the first 20 rows of vol_r
    will be NaN (rolling mean not yet available).
    """
    out = pd.DataFrame(index=df.index)
    prev_close = df["Close"].shift(1)
    out["open_r"]  = df["Open"]   / (prev_close + EPS) - 1
    out["high_r"]  = df["High"]   / (prev_close + EPS) - 1
    out["low_r"]   = df["Low"]    / (prev_close + EPS) - 1
    out["close_r"] = df["Close"]  / (prev_close + EPS) - 1
    vol_mean       = df["Volume"].rolling(20, min_periods=5).mean()
    out["vol_r"]   = np.log(df["Volume"] / (vol_mean + EPS) + EPS)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class OHLCVWindowDataset(Dataset):
    """
    Sliding-window dataset over normalized OHLCV features.
    Each sample is a window of shape [OHLCV_SEQ_LEN, N_OHLCV].
    Target = input itself (reconstruction / autoencoder).
    """

    def __init__(self, norm_array: np.ndarray, seq_len: int = OHLCV_SEQ_LEN) -> None:
        self.data    = torch.tensor(norm_array, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx : idx + self.seq_len]   # [seq_len, N_OHLCV]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────────────────────────────────────

class OHLCVEncoder(nn.Module):
    """
    LSTM encoder: [B, seq_len, N_OHLCV] → [B, LATENT_DIM]
    Uses the final hidden state as the sequence summary.
    """

    def __init__(
        self,
        n_input:    int = N_OHLCV,
        hidden:     int = HIDDEN_ENC,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.lstm    = nn.LSTM(n_input, hidden, batch_first=True)
        self.project = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)          # h_n: [1, B, hidden]
        return self.project(h_n.squeeze(0)) # [B, latent_dim]


class OHLCVDecoder(nn.Module):
    """
    Decoder: [B, LATENT_DIM] → [B, seq_len, N_OHLCV]
    Expands the latent vector, repeats it across time, then decodes with LSTM.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden:     int = HIDDEN_ENC,
        n_output:   int = N_OHLCV,
        seq_len:    int = OHLCV_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.seq_len  = seq_len
        self.expand   = nn.Linear(latent_dim, hidden)
        self.lstm     = nn.LSTM(hidden, hidden, batch_first=True)
        self.out      = nn.Linear(hidden, n_output)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.expand(z)                                  # [B, hidden]
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)      # [B, seq_len, hidden]
        out, _ = self.lstm(h)                               # [B, seq_len, hidden]
        return self.out(out)                                # [B, seq_len, N_OHLCV]


class OHLCVAutoencoder(nn.Module):
    def __init__(
        self,
        n_input:    int = N_OHLCV,
        hidden:     int = HIDDEN_ENC,
        latent_dim: int = LATENT_DIM,
        seq_len:    int = OHLCV_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.encoder = OHLCVEncoder(n_input, hidden, latent_dim)
        self.decoder = OHLCVDecoder(latent_dim, hidden, n_input, seq_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z    = self.encoder(x)      # [B, latent_dim]
        recon = self.decoder(z)     # [B, seq_len, N_OHLCV]
        return recon, z


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pre-training
# ─────────────────────────────────────────────────────────────────────────────

def _build_training_array(csv_files: list[str]) -> np.ndarray:
    """Load multiple stock CSVs, normalize, concatenate into one flat array."""
    segments = []
    for i, path in enumerate(csv_files):
        if (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(csv_files)} files …", flush=True)
        try:
            raw = pd.read_csv(path)
            raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])
            raw = raw.sort_values("TradingDate").reset_index(drop=True)
            normed = normalize_ohlcv(raw).dropna()
            if len(normed) >= OHLCV_SEQ_LEN + 10:
                segments.append(normed.values.astype(np.float32))
        except Exception:
            continue
    if not segments:
        raise ValueError("No valid CSV files found.")
    return np.concatenate(segments, axis=0)


def train_ohlcv_encoder(
    csv_files:  list[str],
    save_path:  str | None = None,
    epochs:     int  = EPOCHS_ENC,
    batch_size: int  = BATCH_SIZE,
    lr:         float = LR_ENC,
    device:     str  = DEVICE,
) -> OHLCVEncoder:
    """
    Pre-train OHLCV autoencoder on reconstruction loss (MSE).
    Returns the encoder (decoder is discarded after training).

    Parameters
    ----------
    csv_files : list of paths to stock CSV files
    save_path : if given, save encoder weights + scaler here (.pt)
    """
    print(f"Pre-training OHLCV encoder on {len(csv_files)} files  |  device={device}", flush=True)

    raw_array = _build_training_array(csv_files)
    print(f"Training array: {raw_array.shape[0]:,} rows  →  "
          f"{raw_array.shape[0] - OHLCV_SEQ_LEN + 1:,} windows", flush=True)

    # Global z-score scaling so MSE loss has consistent scale across features
    scaler    = StandardScaler()
    scaled    = scaler.fit_transform(raw_array).astype(np.float32)

    dataset   = OHLCVWindowDataset(scaled)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model     = OHLCVAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"{'Epoch':>6}  {'MSE Loss':>10}", flush=True)
    print("-" * 20, flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / len(dataset)
            print(f"{epoch:6d}  {avg:10.6f}", flush=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({
            "encoder_state": model.encoder.state_dict(),
            "scaler":        scaler,
            "config": {
                "n_input":    N_OHLCV,
                "hidden":     HIDDEN_ENC,
                "latent_dim": LATENT_DIM,
                "seq_len":    OHLCV_SEQ_LEN,
            },
        }, save_path)
        print(f"Encoder saved → {save_path}", flush=True)

    return model.encoder


def load_ohlcv_encoder(path: str, device: str = DEVICE) -> tuple[OHLCVEncoder, StandardScaler]:
    """Load a pre-trained encoder from disk."""
    ckpt    = torch.load(path, map_location=device)
    cfg     = ckpt["config"]
    encoder = OHLCVEncoder(cfg["n_input"], cfg["hidden"], cfg["latent_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    return encoder, ckpt["scaler"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Latent extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(
    df:         pd.DataFrame,
    encoder:    OHLCVEncoder,
    scaler:     StandardScaler,
    seq_len:    int  = OHLCV_SEQ_LEN,
    batch_size: int  = 1024,
    device:     str  = DEVICE,
) -> np.ndarray:
    """
    Compute one latent vector per row using a sliding window of `seq_len` days.

    Returns a float32 array of shape [len(df), LATENT_DIM].
    Rows before index `seq_len - 1` are filled with zeros (no full window yet).
    """
    normed = normalize_ohlcv(df).values.astype(np.float32)
    # fill NaN with 0 so the encoder still runs (early rows get zero-vectors anyway)
    normed = np.nan_to_num(normed, nan=0.0)
    normed = scaler.transform(normed).astype(np.float32)

    n      = len(normed)
    latent_dim = encoder.project.out_features
    latents    = np.zeros((n, latent_dim), dtype=np.float32)

    encoder.eval()
    windows, indices = [], []
    for i in range(seq_len - 1, n):
        windows.append(normed[i - seq_len + 1 : i + 1])
        indices.append(i)

    # batch inference
    for start in range(0, len(windows), batch_size):
        batch = torch.tensor(
            np.stack(windows[start : start + batch_size]), dtype=torch.float32
        ).to(device)
        z = encoder(batch).cpu().numpy()
        for j, idx in enumerate(indices[start : start + batch_size]):
            latents[idx] = z[j]

    return latents


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point — pre-train standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=150,
                        help="Cap number of stocks for training (default 150, use 0 for all)")
    args = parser.parse_args()

    DATA_DIR  = os.path.join(os.getcwd(), "data", "data-vn-20230228", "stock-historical-data")
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*-VNINDEX-History.csv")))
    csv_files = all_files[:args.max_stocks] if args.max_stocks > 0 else all_files
    print(f"Found {len(all_files)} VNINDEX stocks  |  using {len(csv_files)}", flush=True)

    encoder = train_ohlcv_encoder(
        csv_files,
        save_path=os.path.join(os.getcwd(), "data", "ohlcv_encoder.pt"),
    )
    print("Done.", flush=True)
