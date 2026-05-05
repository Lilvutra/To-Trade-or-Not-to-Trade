# To-Trade-or-Not-to-Trade

An quantitative pipeline for Vietnam stock data

## Features 
- **Data Preprocessing**
- **Feature engineering**
- **Feature selection**
- **Mixture of Expert model architecture**
- **Portfolio optimization**

## Project Structure 

```
To-Trade-or-Not-to-Trade/
│
├── data/
│   ├── data-vn-20230228/                   # Raw Vietnam market data
│   │   ├── stock-historical-data/          # OHLCV CSVs per stock (TICKER-INDEX-History.csv)
│   │   ├── financial-ratio/                # Financial ratios per stock
│   │   ├── dividend-history/               # Dividend history per stock
│   │   ├── industry-analysis/              # Industry classification per stock
│   │   ├── companies.csv                   # Company metadata
│   │   └── ticker-overview.csv             # Ticker reference table
│   │
│   ├── cross_sectional/                    # Fama-MacBeth regression outputs
│   │   ├── factor_returns.csv              # Daily factor return time series
│   │   ├── full_summary.csv                # Full-sample FM summary statistics
│   │   ├── regime_QUIET_BEAR.csv           # Per-regime FM results
|   |   ├── regime_QUIET_BEAR.csv
│   │   ├── regime_PANIC_BEAR.csv
│   │   ├── regime_QUIET_BULL.csv
│   │   ├── regime_VOLATILE_BULL.csv
│   │   └── plot_*.png                      # Diagnostic plots (t-stats, Sharpe, heatmaps)
│   │
│   ├── ohlcv_encoder.pt                    # Pre-trained OHLCV LSTM autoencoder weights
│   ├── regime_moe_model.pt                 # Trained MoE model (baseline)
│   ├── regime_moe_latent_model.pt          # Trained MoE model (with OHLCV latents)
│   ├── dynamic_factor_model.pt             # Dynamic factor model weights
│   ├── moe_model.pt                        # Earlier MoE checkpoint
│   ├── moe_model_v2.pt                     # Earlier MoE checkpoint v2
│   └── *.png                               # Training curves and regime comparison plots
│
├── src/
│   ├── build/                              # Core pipeline scripts
│   │   ├── get_regime.py                   # Regime detection + regime-conditional features
│   │   ├── get_regime_features.py          # Feature engineering utilities
│   │   ├── get_macro.py                    # Macro data features
│   │   ├── get_commodity.py                # Commodity data features
│   │   ├── run_cross_sectional.py          # Fama-MacBeth cross-sectional regression
│   │   ├── plot_cross_sectional.py         # Cross-sectional diagnostic plots
│   │   ├── plot_regime.py                  # Regime visualisation plots
│   │   ├── ohlcv_encoder.py                # LSTM autoencoder for raw OHLCV sequences
│   │   ├── train_regime_expert.py          # Regime-Aware MoE training (baseline)
│   │   ├── train_regime_expert_.py         # Regime-Aware MoE training (+ OHLCV latents)
│   │   ├── train_dynamic_factor.py         # Dynamic factor model training
│   │   └── train_mixoe_2.py                # Earlier MoE experiment
│   │
│   ├── build_factor/                       # Factor research framework
│   │   ├── data_loader.py                  # Panel data loading utilities
│   │   ├── run_experiment.py               # Experiment runner
│   │   ├── feature_engineering/            # Modular feature pipeline
│   │   │   ├── base.py                     # Base feature class
│   │   │   ├── registry.py                 # Feature registry
│   │   │   └── stages/                     # Feature stages (returns, volume, technical)
│   │   ├── models/                         # Model definitions
│   │   │   ├── base.py                     # Base model class
│   │   │   └── ridge.py                    # Ridge regression model
│   │   ├── evaluation/                     # Evaluation utilities
│   │   │   ├── metrics.py                  # Performance metrics
│   │   │   └── walk_forward.py             # Walk-forward validation
│   │   ├── experiments/                    # Experiment tracking
│   │   │   ├── tracker.py                  # Experiment tracker
│   │   │   ├── compare.py                  # Experiment comparison
│   │   │   └── results/                    # Saved experiment results
│   │   └── utils/
│   │       └── seed.py                     # Random seed utilities
│   │
│   ├── fetch/                              # Data fetching scripts
│   │   ├── fetch_all_vn_data.py            # Fetch raw Vietnam market data
│   │   └── consolidate_all_stocks.py       # Consolidate per-stock CSVs
│   │
│   └── test/                               # Test scripts
│
└── README.md
```

## Model Architecture 
Recent version of the model architecture for the pipeline is as below:


![Current MoE Architecture](MoE_arch.drawio.png)

The model is a **Regime-Aware Mixture of Experts (MoE)** that routes each stock-day sample to a specialised expert based on the precomputed market regime.

**Input** — three streams are prepared per sample:
- `feat_seq [B, 20, 14]` — a 20-day rolling window of 14 engineered features (e.g. `delta_dist`, `smart_money_up`, `vol_accel`)
- `current_feat [B, 14]` — the same features at the prediction day only
- `regime` — the precomputed market regime label (QUIET_BEAR / PANIC_BEAR / QUIET_BULL / VOLATILE_BULL)

**SharedLSTMEncoder** — the feature sequence is passed through a single shared LSTM that reads all 20 days and compresses them into a 64-dim temporal context vector `h [B, 64]`, capturing trends and momentum over the past month.

**Concat** — `current_feat`, and `h` are concatenated into a single vector `x [B, dim]` that combines today's snapshot with the 20-day temporal context.

**ExpertMLP (×4, soft routing)** — all four regime experts process `x` independently, each producing `logits [B, 3]`. The outputs are stacked into `all_logits [B, 4, 3]`, then multiplied by a one-hot weight vector derived from the regime label and summed — selecting only the matched expert's prediction. Each expert is a two-layer MLP: `Linear → LayerNorm → ReLU → Dropout(0.1) → Linear → logits [B, 3]`.

**SharedMLP residual** — a global MLP also processes `x` and produces `shared_logits [B, 3]` capturing cross-regime patterns that are predictive in all market conditions.

**Output** — the final prediction is `expert_logits + 0.3 × shared_logits`, giving a three-class signal: **sell** (predicted 10-day return < −3%), **hold**, or **buy** (predicted 10-day return > +5%).

## Routing Modes

The router decides which expert(s) contribute to the final prediction. Three modes are available, controlled by `ROUTING_MODE` in `train_regime_expert.py`.

### Hard routing
Each sample is sent to exactly one expert — the one whose index matches the precomputed regime label. All other experts are bypassed entirely.

```
regime = PANIC_BEAR (1)  →  only Expert 1 computes logits
gradient flows through Expert 1 only
```

**When to use**: when you trust the regime detector fully and want strict specialisation. Each expert trains only on its own regime's data and cannot borrow patterns from others. This is the most interpretable mode — you can inspect each expert in isolation.

**Downside**: at regime transition points (e.g. the market is shifting from QUIET_BEAR to PANIC_BEAR), the hard switch can produce discontinuous predictions.

---

### Soft routing
All four experts always compute logits. Their outputs are blended using a weight vector derived from the regime label — currently a one-hot vector, so numerically the result is the same as hard routing, but the gradient flows through all four experts scaled by their weight.

```
regime = PANIC_BEAR (1)  →  weights = [0, 1, 0, 0]
final_logit = 0·Expert0 + 1·Expert1 + 0·Expert2 + 0·Expert3
gradient still reaches Expert 0, 2, 3 (scaled by 0 — but they stay in the graph)
```

**When to use**: when you want all experts to remain active and learn from every batch. With an external regime probability vector (e.g. `[0.1, 0.7, 0.1, 0.1]` from a probabilistic classifier), soft routing naturally handles regime uncertainty by blending expert outputs proportionally.

**Downside**: slightly slower (four forward passes instead of one) and the non-dominant experts receive zero gradient in practice when a one-hot is used, so they still specialise — just via a different code path than hard routing.

---

### Blend routing
A hybrid: uses **hard** routing when the model is confident about the regime, and **soft** routing when it is uncertain. Confidence is measured as `max(regime_probs)` — if it exceeds `CONFIDENCE_THRESHOLD` (default 0.70), hard routing is used; otherwise all experts blend.

```
regime_probs = [0.05, 0.80, 0.10, 0.05]  →  max=0.80 ≥ 0.70  →  hard (Expert 1)
regime_probs = [0.30, 0.40, 0.20, 0.10]  →  max=0.40 < 0.70  →  soft (weighted blend)
```

**When to use**: when the upstream regime detector outputs a probability distribution rather than a hard label — typically at turning points between regimes. Blend lets the model hedge at uncertain transitions while still committing to one expert when the regime is clear.

**Downside**: requires an external probabilistic regime classifier to be meaningful. If only a hard regime label is available, blend degrades to pure hard routing (no uncertainty signal to act on).














