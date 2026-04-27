"""
run_experiment.py
-----------------
CLI entry point for incremental feature experiments.

Usage
-----
# Run a single named experiment
python run_experiment.py --csv path/to/VCB-VNINDEX-History.csv --exp baseline
python run_experiment.py --csv path/to/VCB-VNINDEX-History.csv --exp returns
python run_experiment.py --csv path/to/VCB-VNINDEX-History.csv --exp volume
python run_experiment.py --csv path/to/VCB-VNINDEX-History.csv --exp technical

# Run all four stages sequentially and print the comparison table
python run_experiment.py --csv path/to/VCB-VNINDEX-History.csv --exp all

# Walk-forward options
python run_experiment.py --csv ... --exp all --method rolling --train 252 --test 63

Experiment ladder
-----------------
  baseline   stages=[baseline]                          1 feature
  returns    stages=[baseline, returns]                 6 features
  volume     stages=[baseline, returns, volume]        10 features
  technical  stages=[baseline, returns, volume, technical] 17 features

Each experiment is fully self-contained and logged to experiments/results/.
After running, use compare.py to see the performance progression.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make imports work whether run from src/build_factor/ or from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_csv
from feature_engineering.stages import (
    BaselineStage, ReturnsStage, VolumeStage, TechnicalStage,
)
from feature_engineering.base import FeaturePipeline
from models.ridge import RidgeModel
from evaluation.walk_forward import WalkForwardValidator
from evaluation.metrics import aggregate_fold_metrics
from experiments.tracker import ExperimentTracker
from experiments.compare import compare_experiments, print_comparison, feature_lift_table
from utils.seed import set_seed

# ─── Experiment definitions ───────────────────────────────────────────────────
# Each entry: experiment_name → ordered list of stages to activate
EXPERIMENT_LADDER: dict[str, list] = {
    "baseline":  [BaselineStage()],
    "returns":   [BaselineStage(), ReturnsStage()],
    "volume":    [BaselineStage(), ReturnsStage(), VolumeStage()],
    "technical": [BaselineStage(), ReturnsStage(), VolumeStage(), TechnicalStage()],
}


def run_single(
    exp_name: str,
    df,
    symbol: str,
    method: str,
    train_size: int,
    test_size: int,
    alpha: float,
    tracker: ExperimentTracker,
    verbose: bool = True,
) -> dict:
    """Run one experiment, log it, and return the aggregate metrics."""
    stages = EXPERIMENT_LADDER[exp_name]
    pipeline = FeaturePipeline(stages, target_horizon=1)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Experiment : {exp_name}")
        pipeline.describe()

    X, y = pipeline.build(df)

    if verbose:
        print(f"  Dataset    : {len(X)} rows × {X.shape[1]} features")

    validator = WalkForwardValidator(
        model_cls=lambda: RidgeModel(alpha=alpha),
        method=method,
        train_size=train_size,
        test_size=test_size,
    )
    fold_results = validator.validate(X, y)
    aggregate    = aggregate_fold_metrics(fold_results)

    run_id = tracker.log(
        experiment      = exp_name,
        stages          = [s.name for s in stages],
        features        = pipeline.feature_columns,
        model_name      = "RidgeModel",
        model_params    = {"alpha": alpha},
        validator_config= validator.config,
        fold_metrics    = fold_results,
        symbol          = symbol,
        n_rows          = len(X),
    )

    if verbose:
        _print_aggregate(exp_name, aggregate)
        print(f"  Saved      : {run_id}.json")

    return aggregate


def _print_aggregate(name: str, agg: dict) -> None:
    da_mean = agg.get("dir_acc_mean", float("nan"))
    da_std  = agg.get("dir_acc_std",  float("nan"))
    ic_mean = agg.get("ic_mean",      float("nan"))
    ic_std  = agg.get("ic_std",       float("nan"))
    mse     = agg.get("mse_mean",     float("nan"))
    n_folds = int(agg.get("n_folds",  0))

    print(f"\n  ┌─ Results ({n_folds} folds) {'─'*38}")
    print(f"  │  Directional Accuracy : {da_mean:.4f} ± {da_std:.4f}")
    print(f"  │  IC (Spearman)        : {ic_mean:.4f} ± {ic_std:.4f}")
    print(f"  │  MSE                  : {mse:.6f}")
    print(f"  └{'─'*52}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental feature experiment runner for Vietnam equities."
    )
    parser.add_argument("--csv",      required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--exp",      default="all",
                        choices=list(EXPERIMENT_LADDER) + ["all"],
                        help="Which experiment to run (default: all)")
    parser.add_argument("--method",   default="expanding", choices=["rolling", "expanding"])
    parser.add_argument("--train",    type=int, default=252,
                        help="Training window / min train size (default: 252 trading days)")
    parser.add_argument("--test",     type=int, default=63,
                        help="Test window per fold (default: 63 = ~1 quarter)")
    parser.add_argument("--alpha",    type=float, default=1.0,
                        help="Ridge regularisation alpha (default: 1.0)")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--results",  default=None,
                        help="Override results directory (default: experiments/results/)")

    args = parser.parse_args()
    set_seed(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {args.csv} ...")
    df     = load_csv(args.csv)
    symbol = df["Symbol"].iloc[0] if "Symbol" in df.columns else "unknown"
    print(f"  {len(df)} rows loaded  |  symbol={symbol}")

    # ── Tracker ───────────────────────────────────────────────────────────────
    results_dir = args.results or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "experiments", "results"
    )
    tracker = ExperimentTracker(results_dir=results_dir)

    # ── Run experiment(s) ─────────────────────────────────────────────────────
    to_run = list(EXPERIMENT_LADDER) if args.exp == "all" else [args.exp]

    all_results: dict[str, dict] = {}
    for exp_name in to_run:
        agg = run_single(
            exp_name  = exp_name,
            df        = df,
            symbol    = symbol,
            method    = args.method,
            train_size= args.train,
            test_size = args.test,
            alpha     = args.alpha,
            tracker   = tracker,
        )
        all_results[exp_name] = agg

    # ── Comparison summary ────────────────────────────────────────────────────
    if len(to_run) > 1:
        print(f"\n\n{'═'*60}")
        print("  FEATURE LIFT SUMMARY")
        print(f"{'═'*60}")
        lift_df = feature_lift_table(results_dir)
        if not lift_df.empty:
            for _, row in lift_df.iterrows():
                lift_str = (
                    f"+{row['lift_vs_prev']:.4f}" if row["lift_vs_prev"] is not None and row["lift_vs_prev"] > 0
                    else (f"{row['lift_vs_prev']:.4f}" if row["lift_vs_prev"] is not None else "  —   ")
                )
                print(
                    f"  {row['experiment']:<12} "
                    f"features={int(row['feature_count']):<3}  "
                    f"dir_acc={row['dir_acc_mean']:.4f}  "
                    f"ic={row['ic_mean']:.4f}  "
                    f"lift={lift_str}"
                )

        print(f"\n{'─'*60}")
        print_comparison(compare_experiments(results_dir))


if __name__ == "__main__":
    main()
