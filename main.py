"""
GVIP Predictor — Main Pipeline Orchestrator
Runs the full pipeline from data collection to model training.

Usage:
    python3.11 main.py --step all          # Run everything
    python3.11 main.py --step ingest       # Data collection only
    python3.11 main.py --step features     # Feature engineering only
    python3.11 main.py --step train        # Model training only
"""

import argparse
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def run_ingest():
    """Step 1 — Collect 13F holdings from SEC EDGAR."""
    logger.info("=" * 50)
    logger.info("Step 1: Building fund universe")
    logger.info("=" * 50)
    from ingestion.fund_universe import get_13f_filers_for_quarter
    from ingestion.edgar_pipeline import run_pipeline
    import yaml
    import pandas as pd

    PROJECT_ROOT = os.path.dirname(__file__)
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    quarters_to_run = [tuple(q) for q in config["pipeline"]["quarters_to_run"]]

    # Generate filer universe for each quarter
    os.makedirs(os.path.join(PROJECT_ROOT, "data/raw/fund_universe"), exist_ok=True)
    for year, quarter in quarters_to_run:
        output_path = os.path.join(
            PROJECT_ROOT, f"data/raw/fund_universe/filers_{year}_Q{quarter}.csv"
        )
        if os.path.exists(output_path):
            logger.info("%d Q%d filers already exist, skipping", year, quarter)
            continue
        df = get_13f_filers_for_quarter(year, quarter)
        df.to_csv(output_path, index=False)

    # Run holdings pipeline
    logger.info("=" * 50)
    logger.info("Step 2: Collecting 13F holdings")
    logger.info("=" * 50)
    for year, quarter in quarters_to_run:
        run_pipeline(year, quarter)

    # Classify securities
    logger.info("=" * 50)
    logger.info("Step 3: Classifying securities via OpenFIGI")
    logger.info("=" * 50)
    from ingestion.security_classifier import classify_cusips
    import glob

    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data/raw/holdings_*.csv")))
    all_cusips = set()
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        all_cusips.update(df["cusip"].dropna().unique())

    output_path = os.path.join(PROJECT_ROOT, "data/processed/cusip_classifications.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        existing = pd.read_csv(output_path)
        already_done = set(existing["cusip"].astype(str).unique())
        remaining = [c for c in all_cusips if c not in already_done]
    else:
        remaining = list(all_cusips)

    if remaining:
        classified_df = classify_cusips(remaining)
        write_header = not os.path.exists(output_path)
        classified_df.to_csv(output_path, mode="a", header=write_header, index=False)


def run_features():
    """Step 2 — Feature engineering."""
    logger.info("=" * 50)
    logger.info("Step 4: Computing crowding features")
    logger.info("=" * 50)
    from features.feature_builder import build_quarterly_features
    import yaml
    import os

    PROJECT_ROOT = os.path.dirname(__file__)
    data_dir = os.path.join(PROJECT_ROOT, "data/raw")
    features_df = build_quarterly_features(data_dir)

    output_path = os.path.join(PROJECT_ROOT, "data/processed/crowding_features.csv")
    features_df.to_csv(output_path, index=False)
    logger.info("Crowding features saved: %s", output_path)

    logger.info("=" * 50)
    logger.info("Step 5: Caching prices and computing technical features")
    logger.info("=" * 50)
    from ingestion.yfinance_pipeline import cache_all_prices, compute_quarter_features
    import yaml

    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    quarters_to_run = [tuple(q) for q in config["pipeline"]["quarters_to_run"]]
    cache_all_prices(max_workers=8)

    for year, quarter in quarters_to_run:
        compute_quarter_features(year, quarter)

    logger.info("=" * 50)
    logger.info("Step 6: Building final feature matrix")
    logger.info("=" * 50)
    from features.merge_features import build_feature_matrix

    feature_matrix = build_feature_matrix()
    output_path = os.path.join(PROJECT_ROOT, "data/features/feature_matrix.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_matrix.to_csv(output_path, index=False)
    logger.info("Feature matrix saved: %s", output_path)


def run_train():
    """Step 3 — Model training and backtesting."""
    logger.info("=" * 50)
    logger.info("Step 7: Training XGBoost model")
    logger.info("=" * 50)
    from model.train import load_feature_matrix, train_model
    import pickle

    PROJECT_ROOT = os.path.dirname(__file__)
    df = load_feature_matrix()
    model, importance = train_model(df)

    model_path = os.path.join(PROJECT_ROOT, "data/features/gvip_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved: %s", model_path)

    logger.info("=" * 50)
    logger.info("Step 8: Running backtest")
    logger.info("=" * 50)
    from model.backtest import run_backtest, compute_portfolio_returns
    import pandas as pd

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/features/feature_matrix.csv"),
                     low_memory=False)

    backtest_df = run_backtest(df, model, test_start_idx=22)

    quarter_dates = {
        "2025 Q3": ("2025-09-30", "2025-12-31"),
        "2025 Q4": ("2025-12-31", "2026-03-31"),
    }

    returns_df = compute_portfolio_returns(backtest_df, df, quarter_dates)
    logger.info("\n%s", returns_df.to_string(index=False))

    output_path = os.path.join(PROJECT_ROOT, "data/features/backtest_results.csv")
    backtest_df.to_csv(output_path, index=False)
    logger.info("Backtest results saved: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="GVIP Predictor Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "ingest", "features", "train"],
        default="all",
        help="Pipeline step to run"
    )
    args = parser.parse_args()

    if args.step == "all":
        run_ingest()
        run_features()
        run_train()
    elif args.step == "ingest":
        run_ingest()
    elif args.step == "features":
        run_features()
    elif args.step == "train":
        run_train()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()