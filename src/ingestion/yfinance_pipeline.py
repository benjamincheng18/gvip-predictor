import pandas as pd
import os
import yaml
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yfinance_fetcher import fetch_and_cache_prices, compute_technical_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "yfinance")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def quarter_end_date(year: int, quarter: int) -> str:
    """Return the last calendar day of a quarter as 'YYYY-MM-DD'."""
    end_month = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return f"{year}-{end_month[quarter]}"


def load_all_tickers() -> list:
    """
    Load all unique tickers from cusip_ticker_map.csv.
    Used for the one-time price cache fetch.
    """
    ticker_map_path = os.path.join(PROJECT_ROOT, "data", "processed", "cusip_ticker_map.csv")
    ticker_map = pd.read_csv(ticker_map_path)
    tickers = ticker_map["ticker"].dropna().unique().tolist()
    logger.info("Total unique tickers to cache: %d", len(tickers))
    return tickers


def load_tickers_for_quarter(year: int, quarter: int) -> list:
    """
    Load unique tickers that appear in holdings for a given quarter.
    Cross-referenced with cusip_ticker_map.csv.
    """
    holdings_path = os.path.join(PROJECT_ROOT, "data", "raw", f"holdings_{year}_Q{quarter}.csv")
    ticker_map_path = os.path.join(PROJECT_ROOT, "data", "processed", "cusip_ticker_map.csv")

    if not os.path.exists(holdings_path):
        logger.warning("Holdings file not found: %s", holdings_path)
        return []

    holdings = pd.read_csv(holdings_path, usecols=["cusip"])
    ticker_map = pd.read_csv(ticker_map_path)

    merged = holdings.merge(ticker_map, on="cusip", how="left")
    tickers = merged["ticker"].dropna().unique().tolist()

    logger.info("%d unique tickers for %d Q%d", len(tickers), year, quarter)
    return tickers


def cache_all_prices(max_workers: int = 8) -> None:
    """
    One-time job: fetch and cache price history for all unique tickers.
    Skips tickers already cached. Run this once before computing features.
    """
    tickers = load_all_tickers()
    price_cache_dir = os.path.join(PROJECT_ROOT, "data", "processed", "yfinance", "prices")

    # Find already cached tickers
    already_cached = set()
    if os.path.exists(price_cache_dir):
        already_cached = {f.replace(".csv", "") for f in os.listdir(price_cache_dir)}

    remaining = [t for t in tickers if t not in already_cached]
    logger.info("Caching prices: %d done, %d remaining", len(already_cached), len(remaining))

    if not remaining:
        logger.info("All prices already cached")
        return

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_cache_prices, t): t for t in remaining}

        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            result = future.result()
            if result:
                success += 1
            else:
                failed += 1

            if i % 100 == 0:
                logger.info("Price cache progress: %d / %d (success=%d, failed=%d)",
                           i, len(remaining), success, failed)

    logger.info("Price caching complete: %d success, %d failed", success, failed)


def compute_quarter_features(
    year: int,
    quarter: int,
    max_workers: int = 8,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Compute technical features for all tickers in a given quarter.
    Saves to data/processed/yfinance/technical_{year}_Q{quarter}.csv
    """
    output_path = os.path.join(FEATURES_DIR, f"technical_{year}_Q{quarter}.csv")
    Path(FEATURES_DIR).mkdir(parents=True, exist_ok=True)

    # Checkpointing
    if os.path.exists(output_path) and not force_refresh:
        existing = pd.read_csv(output_path)
        done_tickers = set(existing["ticker"].tolist())
        logger.info("Checkpoint: %d tickers done for %d Q%d", len(done_tickers), year, quarter)
    else:
        existing = pd.DataFrame()
        done_tickers = set()

    tickers = load_tickers_for_quarter(year, quarter)
    remaining = [t for t in tickers if t not in done_tickers]

    if not remaining:
        logger.info("All tickers done for %d Q%d", year, quarter)
        return existing

    as_of = quarter_end_date(year, quarter)
    batch_results = []
    checkpoint_every = 100

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_technical_features, t, as_of): t
            for t in remaining
        }

        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            batch_results.append(row)

            if i % checkpoint_every == 0:
                checkpoint_df = pd.concat(
                    [existing, pd.DataFrame(batch_results)], ignore_index=True
                )
                checkpoint_df.to_csv(output_path, index=False)
                logger.info("Checkpoint: %d / %d tickers for %d Q%d",
                           i, len(remaining), year, quarter)

    # Final save
    all_results = pd.concat(
        [existing, pd.DataFrame(batch_results)], ignore_index=True
    ) if batch_results else existing

    all_results.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(all_results), output_path)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    quarters_to_run = [tuple(q) for q in config["pipeline"]["quarters_to_run"]]

    # Step 1 — cache all prices once
    print("=" * 40)
    print("Step 1: Caching all ticker prices")
    print("=" * 40)
    cache_all_prices(max_workers=8)

    # Step 2 — compute technical features per quarter
    print("\n" + "=" * 40)
    print("Step 2: Computing technical features per quarter")
    print("=" * 40)

    for year, quarter in quarters_to_run:
        print(f"\nProcessing {year} Q{quarter}...")
        compute_quarter_features(year, quarter)

    print("\nDone.")