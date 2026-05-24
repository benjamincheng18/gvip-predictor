import pandas as pd
import os
import yaml
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.ingestion.yfinance_fetcher import fetch_ticker_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "features", "yfinance")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def fetch_quarter_features(
        tickers: list, 
        as_of_date: str, 
        cache_dir: str=CACHE_DIR, 
        force_refresh: bool = False,
        max_workers: int = 4,
        sleep_per_ticker: float = 0.3,
    ) -> pd.DataFrame:
    """
    Fetch features for all tickers for a given quarter-end date.
    On re-run, already-fetched tickers are skipped automatically.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"yfinance_{as_of_date}.csv")
 
    # Load existing checkpoint
    if os.path.exists(cache_file) and not force_refresh:
        existing = pd.read_csv(cache_file)
        done_tickers = set(existing["ticker"].tolist())
        remaining = [t for t in tickers if t not in done_tickers]
        logger.info(
            "Checkpoint loaded: %d done, %d remaining for %s",
            len(done_tickers), len(remaining), as_of_date
        )
        if not remaining:
            return existing
    else:
        existing = pd.DataFrame()
        remaining = tickers

    results = []
    failed = []

    def fetch_one(ticker): 
        try:
            return fetch_ticker_features(ticker, as_of_date, sleep=sleep_per_ticker)
        except Exception as e:
            logger.warning("Failed on %s: %s", ticker, e)
            return {"ticker": ticker, "as_of_date": as_of_date}
        
    # Use threads for I/O-bound yfinance calls
    # max_workers=4 is conservative; increase if not hitting rate limits
    checkpoint_every = 50
    batch_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, t): t for t in remaining}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            batch_results.append(row)

            # Save checkpointing every N tickers
            if i % checkpoint_every == 0:
                checkpoint_df = pd.concat(
                    [existing, pd.DataFrame(batch_results)], ignore_index=True
                )
                checkpoint_df.to_csv(cache_file, index=False)
                logger.info("Checkpoint saved: %d / %d tickers processed", i, len(remaining))
            
    # Final save
    all_results = pd.concat(
        [existing, pd.DataFrame(batch_results)], ignore_index=True
    ) if batch_results else existing
 
    all_results.to_csv(cache_file, index=False)
    logger.info(
        "Saved %d rows to %s | Missing %%:\n%s",
        len(all_results),
        cache_file,
        all_results.drop(columns=["ticker", "as_of_date", "sector"], errors="ignore")
            .isnull().mean().round(3).to_string()
    )
    return all_results


def quarter_end_date(year: int, quarter: int) -> str:
    """Return the last calendar day of a quarter as 'YYYY-MM-DD'."""
    end_month = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return f"{year}-{end_month[quarter]}"
 
 
def load_tickers_for_quarter(year: int, quarter: int) -> list:
    holdings_path = os.path.join(PROJECT_ROOT, "data", "raw", f"holdings_{year}_Q{quarter}.csv")
    classifications_path = os.path.join(PROJECT_ROOT, "data", "processed", "cusip_classifications.csv")
    
    holdings = pd.read_csv(holdings_path, usecols=["cusip"])
    classifications = pd.read_csv(classifications_path, usecols=["cusip", "ticker"])
    
    merged = holdings.merge(classifications, on="cusip", how="left")
    return merged["ticker"].dropna().unique().tolist()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    quarters_to_run = [tuple(q) for q in config["pipeline"]["quarters_to_run"]]

    for year, quarter in quarters_to_run:
        holdings_path = os.path.join(PROJECT_ROOT, "data", "raw", f"holdings_{year}_Q{quarter}.csv")

        if not os.path.exists(holdings_path):
            logger.warning("Holdings file not found, skipping: %s", holdings_path)
            continue

        tickers = load_tickers_from_holdings(holdings_path)
        as_of = quarter_end_date(year, quarter)

        print(f"\n{'='*40}")
        print(f"Fetching yfinance features {year} Q{quarter} ({len(tickers)} tickers)")
        print(f"{'='*40}")

        fetch_quarter_features(
            tickers=tickers,
            as_of_date=as_of,
        )