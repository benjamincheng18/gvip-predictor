import sys
import os
import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from edgar_fetcher import get_fund_filings, get_holdings_xml_url, parse_holdings_xml, parse_cover_page

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MIN_PORTFOLIO_VALUE = config["fund_universe"]["min_portfolio_value_thousands"]
MIN_POSITIONS = config["fund_universe"]["min_positions"]
MAX_POSITIONS = config["fund_universe"]["max_positions"]
MAX_WORKERS = config["pipeline"]["max_workers"]

def process_fund(cik: str, year: int, quarter: int) -> pd.DataFrame:
    """
    Fetch and validate a single fund's holdings for a given quarter.
    Returns holdings DataFrame if fund qualifies, empty DataFrame if not.
    """
    try:
        fund_data = get_fund_filings(cik)
        if not fund_data:
            return pd.DataFrame()

        filings = fund_data["filings"]["recent"]
        df_filings = pd.DataFrame(filings)
        df_13f = df_filings[df_filings["form"] == "13F-HR"]

        if df_13f.empty:
            return pd.DataFrame()
        
        # Filter to the target quarter
        df_13f["reportDate"] = pd.to_datetime(df_13f["reportDate"])
        target_month = quarter * 3
        df_quarter = df_13f[
            (df_13f["reportDate"].dt.year == year) &
            (df_13f["reportDate"].dt.month == target_month)
        ]

        if df_quarter.empty:
            return pd.DataFrame()
        
        accession = df_quarter.iloc[0]["accessionNumber"]
        
        # Stage 1 — cover page pre-filter (fast, 1 API call)
        cover = parse_cover_page(accession, cik)
        if not cover:
            return pd.DataFrame()

        value_total = int(cover["value_total"]) if cover["value_total"] else 0
        entry_total = int(cover["entry_total"]) if cover["entry_total"] else 0

        if value_total < MIN_PORTFOLIO_VALUE:
            return pd.DataFrame()

        if not (MIN_POSITIONS <= entry_total <= MAX_POSITIONS):
            return pd.DataFrame()
        
        # Stage 2 — fetch full holdings (only if passes pre-filter)
        xml_url = get_holdings_xml_url(accession, cik)
        if not xml_url:
            return pd.DataFrame()

        holdings = parse_holdings_xml(xml_url)
        if holdings.empty:
            return pd.DataFrame()
        
        # Aggregate duplicate CUSIPs
        holdings = holdings.groupby(["cusip", "name"]).agg(
            value=("value", "sum"),
            shares=("shares", "sum")
        ).reset_index()
        
        # Add metadata
        holdings["cik"] = cik
        holdings["year"] = year
        holdings["quarter"] = quarter

        return holdings
    
    except Exception as e:
        print(f"Error processing CIK {cik}: {e}")
        return pd.DataFrame()
    

def run_pipeline(year: int, quarter: int):
    """
    Run the full pipeline for a given quarter.
    Fetches all 13F filers, filters to qualifying funds,
    and saves holdings to CSV with checkpointing.
    """
    print(f"Starting pipeline for {year} Q{quarter}")

    # Load filer universe for this quarter
    filers_path = os.path.join(PROJECT_ROOT, f"data/raw/fund_universe/filers_{year}_Q{quarter}.csv")

    if not os.path.exists(filers_path):
        print(f"Filers file not found: {filers_path}")
        print("Run fund_universe.py first to generate the filers list")
        return

    filers = pd.read_csv(filers_path)

    # Output paths
    output_path = os.path.join(PROJECT_ROOT, f"data/raw/holdings_{year}_Q{quarter}.csv")
    processed_path = os.path.join(PROJECT_ROOT, f"data/raw/processed_ciks_{year}_Q{quarter}.txt")

    # Checkpointing
    if os.path.exists(processed_path):
        with open(processed_path, "r") as f:
            processed_ciks = set(f.read().splitlines())
        print(f"Resuming — {len(processed_ciks)} CIKs already processed")
    else:
        processed_ciks = set()
        print("Starting fresh")

    remaining = filers[~filers["cik"].astype(str).isin(processed_ciks)]
    print(f"Remaining funds to process: {len(remaining)}")

    if remaining.empty:
        print("All funds already processed")
        return

    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_fund, str(row['cik']), year, quarter): row['cik']
            for _, row in remaining.iterrows()
        }

        for future in as_completed(futures):
            cik = futures[future]
            result = future.result()

            with open(processed_path, "a") as f:
                f.write(f"{cik}\n")

            if not result.empty:
                write_header = not os.path.exists(output_path)
                result.to_csv(output_path, mode='a', header=write_header, index=False)

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} minutes")

   
if __name__ == "__main__":
    quarters_to_run = [
        (2022, 1), (2022, 2), (2022, 3), (2022, 4),
        (2023, 1), (2023, 2), (2023, 3), (2023, 4),
        (2024, 1), (2024, 2), (2024, 3), (2024, 4),
        (2025, 1), (2025, 2), (2025, 3), (2025, 4),
        (2026, 1)
    ]

    for year, quarter in quarters_to_run:
        print(f"\n{'='*40}")
        print(f"Processing {year} Q{quarter}")
        print(f"{'='*40}")
        run_pipeline(year, quarter)