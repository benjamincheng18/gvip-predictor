import pandas as pd
import os
import time
from edgar_fetcher import get_fund_filings, get_holdings_xml_url, parse_holdings_xml, parse_cover_page
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# GS GVIP qualifying criteria
MIN_PORTFOLIO_VALUE = 100_000   # $100M represented in thousands 
MIN_POSITIONS = 10
MAX_POSITIONS = 200


def process_fund(cik: str, year: int, quarter: int) -> pd.DataFrame:
    """
    Fetch and validate a single fund's holdings for a given quarter.
    Returns holdings DataFrame if fund qualifies, empty DataFrame if not.
    """
    try:
        # Get all filings for this fund
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

   
if __name__ == "__main__":
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    YEAR = 2024
    QUARTER = 1
    
    filers = pd.read_csv("data/raw/fund_universe/filers_2024_Q1.csv")
    filers = filers.head(20)  # comment this out for full run
    
    output_path = f"data/raw/holdings_{YEAR}_Q{QUARTER}.csv"
    processed_path = f"data/raw/processed_ciks_{YEAR}_Q{QUARTER}.txt"
    
    # Load already processed CIKs from tracking file
    if os.path.exists(processed_path):
        with open(processed_path, "r") as f:
            processed_ciks = set(f.read().splitlines())
        print(f"Resuming — {len(processed_ciks)} CIKs already processed")
    else:
        processed_ciks = set()
        print("Starting fresh")
    
    remaining = filers[~filers["cik"].astype(str).isin(processed_ciks)]
    print(f"Remaining funds to process: {len(remaining)}")
    
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(process_fund, str(row['cik']), YEAR, QUARTER): row['cik']
            for _, row in remaining.iterrows()
        }
        
        for future in as_completed(futures):
            cik = futures[future]
            result = future.result()
            
            # Save to tracking file regardless of qualification
            with open(processed_path, "a") as f:
                f.write(f"{cik}\n")
            
            if not result.empty:
                write_header = not os.path.exists(output_path)
                result.to_csv(output_path, mode='a', header=write_header, index=False)
                print(f"CIK {cik}: qualified, saved {len(result)} holdings")
    
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")