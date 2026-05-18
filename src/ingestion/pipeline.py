import pandas as pd
import os
import time
from edgar_fetcher import get_fund_filings, get_holdings_xml_url, parse_holdings_xml
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
        
        df_13f['reportDate'] = pd.to_datetime(df_13f['reportDate'])
        target_month = quarter * 3
        df_quarter = df_13f[
            (df_13f['reportDate'].dt.year == year) &
            (df_13f['reportDate'].dt.month == target_month)
        ]
        
        if df_quarter.empty:
            return pd.DataFrame()
        
        # Get holdings
        accession = df_quarter.iloc[0]['accessionNumber']
        xml_url = get_holdings_xml_url(accession, cik)
        if not xml_url:
            return pd.DataFrame()
        
        holdings = parse_holdings_xml(xml_url)
        if holdings.empty:
            return pd.DataFrame()
        
        # Aggregate duplicate CUSIPs
        holdings = holdings.groupby(['cusip', 'name']).agg(
            value = ("value", "sum"),
            shares = ("shares", "sum")
        ).reset_index()

        # Apply GS filters
        total_value = holdings["value"].sum()
        distinct_positions = holdings["cusip"].nunique()

        # Note: value in filing is in thousands
        if total_value < MIN_PORTFOLIO_VALUE:
            return pd.DataFrame()
        
        if not (MIN_POSITIONS <= distinct_positions <= MAX_POSITIONS):
            return pd.DataFrame()
        
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
    
    # Load filers list
    filers = pd.read_csv("data/raw/fund_universe/filers_2024_Q1.csv")
    
    # Test on 5 random funds
    sample = filers.sample(5, random_state=42)
    
    start = time.time()
    
    for _, row in sample.iterrows():
        print(f"Processing {row['company_name']} (CIK: {row['cik']})")
        result = process_fund(str(row['cik']), 2024, 1)
        print(f"  Qualifies: {not result.empty}")
    
    elapsed = time.time() - start
    print(f"\n5 funds took {elapsed:.1f} seconds")
    print(f"Estimated time for 7253 funds: {(elapsed/5 * 7253)/60:.1f} minutes")