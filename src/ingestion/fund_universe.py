import requests
import pandas as pd
import time
import os

HEADERS = {
    "User-Agent": "gvip-predictor benjamincheng18@gmail.com"
}

def get_13f_filers_for_quarter(year: int, quarter: int) -> pd.DataFrame:
    """
    Fetch all 13F-HR filers for a given quarter from EDGAR full index.
    Returns DataFrame with CIK and company name.
    """
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/company.idx"
    
    response = requests.get(url, headers=HEADERS)
    time.sleep(0.1)

    if response.status_code != 200:
        print(f"Failed to fetch index for {year} Q{quarter}")
        return pd.DataFrame()

    lines = response.text.splitlines()

    # Skip header lines (first 10 lines are headers and seperators)
    data_lines = lines[10:]

    records = []
    for line in data_lines:
        if "13F-HR" not in line:
            continue
        
        #Fixed width columns
        company_name = line[0:62].strip()
        form_type = line[62:74].strip()
        cik = line[74:86].strip()
        date_filed = line[86:98].strip()

        records.append({
            "company_name": company_name, 
            "cik": cik, 
            "date_filed": date_filed
        })
    
    df = pd.DataFrame(records)
    # Deduplicate by CIK — keep first occurrence
    df = df.drop_duplicates(subset=["cik"])

    print(f"{year} Q{quarter}: found {len(df)} 13F-HR filers")
    return df

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("data/raw/fund_universe", exist_ok=True)
    
    df = get_13f_filers_for_quarter(2024, 1)
    
    # Save to CSV
    output_path = "data/raw/fund_universe/filers_2024_Q1.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")