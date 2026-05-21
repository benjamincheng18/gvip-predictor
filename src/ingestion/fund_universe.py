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
    os.makedirs(os.path.join("data/raw/fund_universe"), exist_ok=True)

    quarters_to_run = [
        (2022, 1), (2022, 2), (2022, 3), (2022, 4),
        (2023, 1), (2023, 2), (2023, 3), (2023, 4),
        (2024, 1), (2024, 2), (2024, 3), (2024, 4),
        (2025, 1), (2025, 2), (2025, 3), (2025, 4),
        (2026, 1)
    ]

    for year, quarter in quarters_to_run:
        output_path = f"data/raw/fund_universe/filers_{year}_Q{quarter}.csv"

        if os.path.exists(output_path):
            print(f"{year} Q{quarter}: already exists, skipping")
            continue

        df = get_13f_filers_for_quarter(year, quarter)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")