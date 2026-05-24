import requests
import pandas as pd
import time
import os
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_HEADERS = {
    "Content-Type": "application/json",
    "X-OPENFIGI-APIKEY": config['openfigi']['openfigi_api_key']
}


def classify_cusips(cusips: list) -> pd.DataFrame:
    """
    Given a list of CUSIPs, classify each as stock or ETF/fund
    using OpenFIGI API. Free, no API key required.
    
    OpenFIGI accepts batches of 50 CUSIPs per request.
    """
    results = []
    
    # Process in batches of 50
    for i in range(0, len(cusips), 10):
        batch = cusips[i:i+10]
        
        # Clean batch — ensure all are strings
        batch = [str(c).strip() for c in batch]
        
        payload = [{"idType": "ID_CUSIP", "idValue": cusip} for cusip in batch]
        
        try:
            response = requests.post(OPENFIGI_URL, json=payload, headers=OPENFIGI_HEADERS, timeout=10)
            time.sleep(1.0)  # increased sleep
            
            if response.status_code == 429:  # rate limited
                print("Rate limited, waiting 60s...")
                time.sleep(60)
                response = requests.post(OPENFIGI_URL, json=payload, headers=OPENFIGI_HEADERS, timeout=10)
            
            if response.status_code != 200:
                print(f"OpenFIGI error {response.status_code} on batch {i//50 + 1}")
                print(f"Response: {response.text[:200]}")
                continue
            
            data = response.json()
            
            for cusip, result in zip(batch, data):
                if "data" in result and result["data"]:
                    figi_data = result["data"][0]
                    results.append({
                        "cusip": cusip,
                        "security_type": figi_data.get("securityType", ""),
                        "security_type2": figi_data.get("securityType2", ""),
                        "market_sector": figi_data.get("marketSector", ""),
                        "name": figi_data.get("name", "")
                    })
                else:
                    results.append({
                        "cusip": cusip,
                        "security_type": "Unknown",
                        "security_type2": "",
                        "market_sector": "",
                        "name": ""
                    })
        
        except Exception as e:
            print(f"Error on batch {i//50 + 1}: {e}")
            continue
        
        if (i//50 + 1) % 10 == 0:
            print(f"Processed batch {i//50 + 1}/{(len(cusips)-1)//50 + 1}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import glob

    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data/raw/holdings_*.csv")))
    
    all_cusips = set()
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        all_cusips.update(df["cusip"].dropna().unique())
    
    output_path = os.path.join(PROJECT_ROOT, "data/processed/cusip_classifications.csv")    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Checkpointing — skip already classified CUSIPs
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        existing = pd.read_csv(output_path)
        already_done = set(existing["cusip"].astype(str).unique())
        print(f"Already classified: {len(already_done)}")
    else:
        already_done = set()
        print("Starting fresh")
    
    remaining_cusips = [c for c in all_cusips if c not in already_done]
    print(f"Remaining to classify: {len(remaining_cusips)}")
    
    if not remaining_cusips:
        print("All CUSIPs already classified")
    else:
        classified_df = classify_cusips(remaining_cusips)
        
        # Append to existing
        write_header = not os.path.exists(output_path)
        classified_df.to_csv(output_path, mode='a', header=write_header, index=False)
        print(f"Saved to {output_path}")
    
    # Summary
    final_df = pd.read_csv(output_path)
    print("\nSecurity type breakdown:")
    print(final_df["security_type"].value_counts())