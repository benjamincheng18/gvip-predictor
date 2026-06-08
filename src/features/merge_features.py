import pandas as pd
import numpy as np
import os
import glob
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def build_feature_matrix() -> pd.DataFrame:
    """
    Merge crowding features and technical features per quarter.
    Constructs target variable: in_gvip_next_quarter.
    
    Target = 1 if stock is in top 50 by top10_count in the NEXT quarter.
    This proxies GVIP index membership based on GS's methodology.
    """

    # Load crowding features
    crowding_path = os.path.join(PROJECT_ROOT, "data/processed/crowding_features.csv")
    crowding = pd.read_csv(crowding_path, low_memory=False)
    print(f"Crowding features: {crowding.shape}")

    # Load ticker map
    ticker_map_path = os.path.join(PROJECT_ROOT, "data/processed/cusip_ticker_map.csv")
    ticker_map = pd.read_csv(ticker_map_path)

    # Merge ticker into crowding features
    crowding = crowding.merge(ticker_map, on="cusip", how="left")
    tech_files = sorted(glob.glob(
        os.path.join(PROJECT_ROOT, "data/processed/yfinance/technical_*.csv")
    ))

    tech_dfs = []
    for f in tech_files:
        basename = os.path.basename(f)
        parts = basename.replace("technical_", "").replace(".csv", "").split("_")
        year = int(parts[0])
        quarter = int(parts[1].replace("Q", ""))

        df = pd.read_csv(f)
        df["year"] = year
        df["quarter"] = quarter
        tech_dfs.append(df)

    tech_all = pd.concat(tech_dfs, ignore_index=True)
    print(f"Technical features: {tech_all.shape}")

    # Merge crowding + technicals
    merged = crowding.merge(
        tech_all, 
        on=["ticker", "year", "quarter"],
        how="left"
    )
    print(f"After merge: {merged.shape}")

    # Construct target variable
    # For each quarter, identify top 50 stocks by top10_count
    # Target = 1 if stock is in top 50 in the NEXT quarter

    # Sort to ensure correct quarter ordering
    merged = merged.sort_values(["cusip", "year", "quarter"]).reset_index(drop=True)

    # Create a quarter index for easy shifting
    quarter_order = sorted(merged[["year", "quarter"]].drop_duplicates().values.tolist())
    quarter_to_idx = {(y, q): i for i, (y, q) in enumerate(quarter_order)}
    merged["quarter_idx"] = merged.apply(
        lambda r: quarter_to_idx[(r["year"], r["quarter"])], axis=1
    )

    # For each quarter, get top 50 CISIPs by top10_count
    top50_per_quarter = {}
    for (year, quarter), group in merged.groupby(["year", "quarter"]):
        top50 = group.nlargest(50, "top10_count")["cusip"].tolist()
        idx = quarter_to_idx[(year, quarter)]
        top50_per_quarter[idx] = set(top50)

    # Target: is this stock in top50 of the next quarter?
    def get_target(row):
        next_idx = row["quarter_idx"] + 1
        if next_idx not in top50_per_quarter:
            return np.nan
        return 1 if row["cusip"] in top50_per_quarter[next_idx] else 0

    print("Computing target variable...")
    merged["target"] = merged.apply(get_target, axis=1)

    # Drop last quarter — no next quarter to predict
    merged = merged[merged["target"].notna()].copy()
    merged["target"] = merged["target"].astype(int)

    print(f"\nFinal feature matrix: {merged.shape}")
    print(f"Target distribution:\n{merged['target'].value_counts()}")
    print(f"Positive rate: {merged['target'].mean():.4f}")

    return merged


if __name__ == "__main__":
    feature_matrix = build_feature_matrix()

    output_path = os.path.join(PROJECT_ROOT, "data/features/feature_matrix.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_matrix.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")