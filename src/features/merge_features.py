import glob
import os

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def _load_technical_features() -> pd.DataFrame:
    tech_files = sorted(
        glob.glob(os.path.join(PROJECT_ROOT, "data/processed/yfinance/technical_*.csv"))
    )
    if not tech_files:
        raise FileNotFoundError(
            "No technical feature files found under data/processed/yfinance/"
        )

    tech_dfs = []
    for file_path in tech_files:
        basename = os.path.basename(file_path)
        parts = basename.replace("technical_", "").replace(".csv", "").split("_")
        if len(parts) != 2 or not parts[1].startswith("Q"):
            raise ValueError(f"Unexpected technical file name: {basename}")

        year = int(parts[0])
        quarter = int(parts[1].replace("Q", ""))

        df = pd.read_csv(file_path, low_memory=False)
        df["year"] = year
        df["quarter"] = quarter
        tech_dfs.append(df)

    tech_all = pd.concat(tech_dfs, ignore_index=True)

    # Ensure one row per ticker-year-quarter before merging.
    tech_all = tech_all.sort_values(["ticker", "year", "quarter"]).drop_duplicates(
        subset=["ticker", "year", "quarter"], keep="last"
    )

    return tech_all


def build_feature_matrix() -> pd.DataFrame:
    """
    Merge crowding features and technical features per quarter.

    Target definition:
    - is_next_q_top50 = 1 if the stock appears in the next quarter's top 50
      by top10_count.
    - This is a proxy label inspired by GVIP construction, not the official index.
    """
    crowding_path = os.path.join(PROJECT_ROOT, "data/processed/crowding_features.csv")
    ticker_map_path = os.path.join(PROJECT_ROOT, "data/processed/cusip_ticker_map.csv")

    crowding = pd.read_csv(crowding_path, low_memory=False)
    ticker_map = pd.read_csv(ticker_map_path, low_memory=False)

    print(f"Crowding features: {crowding.shape}")

    crowding = crowding.sort_values(["cusip", "year", "quarter"]).drop_duplicates(
        subset=["cusip", "year", "quarter"], keep="last"
    )
    ticker_map = ticker_map.drop_duplicates(subset=["cusip"], keep="last")

    crowding = crowding.merge(ticker_map, on="cusip", how="left", validate="many_to_one")

    tech_all = _load_technical_features()
    print(f"Technical features: {tech_all.shape}")

    merged = crowding.merge(
        tech_all,
        on=["ticker", "year", "quarter"],
        how="left",
        validate="many_to_one",
    )
    print(f"After merge: {merged.shape}")

    dup_mask = merged.duplicated(subset=["cusip", "year", "quarter"], keep=False)
    if dup_mask.any():
        dup_rows = merged.loc[dup_mask, ["cusip", "year", "quarter"]].head(20)
        raise ValueError(
            "Duplicate rows detected after merge for cusip-year-quarter. "
            f"Example duplicates:\n{dup_rows.to_string(index=False)}"
        )

    quarter_order = (
        merged[["year", "quarter"]]
        .drop_duplicates()
        .sort_values(["year", "quarter"])
        .reset_index(drop=True)
    )
    quarter_to_idx = {
        (int(row.year), int(row.quarter)): idx
        for idx, row in quarter_order.iterrows()
    }

    merged["quarter_idx"] = merged.apply(
        lambda r: quarter_to_idx[(int(r["year"]), int(r["quarter"]))], axis=1
    )

    top50_per_quarter = {}
    for (year, quarter), group in merged.groupby(["year", "quarter"], sort=True):
        top50 = group.nlargest(50, "top10_count")["cusip"].tolist()
        idx = quarter_to_idx[(int(year), int(quarter))]
        top50_per_quarter[idx] = set(top50)

    def get_label(row):
        next_idx = int(row["quarter_idx"]) + 1
        if next_idx not in top50_per_quarter:
            return np.nan
        return int(row["cusip"] in top50_per_quarter[next_idx])

    print("Computing label...")
    merged["is_next_q_top50"] = merged.apply(get_label, axis=1)

    merged["target"] = merged["is_next_q_top50"]

    merged = merged[merged["is_next_q_top50"].notna()].copy()
    merged["is_next_q_top50"] = merged["is_next_q_top50"].astype(int)
    merged["target"] = merged["target"].astype(int)

    print(f"\nFinal feature matrix: {merged.shape}")
    print("Label distribution:")
    print(merged["is_next_q_top50"].value_counts())
    print(f"Positive rate: {merged['is_next_q_top50'].mean():.4f}")

    return merged


if __name__ == "__main__":
    feature_matrix = build_feature_matrix()
    output_path = os.path.join(PROJECT_ROOT, "data/features/feature_matrix.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_matrix.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")