import pandas as pd
import os
import glob
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def compute_crowding_features(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw holdings for one quarter, compute stock-level crowding features.
    
    Features:
    - total_holders: number of funds holding the stock
    - top10_count: number of funds with stock in top-10 holdings by value
    - avg_portfolio_weight: average weight of stock across all holding funds
    - avg_rank: average rank of stock within holding funds' portfolios
    """
    results = []

    # Process each fund seperately
    for cik, fund_holdings in holdings_df.groupby("cik"):
        # Sort by value descending to get rank
        fund_holdings = fund_holdings.sort_values("value", ascending=False).reset_index(drop=True)
        fund_holdings['rank'] = fund_holdings.index + 1

        # Calculate portfolio weight for each position
        total_value = fund_holdings["value"].sum()
        fund_holdings['weight'] = fund_holdings['value'] / total_value

        # Flag top-10 holdings
        fund_holdings['in_top10'] = fund_holdings["rank"] <= 10

        results.append(fund_holdings)

    # Combine all funds
    all_holdings = pd.concat(results, ignore_index=True)

    # Aggregate to stock level
    features = all_holdings.groupby("cusip").agg(
        name=("name", "first"), 
        total_holders=("cik", "count"), 
        top10_count=("in_top10", "sum"), 
        avg_portfolio_weight=("weight", "mean"), 
        avg_rank=("rank", "mean"), 
        total_value=("value", "sum")
    ).reset_index()

    # Calculate top10 ratio
    features["top10_ratio"] = features["top10_count"] / features["total_holders"]

    return features.sort_values("top10_count", ascending=False)


def build_quarterly_features(data_dir: str) -> pd.DataFrame:
    """
    Build crowding features for all quarters and compute QoQ changes.
    """
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, "holdings_*.csv")))

    quarterly_features = []

    for f in files:
        # Extract year and quarter from filename
        basename = os.path.basename(f)
        parts = basename.replace("holdings_", "").replace(".csv", "").split("_")
        year = int(parts[0])
        quarter = int(parts[1].replace("Q", ""))

        print(f"Computing features for {year} Q{quarter}...")

        df = pd.read_csv(f, low_memory=False)
        features = compute_crowding_features(df)
        features['year'] = year
        features['quarter'] = quarter
        features['quarter_id'] = f"{year}_Q{quarter}"

        quarterly_features.append(features)

    # Combine all quarters
    all_features = pd.concat(quarterly_features, ignore_index=True)

    # Sort for QoQ calculation
    all_features = all_features.sort_values(["cusip", "year", "quarter"])

    # QoQ changes
    all_features["holders_qoq"] = all_features.groupby("cusip")["total_holders"].diff()
    all_features["top10_qoq"] = all_features.groupby("cusip")["top10_count"].diff()
    all_features["weight_qoq"] = all_features.groupby("cusip")["avg_portfolio_weight"].diff()

    return all_features


if __name__ == "__main__":
    data_dir = os.path.join(PROJECT_ROOT, "data/raw")
    features_df = build_quarterly_features(data_dir)
    
    print("\nShape:", features_df.shape)
    print("\nSample:")
    print(features_df[features_df["cusip"] == "037833100"])  # Apple