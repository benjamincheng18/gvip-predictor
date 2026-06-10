import pandas as pd
import numpy as np
import os
import pickle
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FEATURE_COLS = [
    "total_holders", "top10_count", "top10_ratio",
    "avg_portfolio_weight", "avg_rank", "total_value",
    "holders_qoq", "top10_qoq", "weight_qoq",
    "return_1m", "return_3m", "return_6m", "return_12m",
    "ma50_ratio", "ma200_ratio", "high52w_ratio",
    "volatility_3m", "volatility_ratio",
    "rel_volume", "dollar_volume",
    "rsi_14", "bollinger_position",
    "macd_histogram"
]


def load_model():
    model_path = os.path.join(PROJECT_ROOT, "data/features/gvip_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def run_backtest(df: pd.DataFrame, model, test_start_idx: int = 22) -> pd.DataFrame:
    """
    Run quarterly backtest on test set.
    For each quarter, predict top-50 stocks and measure overlap
    with actual GVIP proxy (actual top-50 by top10_count next quarter).
    """
    results = []

    test_quarters = df[df["quarter_idx"] >= test_start_idx][
        ["year", "quarter", "quarter_idx"]
    ].drop_duplicates().sort_values("quarter_idx")

    for _, row in test_quarters.iterrows():
        year = int(row["year"])
        quarter = int(row["quarter"])
        q_idx = int(row["quarter_idx"])

        # Current quarter data — used for prediction
        current = df[df["quarter_idx"] == q_idx].copy()

        if current.empty:
            continue

        # Get predictions
        X = current[FEATURE_COLS]
        current["proba"] = model.predict_proba(X)[:, 1]

        # Top-50 predicted stocks
        top50_predicted = set(
            current.nlargest(50, "proba")["cusip"].tolist()
        )

        # Actual GVIP proxy — top-50 by top10_count in NEXT quarter
        next_quarter = df[df["quarter_idx"] == q_idx + 1]
        if next_quarter.empty:
            continue

        top50_actual = set(
            next_quarter.nlargest(50, "top10_count")["cusip"].tolist()
        )

        # Overlap
        overlap = top50_predicted & top50_actual
        overlap_count = len(overlap)
        precision = overlap_count / 50
        recall = overlap_count / len(top50_actual) if top50_actual else 0

        results.append({
            "year": year,
            "quarter": quarter,
            "quarter_label": f"{year} Q{quarter}",
            "overlap": overlap_count,
            "precision": precision,
            "recall": recall,
            "predicted_cusips": list(top50_predicted),
            "actual_cusips": list(top50_actual),
        })

        print(f"{year} Q{quarter}: overlap={overlap_count}/50, "
              f"precision={precision:.2f}, recall={recall:.2f}")

    return pd.DataFrame(results)


def get_forward_return(ticker: str, from_date: str, to_date: str) -> float:
    """
    Compute forward return for a ticker between two dates.
    Uses cached price data.
    """
    price_cache_dir = os.path.join(PROJECT_ROOT, "data", "processed", "yfinance", "prices")
    cache_path = os.path.join(price_cache_dir, f"{ticker}.csv")

    if not os.path.exists(cache_path):
        return None

    try:
        hist = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        from_ts = pd.Timestamp(from_date)
        to_ts = pd.Timestamp(to_date)

        # Get closest available prices
        hist_from = hist[hist.index >= from_ts]
        hist_to = hist[hist.index <= to_ts]

        if hist_from.empty or hist_to.empty:
            return None

        price_start = hist_from.iloc[0]["Close"]
        price_end = hist_to.iloc[-1]["Close"]

        return float((price_end - price_start) / price_start) if price_start != 0 else None

    except Exception as e:
        return None


def compute_portfolio_returns(
    backtest_results: pd.DataFrame,
    df: pd.DataFrame,
    quarter_dates: dict
) -> pd.DataFrame:
    """
    Compute forward returns for predicted vs actual GVIP portfolios.
    quarter_dates: maps quarter_label to (start_date, end_date) tuple
    """
    ticker_map = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data/processed/cusip_ticker_map.csv")
    )
    cusip_to_ticker = dict(zip(ticker_map["cusip"], ticker_map["ticker"]))

    portfolio_results = []

    for _, row in backtest_results.iterrows():
        qlabel = row["quarter_label"]
        if qlabel not in quarter_dates:
            continue

        from_date, to_date = quarter_dates[qlabel]

        # Compute forward returns for predicted portfolio
        pred_returns = []
        for cusip in row["predicted_cusips"]:
            ticker = cusip_to_ticker.get(cusip)
            if ticker:
                ret = get_forward_return(ticker, from_date, to_date)
                if ret is not None:
                    pred_returns.append(ret)

        # Compute forward returns for actual GVIP portfolio
        actual_returns = []
        for cusip in row["actual_cusips"]:
            ticker = cusip_to_ticker.get(cusip)
            if ticker:
                ret = get_forward_return(ticker, from_date, to_date)
                if ret is not None:
                    actual_returns.append(ret)

        # Universe return — all stocks in feature matrix that quarter
        year, quarter = qlabel.split(" Q")
        q_idx = df[
            (df["year"] == int(year)) & (df["quarter"] == int(quarter))
        ]["quarter_idx"].iloc[0]
        universe_cusips = df[df["quarter_idx"] == q_idx]["cusip"].tolist()
        universe_returns = []
        for cusip in universe_cusips:
            ticker = cusip_to_ticker.get(cusip)
            if ticker:
                ret = get_forward_return(ticker, from_date, to_date)
                if ret is not None:
                    universe_returns.append(ret)

        pred_return = np.mean(pred_returns) if pred_returns else None
        actual_return = np.mean(actual_returns) if actual_returns else None
        universe_return = np.mean(universe_returns) if universe_returns else None

        portfolio_results.append({
            "quarter_label": qlabel,
            "from_date": from_date,
            "to_date": to_date,
            "predicted_return": pred_return,
            "actual_gvip_return": actual_return,
            "universe_return": universe_return,
            "alpha_vs_universe": (pred_return - universe_return)
                if pred_return and universe_return else None,
            "n_predicted_with_data": len(pred_returns),
            "n_actual_with_data": len(actual_returns),
        })

    return pd.DataFrame(portfolio_results)


if __name__ == "__main__":
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data/features/feature_matrix.csv"),
        low_memory=False
    )
    model = load_model()

    print("=" * 40)
    print("Running backtest on test set")
    print("=" * 40)

    backtest_df = run_backtest(df, model, test_start_idx=22)

    print("\n── Summary ──────────────────")
    print(f"Avg overlap:   {backtest_df['overlap'].mean():.1f}/50")
    print(f"Avg precision: {backtest_df['precision'].mean():.4f}")
    print(f"Avg recall:    {backtest_df['recall'].mean():.4f}")

    # Quarter forward return periods
    # From quarter-end to next quarter-end
    quarter_dates = {
        "2025 Q3": ("2025-09-30", "2025-12-31"),
        "2025 Q4": ("2025-12-31", "2026-03-31"),
    }

    print("\n── Portfolio Forward Returns ──────────────────")
    returns_df = compute_portfolio_returns(backtest_df, df, quarter_dates)
    print(returns_df[["quarter_label", "predicted_return", "actual_gvip_return",
                       "universe_return", "alpha_vs_universe",
                       "n_predicted_with_data"]].to_string(index=False))

    print(f"\nAvg predicted return:   {returns_df['predicted_return'].mean():.4f}")
    print(f"Avg actual GVIP return: {returns_df['actual_gvip_return'].mean():.4f}")
    print(f"Avg universe return:    {returns_df['universe_return'].mean():.4f}")
    print(f"Avg alpha vs universe:  {returns_df['alpha_vs_universe'].mean():.4f}")

    output_path = os.path.join(PROJECT_ROOT, "data/features/backtest_results.csv")
    backtest_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")