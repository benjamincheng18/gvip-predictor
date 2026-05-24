import yfinance as yf
import pandas as pd
import numpy as np
import os
import yaml
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PRICE_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "yfinance", "prices")


def fetch_and_cache_prices(ticker: str, force_refresh: bool = False) -> bool:
    """
    Download full price + volume history for a ticker from 2019 and save to disk.
    Returns True on success, False on failure.
    """
    os.makedirs(PRICE_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(PRICE_CACHE_DIR, f"{ticker}.csv")

    if os.path.exists(cache_path) and not force_refresh:
        return True

    try:
        hist = yf.Ticker(ticker).history(
            start="2019-01-01",
            auto_adjust=True,
            actions=False
        )
        if hist.empty:
            return False

        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist.to_csv(cache_path)
        return True

    except Exception as e:
        logger.debug("Price fetch error for %s: %s", ticker, e)
        return False


def compute_technical_features(ticker: str, as_of_date: str) -> dict:
    """
    Compute point-in-time technical features from cached price/volume data.
    Uses only data up to and including as_of_date — no look-ahead bias.

    Features:
        Momentum:     return_1m, return_3m, return_6m, return_12m
        Trend:        ma50_ratio, ma200_ratio, high52w_ratio
        Volatility:   volatility_3m, volatility_ratio
        Volume:       rel_volume, dollar_volume
        Mean Rev:     rsi_14, bollinger_position
        MACD:         macd_histogram
    """
    result = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        # Momentum
        "return_1m": None,
        "return_3m": None,
        "return_6m": None,
        "return_12m": None,
        # Trend
        "ma50_ratio": None,
        "ma200_ratio": None,
        "high52w_ratio": None,
        # Volatility
        "volatility_3m": None,
        "volatility_ratio": None,
        # Volume
        "rel_volume": None,
        "dollar_volume": None,
        # Mean reversion
        "rsi_14": None,
        "bollinger_position": None,
        # MACD
        "macd_histogram": None,
    }

    cache_path = os.path.join(PRICE_CACHE_DIR, f"{ticker}.csv")
    if not os.path.exists(cache_path):
        return result

    try:
        hist = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        as_of_ts = pd.Timestamp(as_of_date)

        # Strict PIT — only use data up to as_of_date
        hist = hist[hist.index <= as_of_ts].copy()

        if hist.empty or len(hist) < 20:
            return result

        closes = hist["Close"].dropna()
        volumes = hist["Volume"].dropna()
        current_price = closes.iloc[-1]

        # ── Momentum ──────────────────────────────────────────────
        def safe_return(trading_days: int):
            if len(closes) <= trading_days:
                return None
            past = closes.iloc[-trading_days - 1]
            return float((current_price - past) / past) if past != 0 else None

        result["return_1m"] = safe_return(21)
        result["return_3m"] = safe_return(63)
        result["return_6m"] = safe_return(126)
        result["return_12m"] = safe_return(252)

        # ── Trend ─────────────────────────────────────────────────
        if len(closes) >= 50:
            ma50 = closes.iloc[-50:].mean()
            result["ma50_ratio"] = float(current_price / ma50) if ma50 != 0 else None

        if len(closes) >= 200:
            ma200 = closes.iloc[-200:].mean()
            result["ma200_ratio"] = float(current_price / ma200) if ma200 != 0 else None

        if len(closes) >= 252:
            high_52w = closes.iloc[-252:].max()
            result["high52w_ratio"] = float(current_price / high_52w) if high_52w != 0 else None

        # ── Volatility ────────────────────────────────────────────
        daily_rets = closes.pct_change().dropna()

        if len(daily_rets) >= 63:
            vol_3m = daily_rets.iloc[-63:].std() * (252 ** 0.5)
            result["volatility_3m"] = float(vol_3m)

            if len(daily_rets) >= 126:
                vol_6m = daily_rets.iloc[-126:].std() * (252 ** 0.5)
                # Ratio > 1 means volatility increasing (avoid)
                # Ratio < 1 means volatility decreasing (attractive)
                result["volatility_ratio"] = float(vol_3m / vol_6m) if vol_6m != 0 else None

        # ── Volume ────────────────────────────────────────────────
        if len(volumes) >= 60:
            avg_vol_20 = volumes.iloc[-20:].mean()
            avg_vol_60 = volumes.iloc[-60:].mean()
            result["rel_volume"] = float(avg_vol_20 / avg_vol_60) if avg_vol_60 != 0 else None
            # Average daily dollar volume (liquidity proxy)
            result["dollar_volume"] = float(avg_vol_20 * current_price)

        # ── RSI 14 ────────────────────────────────────────────────
        if len(daily_rets) >= 14:
            gains = daily_rets.clip(lower=0).iloc[-14:]
            losses = (-daily_rets.clip(upper=0)).iloc[-14:]
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                result["rsi_14"] = float(100 - (100 / (1 + rs)))
            else:
                result["rsi_14"] = 100.0

        # ── Bollinger Band Position ───────────────────────────────
        # Position = (price - lower) / (upper - lower)
        # 0 = at lower band, 1 = at upper band, 0.5 = at middle
        if len(closes) >= 20:
            rolling_20 = closes.iloc[-20:]
            bb_mean = rolling_20.mean()
            bb_std = rolling_20.std()
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            band_width = bb_upper - bb_lower
            if band_width != 0:
                result["bollinger_position"] = float(
                    (current_price - bb_lower) / band_width
                )

        # ── MACD Histogram ────────────────────────────────────────
        # MACD = EMA12 - EMA26, Signal = EMA9 of MACD
        # Histogram = MACD - Signal
        # Positive = bullish momentum, Negative = bearish
        if len(closes) >= 35:
            ema12 = closes.ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = closes.ewm(span=26, adjust=False).mean().iloc[-1]
            macd_line = ema12 - ema26

            macd_series = closes.ewm(span=12, adjust=False).mean() - \
                          closes.ewm(span=26, adjust=False).mean()
            signal_line = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]

            result["macd_histogram"] = float(macd_line - signal_line)

    except Exception as e:
        logger.debug("Technical features error for %s: %s", ticker, e)

    return result


if __name__ == "__main__":
    ticker = "AAPL"

    print("Fetching and caching prices...")
    success = fetch_and_cache_prices(ticker)
    print(f"Cache success: {success}")

    print("\nComputing technical features for Q1 2024:")
    features = compute_technical_features(ticker, "2024-03-31")
    for k, v in features.items():
        print(f"  {k}: {v}")