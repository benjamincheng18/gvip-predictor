import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import yaml
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Price / Momentum
def get_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily adjusted price history for a ticker.
    end is exclusive (yfinance convention) — pass as_of_date + 1 day.
    Returns empty DataFrame on failure.
    """
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)
        if hist.empty:
            return pd.DataFrame()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist
    except Exception as e:
        logger.debug("Price history error for %s: %s", ticker, e)
        return pd.DataFrame()
    

def compute_momentum_features(ticker: str, as_of_date: str) -> dict:
    """
    Compute point-in-time momentum and volatility features.
    Uses only data up to and including as_of_date.
 
    Returns dict with keys:
        ticker, as_of_date, return_3m, return_6m, return_12m, volatility_3m
    All numeric values are None on insufficient data (not NaN, for JSON safety).
    """
    result = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "return_3m": None,
        "return_6m": None,
        "return_12m": None,
        "volatility_3m": None,
    }
    try:
        as_of_ts = pd.Timestamp(as_of_date)
        # Fetch 13 months to cover 252 trading-day lookback
        start = (as_of_ts - pd.DateOffset(months=13)).strftime("%Y-%m-%d")
        # end is exclusive in yfinance; add 1 day to include as_of_date
        end = (as_of_ts + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

        hist = get_price_history(ticker, start, end)
        if hist.empty or len(hist) < 20:
            return result
        
        # Hard clip: never use data past as_of_date
        hist = hist[hist.index <= as_of_ts]
        closes = hist["Close"].dropna()
 
        if len(closes) < 20:
            return result
        
        current_price = closes.iloc[-1]

        def safe_return(trading_days: int):
            if len(closes) <= trading_days:
                return None
            past_price = closes.iloc[-trading_days - 1]
            return float((current_price - past_price)/past_price) if past_price != 0 else None
        
        result["return_3m"] = safe_return(63)
        result["return_6m"] = safe_return(126)
        result["return_12m"] = safe_return(252)
 
        # 3-month annualized realized volatility
        if len(closes) >= 64:
            daily_rets = closes.pct_change().dropna().iloc[-63:]
            result["volatility_3m"] = float(daily_rets.std() * (252 ** 0.5))
 
    except Exception as e:
        logger.debug("Momentum error for %s: %s", ticker, e)
 
    return result

# Fundamentals
def get_fundamental_features(ticker: str, as_of_date: str) -> dict:
    """
    Fetch point-in-time fundamental features.
    Uses the most recent quarterly filing strictly BEFORE as_of_date.
 
    Returns dict with keys:
        ticker, as_of_date, gross_margin, operating_margin, roe,
        debt_to_equity, revenue_growth_yoy, free_cashflow,
        market_cap, trailing_pe, price_to_book, sector
    """
    result = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "gross_margin": None,
        "operating_margin": None,
        "roe": None,
        "debt_to_equity": None,
        "revenue_growth_yoy": None,   # YoY (same quarter prior year) — less noisy than QoQ
        "free_cashflow": None,
        "market_cap": None,
        "trailing_pe": None,
        "price_to_book": None,
        "sector": None,
    }
    try:
        as_of_ts = pd.Timestamp(as_of_date)
        stock = yf.Ticker(ticker)

        income_stmt = stock.quarterly_income_stmt
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        if income_stmt is None or income_stmt.empty:
            return result
        if balance_sheet is None or balance_sheet.empty:
            return result
        
        # Filter columns to filings strictly BEFORE as_of_date (PIT discipline)
        def pit_cols(df):
            return [c for c in df.columns if pd.Timestamp(c) < as_of_ts]
        
        income_cols  = pit_cols(income_stmt)
        balance_cols = pit_cols(balance_sheet)
        cashflow_cols = pit_cols(cashflow) if cashflow is not None and not cashflow.empty else []
 
        if not income_cols or not balance_cols:
            return result
        
        # Most recent available quarter
        latest_income  = income_stmt[income_cols[0]]
        latest_balance = balance_sheet[balance_cols[0]]
        latest_cashflow = cashflow[cashflow_cols[0]] if cashflow_cols else None
 
        def get(series, key):
            val = series.get(key)
            return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None
 
        revenue         = get(latest_income, "Total Revenue")
        gross_profit    = get(latest_income, "Gross Profit")
        operating_income = get(latest_income, "Operating Income")
        net_income      = get(latest_income, "Net Income")
        total_debt      = get(latest_balance, "Total Debt")
        equity          = get(latest_balance, "Common Stock Equity")

        if revenue and revenue != 0:
            if gross_profit is not None:
                result["gross_margin"] = gross_profit / revenue
            if operating_income is not None:
                result["operating_margin"] = operating_income / revenue
 
        if equity and equity != 0:
            if net_income is not None:
                result["roe"] = net_income / equity
            if total_debt is not None:
                result["debt_to_equity"] = total_debt / abs(equity)
 
        if latest_cashflow is not None:
            result["free_cashflow"] = get(latest_cashflow, "Free Cash Flow")

        # YoY revenue growth: same quarter one year ago (4 quarters back)
        # More stable than QoQ for seasonal businesses
        if len(income_cols) >= 5 and revenue is not None:
            prior_year_revenue = get(income_stmt[income_cols[4]], "Total Revenue")
            if prior_year_revenue and prior_year_revenue != 0:
                result["revenue_growth_yoy"] = (revenue - prior_year_revenue) / abs(prior_year_revenue)
 
        # info fields: these reflect Yahoo's current snapshot (PIT limitation documented)
        info = stock.info or {}
        result["market_cap"]    = info.get("marketCap")
        result["trailing_pe"]   = info.get("trailingPE") or info.get("forwardPE")
        result["price_to_book"] = info.get("priceToBook")
        result["sector"]        = info.get("sector")
 
    except Exception as e:
        logger.debug("Fundamentals error for %s: %s", ticker, e)
 
    return result

def fetch_ticker_features(ticker: str, as_of_date: str, sleep: float = 0.3) -> dict:
    """Fetch and merge momentum + fundamental features for one ticker."""
    momentum    = compute_momentum_features(ticker, as_of_date)
    time.sleep(sleep)
    fundamentals = get_fundamental_features(ticker, as_of_date)
    time.sleep(sleep)
 
    merged = {**momentum}
    for k, v in fundamentals.items():
        if k not in ("ticker", "as_of_date"):
            merged[k] = v
    return merged