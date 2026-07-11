# GVIP Predictor

A machine learning pipeline that predicts which stocks will appear in the **Goldman Sachs Hedge Fund VIP Index (GVIP)** next quarter, based on SEC 13F filings analysis and technical indicators.

---

## Overview

The Goldman Sachs Hedge Fund VIP Index tracks the 50 stocks that appear most frequently in the top-10 holdings of fundamentally-driven hedge funds, rebalanced quarterly. This project replicates and predicts the index construction process using publicly available SEC 13F filings and price data.

**Key question:** Given hedge fund holdings data and technical indicators from this quarter, which 50 stocks will appear in GVIP next quarter?

---

## Methodology

### 1. Data Collection
- **SEC EDGAR API** — directly fetches 13F filings (no third-party wrappers) for all funds filing with $1B+ AUM and 10–200 distinct equity positions across 25 quarters (2020 Q1 → 2026 Q1)
- **OpenFIGI API** — classifies securities by type, filtering out ETFs, funds, and bonds
- **yfinance** — fetches full price history for technical indicator computation

### 2. Feature Engineering
**Crowding features (from 13F filings):**
- `top10_count` — number of funds holding stock in top-10 positions
- `total_holders` — total number of qualifying funds holding the stock
- `top10_ratio` — fraction of holders with stock in top-10
- `avg_portfolio_weight` — average portfolio weight across holders
- Quarter-over-quarter changes for all metrics

**Technical indicators (point-in-time, no look-ahead bias):**
- Momentum: 1M, 3M, 6M, 12M returns
- Trend: 50-day/200-day MA ratio, 52-week high ratio
- Volatility: 3M realized volatility, volatility ratio
- Volume: relative volume, dollar volume
- Mean reversion: RSI-14, Bollinger Band position
- MACD histogram

### 3. Target Variable
A stock is labeled `1` if it appears in the **top 50 stocks by hedge fund top-10 appearances** in the next quarter — our proxy for GVIP index membership.

### 4. Model
**XGBoost classifier** with walk-forward validation:
- Train: 2020 Q1 → 2024 Q4 (20 quarters)
- Validation: 2025 Q1 → 2025 Q2 (2 quarters)
- Test: 2025 Q3 → 2025 Q4 (2 quarters)

Class imbalance handled via `scale_pos_weight` (~66:1 negative to positive ratio).

---

## Results

### Ablation Study

Two models were trained to honestly assess signal sources:

| Model | Val AUC | Test AUC | Top-50 Precision |
|---|---|---|---|
| Full (crowding + technical) | 0.9994 | 0.9989 | 100% (50/50) |
| Technical only (no crowding) | 0.8970 | 0.8917 | 68% (34/50) |

**Interpretation:**

The full model's near-perfect performance is primarily driven by **crowding persistence** — stocks heavily owned by hedge funds this quarter tend to remain heavily owned next quarter. This is a real and exploitable signal, but not a surprising one.

The more credible result is the **technical-only model**: using only price-based indicators (momentum, trend, volatility, volume, RSI, Bollinger Bands, MACD), the model correctly identifies **34 of 50 actual GVIP constituents** — well above the random baseline of ~2-3 correct picks from a universe of 5,800+ stocks.

The gap between 100% and 68% top-50 precision quantifies the pure persistence contribution of crowding features.

### Backtest (2025 Q3, Full Model)

| Metric | Value |
|---|---|
| Overlap with actual GVIP | 43/50 (86%) |
| Predicted portfolio return | +3.15% |
| Actual GVIP return | +8.72% |
| Universe return | +1.68% |
| **Alpha vs universe** | **+1.47%** |

### Feature Importance (Full Model)

| Feature | Importance |
|---|---|
| `top10_count` | 55.5% |
| `total_holders` | 13.9% |
| `dollar_volume` | 7.8% |
| `total_value` | 5.0% |
| `rsi_14` | 2.0% |

Crowding features account for ~74% of importance, confirming that persistence is the dominant signal. Technical indicators contribute the remaining ~26%, providing independent predictive power as demonstrated by the technical-only ablation.

---

## Project Structure

````text
gvip-predictor/
├── src/
│   ├── ingestion/
│   │   ├── edgar_fetcher.py        # SEC EDGAR API — 13F XML parsing
│   │   ├── edgar_pipeline.py       # Parallel ingestion pipeline with checkpointing
│   │   ├── fund_universe.py        # Quarterly 13F filer universe construction
│   │   ├── security_classifier.py  # OpenFIGI CUSIP classification
│   │   ├── yfinance_fetcher.py     # Technical indicator computation
│   │   └── yfinance_pipeline.py    # Price cache and feature pipeline
│   ├── features/
│   │   ├── feature_builder.py      # Crowding feature engineering
│   │   └── merge_features.py       # Feature matrix construction + target variable
│   └── model/
│       ├── train.py                # XGBoost training + walk-forward evaluation
│       └── backtest.py             # Quarterly backtest + portfolio return simulation
├── data/
│   ├── raw/                        # 13F holdings CSVs (gitignored)
│   ├── processed/                  # Cleaned features, classifications (gitignored)
│   └── features/                   # Final feature matrix, model (gitignored)
├── notebooks/                      # EDA and data inspection
├── config.yaml                     # Pipeline parameters
├── requirements.txt
└── README.md
````

---

## Setup

**Prerequisites:** Python 3.11, Homebrew (Mac)

**1. Clone the repository:**
```bash
git clone https://github.com/benjamincheng18/gvip-predictor.git
cd gvip-predictor
```

**2. Create virtual environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Configure API keys:**
```bash
cp .env.example .env
```
Edit `.env` with your credentials:
EDGAR_USER_AGENT=your-project your@email.com
OPENFIGI_API_KEY=your_openfigi_key

**4. Run the pipeline (in order):**
```bash
# Step 1 — Build fund universe
python3.11 src/ingestion/fund_universe.py

# Step 2 — Collect 13F holdings (runs overnight)
nohup python3.11 -u src/ingestion/edgar_pipeline.py > pipeline.log 2>&1 &

# Step 3 — Classify securities
python3.11 src/ingestion/security_classifier.py

# Step 4 — Cache prices and compute technical features
python3.11 src/ingestion/yfinance_pipeline.py

# Step 5 — Build feature matrix
python3.11 src/features/feature_builder.py
python3.11 src/features/merge_features.py

# Step 6 — Train model and backtest
python3.11 src/model/train.py
python3.11 src/model/backtest.py
```

---

## Known Limitations

- **13F filing lag** — filings are due 45 days after quarter-end, so predictions are based on data that is 45–135 days old by the time the next quarter begins
- **$1B AUM threshold** — uses a higher threshold than GS's $100M to reduce noise from wealth managers; configurable in `config.yaml`
- **Fundamental features excluded** — yfinance only provides ~5 recent quarters of financials, making point-in-time fundamental features infeasible without paid data
- **Single backtest quarter** — limited test period due to data availability; results should be interpreted cautiously
- **Ticker mapping errors** — OpenFIGI occasionally returns incorrect tickers for some CUSIPs (e.g. MELI mapped as MLB1)
- **Bear market inconsistency** — fewer funds qualify during 2020–2022 due to lower AUM from market drawdowns, creating inconsistency in training data

---

## Technologies

- **Python 3.11** — pandas, numpy, xgboost, scikit-learn, yfinance, lxml
- **SEC EDGAR API** — free, no authentication required
- **OpenFIGI API** — free tier, CUSIP security classification
- **yfinance** — price history and technical indicator computation