# Hedge Fund Crowding Ranker
Inspired by the construction methodology behind Goldman Sachs' Hedge Fund VIP (GVIP) portfolio.

## Overview

Institutional investors often exhibit persistent crowding in a relatively small set of stocks. Inspired by Goldman Sachs' Hedge Fund VIP (GVIP) portfolio, this project investigates whether publicly available SEC 13F filings can be used to rank stocks that are likely to become the next quarter's most crowded hedge fund positions.

Rather than attempting to replicate the proprietary GVIP index, this project formulates the problem as a supervised machine learning ranking task using quarterly hedge fund ownership features together with technical market indicators.

**Given this quarter’s hedge-fund crowding data and technical indicators, which stocks are most likely to appear in next quarter’s crowding top 50?**

This is not an exact replication of the GVIP index. Instead, it is a proxy-ranking problem built from public data.

## What the model predicts

A stock is labeled `1` if it appears in the **top 50 by hedge-fund top-10 appearances in the next quarter**.
That label is used as a practical proxy for next-quarter crowding strength.

## Data sources

* **SEC EDGAR 13F filings** for fund holdings
* **OpenFIGI** for security classification and ticker mapping
* **yfinance** for historical prices and technical indicators

## Features

### Crowding features

From 13F filings, the pipeline builds features such as:

* `total_holders`
* `top10_count`
* `top10_ratio`
* `avg_portfolio_weight`
* quarter-over-quarter changes in crowding measures

### Technical features

Point-in-time market features include:

* 1M / 3M / 6M / 12M returns
* moving-average ratios
* 52-week high ratio
* realized volatility
* relative volume
* dollar volume
* RSI
* Bollinger Band position
* MACD histogram

## Model

The main model is an **XGBoost classifier** with walk-forward validation.

The project compares:

* a **full model** using crowding + technical features
* a **technical-only model** with crowding features removed

This ablation is intentional: it separates persistence in hedge-fund ownership from signal coming from market data.

## Evaluation

The model is evaluated with:

* AUC
* precision / recall / F1
* top-50 precision
* quarterly backtest overlap
* forward return comparison for predicted vs actual portfolios

## Repository structure

```text
src/
├── ingestion/
│   ├── edgar_fetcher.py
│   ├── edgar_pipeline.py
│   ├── fund_universe.py
│   ├── security_classifier.py
│   ├── yfinance_fetcher.py
│   └── yfinance_pipeline.py
├── features/
│   ├── feature_builder.py
│   └── merge_features.py
└── model/
    ├── train.py
    └── backtest.py

main.py
config.yaml
requirements.txt
```

## How to run

### 1. Install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file with your EDGAR user agent and OpenFIGI API key.

### 3. Run the pipeline

Run everything:

```bash
python3.11 main.py --step all
```

Run one stage at a time:

```bash
python3.11 main.py --step ingest
python3.11 main.py --step features
python3.11 main.py --step train
```

## Results

| Experiment               | Result      |
| ------------------------ | ----------- |
| Full Model               | AUC ≈ 0.999 |
| Technical Only           | AUC ≈ 0.89  |
| Top-50 Precision         | 100%        |
| Average Crowding Overlap | 87%         |

The full model demonstrates that hedge fund crowding exhibits strong quarter-to-quarter persistence. Removing crowding features substantially reduces predictive performance, indicating that technical indicators alone capture only part of the signal.

## Key Findings

• Hedge fund crowding is highly persistent across quarters.

• Historical ownership features contribute substantially more predictive power than technical indicators alone.

• Technical indicators still provide incremental predictive information, achieving approximately 0.89 AUC without ownership features.

• Public SEC filings can effectively model institutional crowding despite reporting delays.

## Future Work

- Compare against a naive persistence baseline.
- Explore LightGBM and CatBoost.
- Extend the backtest over additional market cycles.
- Investigate SHAP values for feature interpretability.
- Evaluate portfolio construction methods beyond Top-50 ranking.

## Project goal

This project is best understood as a practical ranking system for hedge-fund crowding, with GVIP construction as the inspiration for the target design.
