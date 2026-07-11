import pandas as pd
import numpy as np
import os
import yaml
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Feature columns — all features except metadata and target
FEATURE_COLS = [
    # Crowding features
    "total_holders", "top10_count", "top10_ratio",
    "avg_portfolio_weight", "avg_rank", "total_value",
    "holders_qoq", "top10_qoq", "weight_qoq",
    # Technical features
    "return_1m", "return_3m", "return_6m", "return_12m",
    "ma50_ratio", "ma200_ratio", "high52w_ratio",
    "volatility_3m", "volatility_ratio",
    "rel_volume", "dollar_volume",
    "rsi_14", "bollinger_position",
    "macd_histogram"
]

# Technical-only feature set (no crowding features)
TECHNICAL_ONLY_COLS = [
    "return_1m", "return_3m", "return_6m", "return_12m",
    "ma50_ratio", "ma200_ratio", "high52w_ratio",
    "volatility_3m", "volatility_ratio",
    "rel_volume", "dollar_volume",
    "rsi_14", "bollinger_position",
    "macd_histogram"
]

# Train/validation/test split by quarter_idx
# 25 quarters total (idx 0-24)
# Train: idx 0-19 (2020 Q1 - 2024 Q4)
# Val:   idx 20-21 (2025 Q1 - 2025 Q2)
# Test:  idx 22-24 (2025 Q3 - 2026 Q1)
TRAIN_END_IDX = 19
VAL_END_IDX = 21


def load_feature_matrix() -> pd.DataFrame:
    path = os.path.join(PROJECT_ROOT, "data/features/feature_matrix.csv")
    df = pd.read_csv(path, low_memory=False)
    print(f"Feature matrix loaded: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    """Split into train, validation, test sets by quarter_idx."""
    train = df[df["quarter_idx"] <= TRAIN_END_IDX].copy()
    val = df[(df["quarter_idx"] > TRAIN_END_IDX) &
             (df["quarter_idx"] <= VAL_END_IDX)].copy()
    test = df[df["quarter_idx"] > VAL_END_IDX].copy()

    print(f"Train: {len(train)} rows, {train['target'].sum()} positives")
    print(f"Val:   {len(val)} rows, {val['target'].sum()} positives")
    print(f"Test:  {len(test)} rows, {test['target'].sum()} positives")

    return train, val, test


def evaluate(model, X, y, label: str):
    """Print evaluation metrics for a dataset."""
    preds_proba = model.predict_proba(X)[:, 1]
    preds = (preds_proba >= 0.5).astype(int)

    auc = roc_auc_score(y, preds_proba)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)

    print(f"\n── {label} ──────────────────")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return auc

def evaluate_topk(model, X, y, k: int = 50, label: str = ""):
    """
    Rank stocks by predicted probability, check top-K precision.
    This is the most relevant metric for our use case —
    we only care about the top 50 predictions.
    """
    probas = model.predict_proba(X)[:, 1]
    
    # Get indices of top-K predictions
    top_k_idx = np.argsort(probas)[::-1][:k]
    
    # How many of top-K are actual positives?
    top_k_precision = y.iloc[top_k_idx].sum() / k
    
    # How many actual positives are in top-K?
    top_k_recall = y.iloc[top_k_idx].sum() / y.sum()
    
    print(f"\n── Top-{k} Precision ({label}) ──")
    print(f"  Positives in top-{k}: {int(y.iloc[top_k_idx].sum())}/{k}")
    print(f"  Top-{k} Precision: {top_k_precision:.4f}")
    print(f"  Top-{k} Recall:    {top_k_recall:.4f}")
    
    return top_k_precision


def train_model(df: pd.DataFrame):
    """Train XGBoost classifier with walk-forward split."""
    train, val, test = split_data(df)

    X_train = train[FEATURE_COLS]
    y_train = train["target"]
    X_val = val[FEATURE_COLS]
    y_val = val["target"]
    X_test = test[FEATURE_COLS]
    y_test = test["target"]

    # Handle class imbalance
    # scale_pos_weight = negative samples / positive samples
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nClass imbalance ratio: {scale_pos_weight:.1f}")

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1
    )

    # Train with early stopping on validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluate
    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_val, y_val, "Validation")
    evaluate(model, X_test, y_test, "Test")
    evaluate_topk(model, X_val, y_val, k=50, label="Validation")
    evaluate_topk(model, X_test, y_test, k=50, label="Test")

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n── Feature Importance (top 10) ──")
    print(importance.head(10).to_string(index=False))

    return model, importance

def train_model_ablation(df: pd.DataFrame):
    """
    Train two models:
    - Model A: full features (crowding + technical)
    - Model B: technical only (no crowding features)
    
    This tests whether performance collapses without crowding features,
    honestly framing the persistence signal vs genuine alpha.
    """
    train, val, test = split_data(df)

    results = {}

    for name, features in [
        ("Full (crowding + technical)", FEATURE_COLS),
        ("Technical only (no crowding)", TECHNICAL_ONLY_COLS)
    ]:
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"{'='*50}")

        X_train = train[features]
        y_train = train["target"]
        X_val = val[features]
        y_val = val["target"]
        X_test = test[features]
        y_test = test["target"]

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_auc = evaluate(model, X_val, y_val, f"Validation ({name})")
        test_auc = evaluate(model, X_test, y_test, f"Test ({name})")
        evaluate_topk(model, X_val, y_val, k=50, label=f"Val Top-50 ({name})")
        evaluate_topk(model, X_test, y_test, k=50, label=f"Test Top-50 ({name})")

        results[name] = {
            "val_auc": val_auc,
            "test_auc": test_auc,
            "model": model
        }

    return results

if __name__ == "__main__":
    df = load_feature_matrix()
    results = train_model_ablation(df)

    print("\n── Ablation Summary ──────────────────")
    for name, r in results.items():
        print(f"{name}:")
        print(f"  Val AUC:  {r['val_auc']:.4f}")
        print(f"  Test AUC: {r['test_auc']:.4f}")

    # Save full model
    model_path = os.path.join(PROJECT_ROOT, "data/features/gvip_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(results["Full (crowding + technical)"]["model"], f)
    print(f"\nFull model saved to {model_path}")