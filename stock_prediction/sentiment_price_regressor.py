from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stock_prediction.evaluator import save_metrics

try:
    from xgboost import XGBRegressor
except Exception as exc:
    raise ImportError(
        "xgboost is required for regression. Install it with: pip install xgboost"
    ) from exc


def run_regressor(
    daily: pd.DataFrame,
    output_dir: Path,
    seed: int,
    use_weighted_sentiment: bool = False,
    target_col: str = "next_day_return",
) -> None:
    # Simple time-ordered train/test split for return regression.
    output_dir.mkdir(parents=True, exist_ok=True)

    score_col = (
        "weighted_sentiment_score"
        if use_weighted_sentiment and "weighted_sentiment_score" in daily.columns
        else "sentiment_score"
    )

    feature_cols = [
        score_col,
        "article_count",
        "day_of_week",
        "rolling_sentiment_3",
        "rolling_sentiment_5",
        "daily_return",
        "lag_return_1",
        "lag_return_3",
        "rolling_return_mean_5",
        "rolling_return_mean_10",
        "rolling_vol_5",
        "rolling_vol_10",
    ]
    model_df = daily.dropna(subset=feature_cols + [target_col]).copy()
    model_df = model_df.sort_values(["published_date", "ticker"]).reset_index(drop=True)

    if model_df.empty:
        raise ValueError("No rows available for regression after feature engineering.")

    X = model_df[feature_cols]
    y = model_df[target_col]

    split_idx = int(len(model_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    reg = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
    )
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    denom = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = float(1.0 - (np.sum((y_test - preds) ** 2) / denom)) if denom > 0 else float("nan")

    reg_metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    save_metrics(reg_metrics, output_dir / "sentiment_price_regression_eval.json")
    save_metrics(reg_metrics, output_dir / "sentiment_price_regression_eval.csv")

    pred_out = model_df.iloc[split_idx:].copy()
    pred_out["predicted_return"] = preds
    pred_out.to_csv(output_dir / "sentiment_price_regression_predictions.csv", index=False)

    print(f"Saved regression metrics to {output_dir}")
