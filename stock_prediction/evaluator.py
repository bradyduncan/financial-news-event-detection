from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def evaluate_sentiment_signal(
    df: pd.DataFrame, score_col: str, target_col: str
) -> dict[str, float]:
    data = df[[score_col, target_col]].dropna()
    if data.empty:
        return {
            "n": 0,
            "pearson_corr": float("nan"),
            "spearman_corr": float("nan"),
            "directional_accuracy": float("nan"),
            "mae": float("nan"),
        }

    scores = data[score_col]
    targets = data[target_col]

    pearson = float(scores.corr(targets, method="pearson"))
    spearman = float(scores.corr(targets, method="spearman"))

    directional_accuracy = float(((scores > 0) == (targets > 0)).mean())
    mae = float((targets - scores).abs().mean())

    return {
        "n": int(len(data)),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "directional_accuracy": directional_accuracy,
        "mae": mae,
    }


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return

    if path.suffix.lower() == ".csv":
        pd.DataFrame([metrics]).to_csv(path, index=False)
        return

    raise ValueError(f"Unsupported metrics file extension: {path.suffix}")
