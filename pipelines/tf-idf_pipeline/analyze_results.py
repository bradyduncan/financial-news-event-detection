"""Analyze and visualize TF-IDF pipeline results."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from preprocessing.config import DEFAULT_SUBSET

LABEL_NAMES = ["negative", "neutral", "positive"]


def main():
    parser = argparse.ArgumentParser(description="Analyze TF-IDF pipeline results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="pipelines/tf-idf_pipeline/results",
        help="Base directory containing subset result folders",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=DEFAULT_SUBSET,
        help="Subset name used in filenames",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).resolve().parents[2] / results_dir

    results_dir = results_dir / args.subset
    summary_path = results_dir / "results_summary.csv"
    json_path = results_dir / "results.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")

    summary = pd.read_csv(summary_path)
    print("Results Summary")
    print(summary.to_string(index=False))

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("\nBest by macro_f1")
    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    print(best.to_string())

    print("\nPer-class F1 (by model)")
    for model_entry in data.get("models", []):
        name = model_entry.get("name", "unknown")
        f1 = model_entry.get("metrics", {}).get("per_class", {}).get("f1", [])
        print(f"{name}: {f1}")

    """Generate confusion matrix heatmaps for each model"""
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    for model_entry in data.get("models", []):
        name = model_entry.get("name", "unknown")
        cm = np.array(model_entry["metrics"]["confusion_matrix"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            ax=ax,
            annot_kws={"size": 14},
            vmin=0,
            vmax=1.0,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {name} ({args.subset})", fontsize=13)
        plt.tight_layout()
        plt.savefig(charts_dir / f"cm_{name}.png", dpi=150)
        plt.close()
        print(f"Saved: cm_{name}.png")

    print(f"\nCharts saved to: {charts_dir}")


if __name__ == "__main__":
    main()