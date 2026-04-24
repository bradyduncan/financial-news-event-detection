"""Comparison dashboard: TF-IDF vs FinBERT across all subsets."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

TFIDF_RESULTS = REPO_ROOT / "pipelines" / "tf-idf_pipeline" / "results"
FINBERT_RESULTS = REPO_ROOT / "pipelines" / "finbert_pipeline" / "results"
OUTPUT_DIR = REPO_ROOT / "analysis" / "dashboard_charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = [
    "sentences_75agree",
]

LABEL_NAMES = ["negative", "neutral", "positive"]


def load_results(base_dir, subset):
    """Load results.json for a given pipeline and subset."""
    json_path = base_dir / subset / "results.json"
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_best_model(results):
    """Get the model with the highest macro_f1 from results."""
    if results is None:
        return None
    best = None
    best_f1 = -1
    for model_entry in results.get("models", []):
        f1 = model_entry.get("metrics", {}).get("macro_f1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best = model_entry
    return best


def collect_all_results():
    """Collect best model results from both pipelines across all subsets."""
    rows = []
    for subset in SUBSETS:
        tfidf_data = load_results(TFIDF_RESULTS, subset)
        finbert_data = load_results(FINBERT_RESULTS, subset)

        tfidf_best = get_best_model(tfidf_data)
        finbert_best = get_best_model(finbert_data)

        if tfidf_best:
            m = tfidf_best["metrics"]
            rows.append({
                "pipeline": "TF-IDF",
                "subset": subset.replace("sentences_", ""),
                "model": tfidf_best["name"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "roc_auc_ovr": m.get("roc_auc_ovr"),
                "f1_negative": m["per_class"]["f1"][0],
                "f1_neutral": m["per_class"]["f1"][1],
                "f1_positive": m["per_class"]["f1"][2],
                "confusion_matrix": m["confusion_matrix"],
            })

        if finbert_best:
            m = finbert_best["metrics"]
            rows.append({
                "pipeline": "FinBERT",
                "subset": subset.replace("sentences_", ""),
                "model": finbert_best["name"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "roc_auc_ovr": m.get("roc_auc_ovr"),
                "f1_negative": m["per_class"]["f1"][0],
                "f1_neutral": m["per_class"]["f1"][1],
                "f1_positive": m["per_class"]["f1"][2],
                "confusion_matrix": m["confusion_matrix"],
            })

    return pd.DataFrame(rows)


def plot_macro_f1_comparison(df):
    """Bar chart comparing macro-F1 across subsets for both pipelines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    subsets = df["subset"].unique()
    x = np.arange(len(subsets))
    width = 0.35

    tfidf = df[df["pipeline"] == "TF-IDF"]
    finbert = df[df["pipeline"] == "FinBERT"]

    tfidf_vals = [tfidf[tfidf["subset"] == s]["macro_f1"].values[0]
                  if len(tfidf[tfidf["subset"] == s]) > 0 else 0 for s in subsets]
    finbert_vals = [finbert[finbert["subset"] == s]["macro_f1"].values[0]
                    if len(finbert[finbert["subset"] == s]) > 0 else 0 for s in subsets]

    bars1 = ax.bar(x - width / 2, tfidf_vals, width, label="TF-IDF", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, finbert_vals, width, label="FinBERT", color="#DD8452")

    ax.bar_label(bars1, fmt="%.2f", fontsize=9)
    ax.bar_label(bars2, fmt="%.2f", fontsize=9)

    ax.set_xlabel("Dataset Subset")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 Comparison: TF-IDF vs FinBERT")
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "macro_f1_comparison.png", dpi=150)
    plt.close()
    print("Saved: macro_f1_comparison.png")


def plot_accuracy_comparison(df):
    """Bar chart comparing accuracy across subsets for both pipelines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    subsets = df["subset"].unique()
    x = np.arange(len(subsets))
    width = 0.35

    tfidf = df[df["pipeline"] == "TF-IDF"]
    finbert = df[df["pipeline"] == "FinBERT"]

    tfidf_vals = [tfidf[tfidf["subset"] == s]["accuracy"].values[0]
                  if len(tfidf[tfidf["subset"] == s]) > 0 else 0 for s in subsets]
    finbert_vals = [finbert[finbert["subset"] == s]["accuracy"].values[0]
                    if len(finbert[finbert["subset"] == s]) > 0 else 0 for s in subsets]

    bars1 = ax.bar(x - width / 2, tfidf_vals, width, label="TF-IDF", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, finbert_vals, width, label="FinBERT", color="#DD8452")

    ax.bar_label(bars1, fmt="%.2f", fontsize=9)
    ax.bar_label(bars2, fmt="%.2f", fontsize=9)

    ax.set_xlabel("Dataset Subset")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison: TF-IDF vs FinBERT")
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()
    print("Saved: accuracy_comparison.png")


def plot_per_class_f1(df):
    """Grouped bar chart showing per-class F1 for each pipeline and subset."""
    subsets = df["subset"].unique()

    for subset in subsets:
        subset_df = df[df["subset"] == subset]
        if len(subset_df) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(LABEL_NAMES))
        width = 0.35

        for i, pipeline in enumerate(["TF-IDF", "FinBERT"]):
            row = subset_df[subset_df["pipeline"] == pipeline]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            vals = [row["f1_negative"], row["f1_neutral"], row["f1_positive"]]
            offset = -width / 2 if i == 0 else width / 2
            color = "#4C72B0" if i == 0 else "#DD8452"
            bars = ax.bar(x + offset, vals, width, label=pipeline, color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=9)

        ax.set_xlabel("Class")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Per-Class F1: TF-IDF vs FinBERT ({subset})")
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_NAMES)
        ax.legend()
        ax.set_ylim(0, 1.15)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"per_class_f1_{subset}.png", dpi=150)
        plt.close()
        print(f"Saved: per_class_f1_{subset}.png")


def plot_confusion_matrices(df):
    """Side-by-side confusion matrix heatmaps for each subset."""
    subsets = df["subset"].unique()

    for subset in subsets:
        subset_df = df[df["subset"] == subset]
        pipelines = subset_df["pipeline"].tolist()
        n_plots = len(pipelines)

        if n_plots == 0:
            continue

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for ax, pipeline in zip(axes, pipelines):
            row = subset_df[subset_df["pipeline"] == pipeline].iloc[0]
            cm = np.array(row["confusion_matrix"])
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            sns.heatmap(
                cm_pct,
                annot=True,
                fmt=".1f",
                cmap="Blues" if pipeline == "TF-IDF" else "Oranges",
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES,
                ax=ax,
                vmin=0,
                vmax=100,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{pipeline} ({subset}) — %")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"confusion_matrix_{subset}.png", dpi=150)
        plt.close()
        print(f"Saved: confusion_matrix_{subset}.png")


def plot_f1_trend_across_subsets(df):
    """Line chart showing how macro-F1 changes across subsets (noise levels)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for pipeline in ["TF-IDF", "FinBERT"]:
        pipe_df = df[df["pipeline"] == pipeline].sort_values("subset")
        if len(pipe_df) == 0:
            continue
        color = "#4C72B0" if pipeline == "TF-IDF" else "#DD8452"
        ax.plot(
            pipe_df["subset"],
            pipe_df["macro_f1"],
            marker="o",
            linewidth=2,
            label=pipeline,
            color=color,
        )
        for _, row in pipe_df.iterrows():
            ax.annotate(
                f"{row['macro_f1']:.2f}",
                (row["subset"], row["macro_f1"]),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=9,
                ha="center",
            )

    ax.set_xlabel("Dataset Subset (decreasing agreement →)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 Trend: Label Quality vs Model Performance")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_trend_across_subsets.png", dpi=150)
    plt.close()
    print("Saved: f1_trend_across_subsets.png")


def main():
    print("Collecting results from both pipelines...\n")
    df = collect_all_results()

    if df.empty:
        print("No results found. Make sure both pipelines have been run.")
        return

    print("Results found:")
    print(df[["pipeline", "subset", "model", "macro_f1", "accuracy"]].to_string(index=False))
    print()

    print("Generating charts...\n")
    plot_macro_f1_comparison(df)
    plot_accuracy_comparison(df)
    plot_per_class_f1(df)
    plot_confusion_matrices(df)
    plot_f1_trend_across_subsets(df)

    """Save combined results table as CSV"""
    df.drop(columns=["confusion_matrix"]).to_csv(
        OUTPUT_DIR / "combined_results.csv", index=False
    )
    print("\nSaved: combined_results.csv")
    print(f"\nAll charts saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()