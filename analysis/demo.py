from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from pipelines.finbert_pipeline.classifiers import (
    HAS_XGBOOST,
    train_linear_svm,
    train_logistic_regression,
    train_xgboost,
)
from pipelines.shared.data_loading import get_splits, load_label_names
from pipelines.shared.evaluation import evaluate_model
from preprocessing.config import DEFAULT_OUTPUT_DIR, DEFAULT_SEED, DEFAULT_SUBSET, DEFAULT_TEST_SIZE


def _save_confusion_matrices(results: dict, results_dir: Path) -> None:
    for model_entry in results.get("models", []):
        model_name = model_entry.get("name", "unknown")
        cm = model_entry.get("metrics", {}).get("confusion_matrix")
        if cm is None:
            continue
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(results_dir / f"{model_name}_confusion_matrix.csv", index=False)


def _print_analysis(results_path: Path, summary_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    print("Results Summary")
    print(summary.to_string(index=False))

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("\nBest by macro_f1")
    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    print(best.to_string())

    print("\nPer-class F1 (by model)")
    for model_entry in data.get("models", []):
        name = model_entry.get("name", "unknown")
        f1 = model_entry.get("metrics", {}).get("per_class", {}).get("f1", [])
        print(f"{name}: {f1}")

    print("\nConfusion matrices")
    for model_entry in data.get("models", []):
        name = model_entry.get("name", "unknown")
        cm = model_entry.get("metrics", {}).get("confusion_matrix", [])
        print(f"{name}:")
        for row in cm:
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo FinBERT classifier stage using cached embeddings."
    )
    parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=str, default="data/finbert_embeddings")
    parser.add_argument("--results-dir", type=str, default="analysis/demo_results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "svm", "xgb"],
        help="Models to train: lr svm xgb",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    cache_dir = cache_dir / args.subset

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    results_dir = results_dir / args.subset
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train_path = cache_dir / "X_train_embeddings.npy"
    X_test_path = cache_dir / "X_test_embeddings.npy"
    if not X_train_path.exists() or not X_test_path.exists():
        raise FileNotFoundError(
            f"Missing cached embeddings in {cache_dir}. "
            "Run pipelines/finbert_pipeline/finbert_pipeline.py once to create them."
        )

    X_train_emb = np.load(X_train_path)
    X_test_emb = np.load(X_test_path)

    label_names = load_label_names(output_dir)
    _, _, y_train, y_test, dataset_label_names = get_splits(
        args.subset,
        args.test_size,
        args.seed,
        output_dir=output_dir,
    )
    if label_names is None:
        label_names = dataset_label_names
    if label_names is None:
        label_names = [str(i) for i in sorted(np.unique(y_train))]

    if len(y_train) != X_train_emb.shape[0] or len(y_test) != X_test_emb.shape[0]:
        # Fallback when output_dir contains cached splits from a different subset.
        _, _, y_train, y_test, dataset_label_names = get_splits(
            args.subset,
            args.test_size,
            args.seed,
            output_dir=None,
        )
        if label_names is None and dataset_label_names is not None:
            label_names = dataset_label_names

    if len(y_train) != X_train_emb.shape[0] or len(y_test) != X_test_emb.shape[0]:
        raise ValueError(
            "Cached embeddings shape still does not match split labels after fallback. "
            "Regenerate cached train/test embeddings for this subset."
        )

    selected = {m.lower() for m in args.models}
    results = {
        "dataset": args.subset,
        "train_size": int(X_train_emb.shape[0]),
        "test_size": int(X_test_emb.shape[0]),
        "models": [],
    }

    if "lr" in selected:
        lr_best, lr_params, lr_score = train_logistic_regression(X_train_emb, y_train, args.seed)
        results["models"].append(
            {
                "name": "logistic_regression",
                "best_params": lr_params,
                "cv_macro_f1": float(lr_score),
                "metrics": evaluate_model(
                    "logistic_regression",
                    lr_best,
                    X_test_emb,
                    y_test,
                    label_names,
                ),
            }
        )

    if "svm" in selected:
        svm_best, svm_params, svm_score = train_linear_svm(X_train_emb, y_train, args.seed)
        results["models"].append(
            {
                "name": "linear_svm",
                "best_params": svm_params,
                "cv_macro_f1": float(svm_score),
                "metrics": evaluate_model(
                    "linear_svm",
                    svm_best,
                    X_test_emb,
                    y_test,
                    label_names,
                ),
            }
        )

    if "xgb" in selected:
        if not HAS_XGBOOST:
            print("Skipping XGBoost (xgboost not installed).")
        else:
            xgb_best, xgb_params, xgb_score = train_xgboost(X_train_emb, y_train, args.seed)
            results["models"].append(
                {
                    "name": "xgboost",
                    "best_params": xgb_params,
                    "cv_macro_f1": float(xgb_score),
                    "metrics": evaluate_model(
                        "xgboost",
                        xgb_best,
                        X_test_emb,
                        y_test,
                        label_names,
                    ),
                }
            )

    results_path = results_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    for model_entry in results["models"]:
        m = model_entry["metrics"]
        rows.append(
            {
                "model": model_entry["name"],
                "accuracy": m["accuracy"],
                "precision": m.get("precision", m.get("macro_precision")),
                "recall": m.get("recall", m.get("macro_recall")),
                "macro_f1": m["macro_f1"],
                "roc_auc_ovr": m["roc_auc_ovr"],
            }
        )
    summary_path = results_dir / "results_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    _save_confusion_matrices(results, results_dir)
    _print_analysis(results_path, summary_path)
    print(f"\nSaved results to {results_dir}")


if __name__ == "__main__":
    main()
