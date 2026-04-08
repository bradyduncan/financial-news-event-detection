import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:
    raise ImportError(
        "transformers is required for FinBERT embeddings. "
        "Install it with: pip install transformers"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from preprocessing.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_SUBSET,
    DEFAULT_TEST_SIZE,
)
from pipelines.finbert_pipeline.classifiers import (
    HAS_XGBOOST,
    train_linear_svm,
    train_logistic_regression,
    train_xgboost,
)
from pipelines.shared.data_loading import get_splits, load_label_names
from pipelines.finbert_pipeline.embeddings import MODEL_NAME, load_or_create_embeddings
from pipelines.shared.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="FinBERT embedding + classifier pipeline.")
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=str, default="data/finbert_embeddings")
    parser.add_argument("--results-dir", type=str, default="pipelines/finbert_pipeline/results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "svm", "xgb"],
        help="Models to train: lr svm xgb",
    )
    args = parser.parse_args()

    # fix path issue
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

    # Load labels for stability across runs
    label_names = load_label_names(output_dir)
    print("Loading train/test splits")
    X_train_texts, X_test_texts, y_train, y_test, dataset_label_names = get_splits(
        args.subset, args.test_size, args.seed, output_dir=output_dir
    )
    print(f"Train size: {len(X_train_texts)}  Test size: {len(X_test_texts)}")
    if label_names is None:
        label_names = dataset_label_names
    if label_names is None:
        label_names = [str(i) for i in sorted(np.unique(y_train))]

    print("Loading FinBERT tokenizer/model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("Computing/loading embeddings")
    X_train_emb = load_or_create_embeddings(
        X_train_texts,
        cache_dir / "X_train_embeddings.npy",
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        expected_len=len(X_train_texts),
    )
    X_test_emb = load_or_create_embeddings(
        X_test_texts,
        cache_dir / "X_test_embeddings.npy",
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        expected_len=len(X_test_texts),
    )
    print("Embeddings ready")

    results = {"dataset": args.subset, "models": []}

    # Normalize model names so CLI input is forgiving
    selected = set([m.lower() for m in args.models])
    if "lr" in selected:
        print("Training Logistic Regression")
        lr_best, lr_params, lr_score = train_logistic_regression(
            X_train_emb, y_train, args.seed
        )
        results["models"].append(
            {
                "name": "logistic_regression",
                "best_params": lr_params,
                "cv_macro_f1": lr_score,
                "metrics": evaluate_model(
                    "logistic_regression", lr_best, X_test_emb, y_test, label_names
                ),
            }
        )

    if "svm" in selected:
        print("Training Linear SVM")
        svm_best, svm_params, svm_score = train_linear_svm(
            X_train_emb, y_train, args.seed
        )
        results["models"].append(
            {
                "name": "linear_svm",
                "best_params": svm_params,
                "cv_macro_f1": svm_score,
                "metrics": evaluate_model(
                    "linear_svm", svm_best, X_test_emb, y_test, label_names
                ),
            }
        )

    if "xgb" in selected:
        if not HAS_XGBOOST:
            print("Skipping XGBoost (xgboost not installed).")
        else:
            print("Training XGBoost")
            xgb_best, xgb_params, xgb_score = train_xgboost(
                X_train_emb, y_train, args.seed
            )
            results["models"].append(
                {
                    "name": "xgboost",
                    "best_params": xgb_params,
                    "cv_macro_f1": xgb_score,
                    "metrics": evaluate_model(
                        "xgboost", xgb_best, X_test_emb, y_test, label_names
                    ),
                }
            )

    print("Saving results")
    
    # Add all results to a results folder
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    flat_rows = []
    for model_entry in results["models"]:
        metrics = model_entry["metrics"]
        flat_rows.append(
            {
                "model": model_entry["name"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc_ovr": metrics["roc_auc_ovr"],
            }
        )
    pd.DataFrame(flat_rows).to_csv(results_dir / "results_summary.csv", index=False)

    print(f"Saved embeddings to {cache_dir}")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
