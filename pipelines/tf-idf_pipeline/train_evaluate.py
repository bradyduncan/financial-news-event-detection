"""Train and evaluate TF-IDF + Classical ML baseline models."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

"""Path setup:
Dynamically add the repo root to sys.path to allow imports from any subdirectory.
This ensures custom modules can be imported regardless of the current working directory."""

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from preprocessing.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_SUBSET,
    DEFAULT_TEST_SIZE,
)
from preprocessing.feature_engineering import combine_features, FEATURE_NAMES
from pipelines.shared.data_loading import get_splits, load_label_names
from pipelines.shared.evaluation import evaluate_model

"""Command line arguments:
Allows users to specify the subset, test size, random seed, and output directory."""

parser = argparse.ArgumentParser(description="Train TF-IDF + Classical ML baseline models.")
parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET)
parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
args = parser.parse_args()

"""Configuration and Directory setup"""

SUBSET = args.subset
TEST_SIZE = args.test_size
SEED = args.seed

OUTPUT_DIR = Path(args.output_dir)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = REPO_ROOT / OUTPUT_DIR

RESULTS_DIR = REPO_ROOT / "pipelines" / "tf-idf_pipeline" / "results" / SUBSET
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

"""Load label names and train/test splits (with caching)"""
print("Loading label names...")
label_names = load_label_names(OUTPUT_DIR)

print("Loading train/test splits...")
X_train_texts, X_test_texts, y_train, y_test, dataset_label_names = get_splits(
    SUBSET, TEST_SIZE, SEED, output_dir=OUTPUT_DIR
)
if label_names is None:
    label_names = dataset_label_names
if label_names is None:
    label_names = [str(i) for i in sorted(np.unique(y_train))]

print(f"Train size: {len(X_train_texts)}  Test size: {len(X_test_texts)}")
print(f"Labels: {label_names}")
print()

"""Load precomputed TF-IDF matrices and pipeline (for feature names)"""
print("Loading TF-IDF matrices...")
with (OUTPUT_DIR / "train_tfidf.pkl").open("rb") as f:
    X_train_tfidf, _ = pickle.load(f)
with (OUTPUT_DIR / "test_tfidf.pkl").open("rb") as f:
    X_test_tfidf, _ = pickle.load(f)
with (OUTPUT_DIR / "tfidf_pipeline.pkl").open("rb") as f:
    tfidf_pipeline = pickle.load(f)

"""Combine TF-IDF features with handcrafted features"""
print("Combining TF-IDF with handcrafted features...")
X_train = combine_features(X_train_tfidf, X_train_texts)
X_test = combine_features(X_test_tfidf, X_test_texts)

print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape}")
print(f"TF-IDF features: {X_train_tfidf.shape[1]}  Handcrafted features: {len(FEATURE_NAMES)}")
print()

"""Model definitions and hyperparameters for grid search"""
models = {
    "logistic_regression": {
        "model": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
    },
    "linear_svm": {
        "model": LinearSVC(
            class_weight="balanced", max_iter=5000, random_state=SEED
        ),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(
            class_weight="balanced", random_state=SEED
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
        },
    },
}

""""Cross-validation setup for hyperparameter tuning"""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

"""Train and evaluate each model
Using a dictionary to store results for each model, including best hyperparameters, CV scores, and test metrics."""
all_results = {"dataset": SUBSET, "models": []}

for name, config in models.items():
    print("=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV macro-F1: {grid.best_score_:.2f}")
    print()

    """Evaluate best model on test set"""
    metrics = evaluate_model(name, grid.best_estimator_, X_test, y_test, label_names)

    print(f"Test accuracy: {metrics['accuracy']:.2f}")
    print(f"Test macro-F1: {metrics['macro_f1']:.2f}")
    if metrics["roc_auc_ovr"] is not None:
        print(f"Test ROC-AUC (OVR): {metrics['roc_auc_ovr']:.2f}")
    print()

    """Per-class metrics"""
    for i, lbl in enumerate(label_names):
        p = metrics["per_class"]["precision"][i]
        r = metrics["per_class"]["recall"][i]
        f = metrics["per_class"]["f1"][i]
        print(f"  {lbl:>10s}:  precision={p:.2f}  recall={r:.2f}  f1={f:.2f}")
    print()

    print("Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")
    print()

    all_results["models"].append({
        "name": name,
        "best_params": grid.best_params_,
        "cv_macro_f1": round(grid.best_score_, 4),
        "metrics": metrics,
    })

    """Save trained model for later analysis (e.g., feature importance)"""
    with (RESULTS_DIR / f"{name}_best.pkl").open("wb") as f:
        pickle.dump(grid.best_estimator_, f)

"""Feature importance analysis for Logistic Regression (if available)"""
print("=" * 60)
print("TOP FEATURES (Logistic Regression coefficients)")
print("=" * 60)

with (RESULTS_DIR / "logistic_regression_best.pkl").open("rb") as f:
    lr = pickle.load(f)

tfidf_step = tfidf_pipeline.named_steps["tfidf"]
tfidf_names = tfidf_step.get_feature_names_out().tolist()
all_feature_names = tfidf_names + FEATURE_NAMES

for i, label in enumerate(label_names):
    coefs = lr.coef_[i]
    top_idx = np.argsort(coefs)[-10:][::-1]
    bottom_idx = np.argsort(coefs)[:10]

    print(f"\n[{label}] Top 10 positive coefficients:")
    for idx in top_idx:
        print(f"  {all_feature_names[idx]:30s}  {coefs[idx]:+.4f}")

    print(f"\n[{label}] Top 10 negative coefficients:")
    for idx in bottom_idx:
        print(f"  {all_feature_names[idx]:30s}  {coefs[idx]:+.4f}")

"""Save results to JSON and CSV for later analysis"""
results_path = RESULTS_DIR / "results.json"
with results_path.open("w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

"""Save a simplified CSV summary for easy comparison across models"""
flat_rows = []
for model_entry in all_results["models"]:
    metrics = model_entry["metrics"]
    flat_rows.append({
        "model": model_entry["name"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "roc_auc_ovr": metrics["roc_auc_ovr"],
    })
pd.DataFrame(flat_rows).to_csv(RESULTS_DIR / "results_summary.csv", index=False)

"""Summary of results"""
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for entry in all_results["models"]:
    m = entry["metrics"]
    print(
        f"  {entry['name']:25s}  "
        f"CV F1={entry['cv_macro_f1']:.2f}  "
        f"Test F1={m['macro_f1']:.2f}  "
        f"Accuracy={m['accuracy']:.2f}"
    )

print(f"\nResults saved to: {RESULTS_DIR}")
print("Done!")