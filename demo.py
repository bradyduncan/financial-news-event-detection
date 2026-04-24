"""demo.py — Standalone demo of the TF-IDF baseline pipeline.
Loads pre-saved models and results for fast execution (no retraining).
Displays dataset info, model performance, feature importance, and live predictions
on sample financial sentences to demonstrate the full pipeline end-to-end.

Usage:
    python demo.py
    python demo.py --subset sentences_75agree
    python demo.py --subset sentences_allagree
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Path setup:

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.preprocess_text import preprocess_text, TOKEN_RE
from preprocessing.feature_engineering import combine_features, FEATURE_NAMES
from preprocessing.load_data import load_phrasebank
from preprocessing.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_SUBSET,
    DEFAULT_TEST_SIZE,
)

# Command line arguments:

parser = argparse.ArgumentParser(description="Demo of TF-IDF Financial Sentiment Analysis.")
parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET)
parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
args = parser.parse_args()

# Configuration and directory setup

SUBSET = args.subset
OUTPUT_DIR = Path(args.output_dir)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = REPO_ROOT / OUTPUT_DIR

RESULTS_DIR = REPO_ROOT / "pipelines" / "tf-idf_pipeline" / "results" / SUBSET
LABEL_NAMES = ["negative", "neutral", "positive"]

"""Sample sentences for live prediction demo:
A mix of clearly positive, negative, and neutral financial statements
to show the model classifying unseen text in real time."""

SAMPLE_SENTENCES = [
    "The company reported a net loss of EUR 5 million compared to a profit in the previous year.",
    "Revenue for the period increased by 12% to EUR 45 million.",
    "The board of directors will meet on March 15 to discuss the proposal.",
    "Operating profit surged 40% driven by strong demand across all segments.",
    "Sales declined sharply due to weakening consumer confidence.",
    "The company announced the appointment of a new chief financial officer.",
    "Earnings per share rose to EUR 1.25 from EUR 0.80 a year earlier.",
    "Net sales fell 8% to EUR 120 million from EUR 130 million.",
]


def section_header(title):
    # Print a formatted section header to separate demo steps.
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def get_feature_name(all_feature_names, idx):
    # Safely retrieve a feature name by index, returning a fallback if out of range.
    if idx < len(all_feature_names):
        return all_feature_names[idx]
    return f"feature_{idx}"


def main():
    print("=" * 65)
    print("  FINANCIAL NEWS SENTIMENT ANALYSIS — TF-IDF BASELINE DEMO")
    print("  CS 6120 NLP — Kirtan Patel | Brady Duncan")
    print("=" * 65)

    # Load dataset and display label distribution

    section_header("STEP 1: Dataset Overview")
    print(f"Loading PhraseBank subset: {SUBSET}")
    texts, labels, label_names = load_phrasebank(SUBSET)
    if label_names is None:
        label_names = LABEL_NAMES

    counts = Counter(labels)
    total = len(labels)
    print(f"Total sentences: {total}")
    print(f"Labels: {label_names}\n")

    print("Label distribution:")
    for idx in sorted(counts):
        name = label_names[idx]
        count = counts[idx]
        pct = count / total * 100
        print(f"  {name:>10s}: {count:5d}  ({pct:.1f}%)")

    # EDA highlights — sentence length and number presence per class

    section_header("STEP 2: EDA Highlights")

    import re
    from collections import defaultdict

    lengths = defaultdict(list)
    num_re = re.compile(r"\d+(?:\.\d+)?%?")
    has_num = defaultdict(int)
    class_total = Counter()

    for text, label in zip(texts, labels):
        lengths[label].append(len(TOKEN_RE.findall(text)))
        class_total[label] += 1
        if num_re.search(text):
            has_num[label] += 1

    print("Average sentence length (tokens) per class:")
    for idx in sorted(lengths):
        avg = sum(lengths[idx]) / len(lengths[idx])
        print(f"  {label_names[idx]:>10s}: {avg:.1f}")

    print("\nSentences containing numbers (%):")
    for idx in sorted(has_num):
        pct = has_num[idx] / class_total[idx] * 100
        print(f"  {label_names[idx]:>10s}: {pct:.1f}%")

    #  Load pre-saved results from results.json

    section_header("STEP 3: Model Performance (from saved results)")

    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        print(f"ERROR: Results not found at {results_path}")
        print("Please run train_evaluate.py first:")
        print(f"  python pipelines/tf-idf_pipeline/train_evaluate.py --subset {SUBSET}")
        sys.exit(1)

    with results_path.open("r", encoding="utf-8") as f:
        all_results = json.load(f)

    """Display summary table of all models"""

    print(f"\n{'Model':<25s} {'CV F1':>8s} {'Test F1':>9s} {'Accuracy':>10s} {'ROC-AUC':>9s}")
    print("-" * 65)

    best_name = None
    best_f1 = -1
    for entry in all_results["models"]:
        m = entry["metrics"]
        roc = m.get("roc_auc_ovr")
        roc_str = f"{roc:.4f}" if roc is not None else "N/A"
        print(
            f"  {entry['name']:<23s} "
            f"{entry['cv_macro_f1']:>8.4f} "
            f"{m['macro_f1']:>9.4f} "
            f"{m['accuracy']:>10.4f} "
            f"{roc_str:>9s}"
        )
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_name = entry["name"]

    print(f"\nBest model: {best_name} (Test Macro-F1: {best_f1:.4f})")

    # Display per-class metrics for each model

    for entry in all_results["models"]:
        name = entry["name"]
        m = entry["metrics"]
        print(f"\n  [{name}]")
        for i, lbl in enumerate(label_names):
            p = m["per_class"]["precision"][i]
            r = m["per_class"]["recall"][i]
            f = m["per_class"]["f1"][i]
            print(f"    {lbl:>10s}:  precision={p:.2f}  recall={r:.2f}  f1={f:.2f}")

    # Display confusion matrix for best model   

    for entry in all_results["models"]:
        if entry["name"] == best_name:
            print(f"\nConfusion Matrix ({best_name}):")
            for row in entry["metrics"]["confusion_matrix"]:
                print(f"  {row}")
            break

    # Feature importance from pre-saved Logistic Regression model

    section_header("STEP 4: Interpretability — Top LR Features")

    lr_path = RESULTS_DIR / "logistic_regression_best.pkl"
    tfidf_path = OUTPUT_DIR / "tfidf_pipeline.pkl"

    if not lr_path.exists() or not tfidf_path.exists():
        print("Saved LR model or TF-IDF pipeline not found. Skipping feature importance.")
    else:
        with lr_path.open("rb") as f:
            lr_model = pickle.load(f)
        with tfidf_path.open("rb") as f:
            tfidf_pipeline = pickle.load(f)

        # Extract feature names from TF-IDF vocabulary and append handcrafted feature names

        tfidf_step = tfidf_pipeline.named_steps["tfidf"]
        tfidf_names = tfidf_step.get_feature_names_out().tolist()
        all_feature_names = tfidf_names + FEATURE_NAMES

        # Limit top_n to indices that exist within the coefficient matrix

        n_features = lr_model.coef_.shape[1]
        top_n = 10

        for i, label in enumerate(label_names):
            coefs = lr_model.coef_[i]

            # Only consider indices within the valid range of the coefficient array
            valid_indices = np.arange(min(n_features, len(coefs)))
            sorted_indices = np.argsort(coefs[valid_indices])

            top_pos = sorted_indices[-top_n:][::-1]
            top_neg = sorted_indices[:top_n]

            print(f"\n[{label}] Top {top_n} positive coefficients:")
            for idx in top_pos:
                fname = get_feature_name(all_feature_names, idx)
                print(f"  {fname:30s}  {coefs[idx]:+.4f}")

            print(f"\n[{label}] Top {top_n} negative coefficients:")
            for idx in top_neg:
                fname = get_feature_name(all_feature_names, idx)
                print(f"  {fname:30s}  {coefs[idx]:+.4f}")

    # Live prediction on sample financial sentences

    section_header("STEP 5: Live Predictions on Sample Sentences")

    if not lr_path.exists() or not tfidf_path.exists():
        print("Saved model artifacts not found. Skipping live predictions.")
    else:
        """Transform raw sentences through the saved TF-IDF pipeline.
        The pipeline handles preprocessing internally (lowercasing, tokenization,
        negation handling, stopword removal) so we pass raw text directly."""

        tfidf_matrix = tfidf_pipeline.transform(SAMPLE_SENTENCES)

        # Combine TF-IDF features with handcrafted features

        X_demo = combine_features(tfidf_matrix, SAMPLE_SENTENCES)

        #Verify feature count matches what the model expects

        expected_features = lr_model.coef_.shape[1]
        actual_features = X_demo.shape[1]

        if actual_features != expected_features:
            print(f"Feature mismatch: model expects {expected_features}, got {actual_features}.")
            print("Skipping live predictions (pipeline may have been retrained).")
        else:
            # Generate predictions and display results

            predictions = lr_model.predict(X_demo)

            # Display predicted probabilities if available (LR supports predict_proba)

            try:
                probas = lr_model.predict_proba(X_demo)
                has_proba = True
            except AttributeError:
                has_proba = False

            for i, sentence in enumerate(SAMPLE_SENTENCES):
                pred_label = label_names[predictions[i]]
                print(f"\n  Sentence: \"{sentence}\"")
                print(f"  Predicted: {pred_label}")
                if has_proba:
                    proba_str = "  Confidence: " + "  ".join(
                        f"{label_names[j]}={probas[i][j]:.2f}"
                        for j in range(len(label_names))
                    )
                    print(proba_str)

    # Summary

    section_header("DEMO COMPLETE")
    print(f"Dataset: {SUBSET} ({total} sentences)")
    print(f"Best model: {best_name} (Macro-F1: {best_f1:.4f})")
    print(f"Handcrafted features: {len(FEATURE_NAMES)} ({', '.join(FEATURE_NAMES)})")
    print(f"\nResults loaded from: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()