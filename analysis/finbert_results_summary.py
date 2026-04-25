from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


DEFAULT_SUBSETS = [
    "sentences_allagree",
    "sentences_75agree",
    "sentences_66agree",
    "sentences_50agree",
]

DEFAULT_CLASSIFIERS = [
    "logistic_regression",
    "linear_svm",
    "xgboost",
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(values: list[float] | None) -> float | None:
    if values is None:
        return None
    try:
        if len(values) == 0:
            return None
        return float(sum(values) / len(values))
    except Exception:
        return None


def _metric_precision(metrics: dict) -> float | None:
    if metrics.get("precision") is not None:
        return float(metrics["precision"])
    report = metrics.get("classification_report") or {}
    macro = report.get("macro avg") or {}
    if macro.get("precision") is not None:
        return float(macro["precision"])
    per_class = (metrics.get("per_class") or {}).get("precision")
    return _safe_mean(per_class)


def _metric_recall(metrics: dict) -> float | None:
    if metrics.get("recall") is not None:
        return float(metrics["recall"])
    report = metrics.get("classification_report") or {}
    macro = report.get("macro avg") or {}
    if macro.get("recall") is not None:
        return float(macro["recall"])
    per_class = (metrics.get("per_class") or {}).get("recall")
    return _safe_mean(per_class)


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def _detect_subsets(results_root: Path) -> list[str]:
    if not results_root.exists():
        return []
    found: set[str] = set()
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "results.json").exists():
            found.add(child.name)

    ordered: list[str] = [s for s in DEFAULT_SUBSETS if s in found]
    extras = sorted([s for s in found if s not in set(DEFAULT_SUBSETS)])
    return ordered + extras


def _extract_models_by_name(results_json: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model_entry in results_json.get("models", []) or []:
        name = model_entry.get("name")
        if not name:
            continue
        out[str(name)] = model_entry
    return out


def _summarize_classifier_across_subsets(
    classifier_name: str,
    subsets: list[str],
    results_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for subset in subsets:
        json_path = results_root / subset / "results.json"
        if not json_path.exists():
            rows.append(
                {
                    "Subset": subset,
                    "Macro F1": "",
                    "ROC-AUC": "",
                    "Precision": "",
                    "Recall": "",
                }
            )
            continue

        data = _load_json(json_path)
        models = _extract_models_by_name(data)
        model_entry = models.get(classifier_name)
        if model_entry is None:
            rows.append(
                {
                    "Subset": subset,
                    "Macro F1": "",
                    "ROC-AUC": "",
                    "Precision": "",
                    "Recall": "",
                }
            )
            continue

        metrics = model_entry.get("metrics", {}) or {}
        rows.append(
            {
                "Subset": subset,
                "Macro F1": _fmt(metrics.get("macro_f1")),
                "ROC-AUC": _fmt(metrics.get("roc_auc_ovr")),
                "Precision": _fmt(_metric_precision(metrics)),
                "Recall": _fmt(_metric_recall(metrics)),
            }
        )

    return pd.DataFrame(rows)


def _best_model_per_subset(
    subsets: list[str],
    results_root: Path,
    restrict_to: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for subset in subsets:
        json_path = results_root / subset / "results.json"
        if not json_path.exists():
            rows.append(
                {
                    "Subset": subset,
                    "Best Model": "",
                    "Macro F1": "",
                    "ROC-AUC": "",
                    "Precision": "",
                    "Recall": "",
                }
            )
            continue

        data = _load_json(json_path)
        best_name: str | None = None
        best_metrics: dict | None = None
        best_f1 = float("-inf")

        for model_entry in data.get("models", []) or []:
            name = str(model_entry.get("name") or "")
            if not name:
                continue
            if restrict_to is not None and name not in restrict_to:
                continue
            metrics = model_entry.get("metrics", {}) or {}
            f1 = metrics.get("macro_f1")
            if f1 is None:
                continue
            try:
                f1_val = float(f1)
            except Exception:
                continue
            if f1_val > best_f1:
                best_f1 = f1_val
                best_name = name
                best_metrics = metrics

        if best_name is None or best_metrics is None:
            rows.append(
                {
                    "Subset": subset,
                    "Best Model": "",
                    "Macro F1": "",
                    "ROC-AUC": "",
                    "Precision": "",
                    "Recall": "",
                }
            )
            continue

        rows.append(
            {
                "Subset": subset,
                "Best Model": best_name,
                "Macro F1": _fmt(best_metrics.get("macro_f1")),
                "ROC-AUC": _fmt(best_metrics.get("roc_auc_ovr")),
                "Precision": _fmt(_metric_precision(best_metrics)),
                "Recall": _fmt(_metric_recall(best_metrics)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print FinBERT results summary tables (macro-F1, ROC-AUC, precision, recall) "
            "for each classifier across subsets."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="pipelines/finbert_pipeline/results",
        help="Root dir containing subset folders with results.json.",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Subset folders to summarize (default: auto-detect, else fallback list).",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=DEFAULT_CLASSIFIERS,
        help="Classifier names as stored in results.json (e.g. logistic_regression linear_svm xgboost).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root

    if args.subsets is None or len(args.subsets) == 0:
        detected = _detect_subsets(results_root)
        subsets = detected if detected else DEFAULT_SUBSETS
    else:
        subsets = args.subsets

    for clf in args.classifiers:
        print()
        print(f"{clf} across all subsets")
        df = _summarize_classifier_across_subsets(clf, subsets, results_root)
        print(df.to_string(index=False))

    print()
    print("best_model across all subsets")
    best_df = _best_model_per_subset(
        subsets,
        results_root,
        restrict_to=set(args.classifiers) if args.classifiers else None,
    )
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
