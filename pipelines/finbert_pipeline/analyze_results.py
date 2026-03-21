import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from preprocessing.config import DEFAULT_SUBSET


def main():
    parser = argparse.ArgumentParser(description="Analyze FinBERT pipeline results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="pipelines/finbert_pipeline/results",
        help="Directory containing results_summary_<subset>.csv and results_<subset>.json",
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

    summary_path = results_dir / f"results_summary_{args.subset}.csv"
    json_path = results_dir / f"results_{args.subset}.json"

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


if __name__ == "__main__":
    main()
