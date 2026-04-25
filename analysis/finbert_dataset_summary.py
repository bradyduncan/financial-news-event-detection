from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from preprocessing.config import DEFAULT_OUTPUT_DIR, DEFAULT_SEED, DEFAULT_SUBSET, DEFAULT_TEST_SIZE
from preprocessing.load_data import load_phrasebank


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _word_len(text: str) -> int:
    # Simple whitespace tokenization (good enough for presentation stats).
    return len(text.split())


def _length_stats(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {
            "n": 0,
            "mean_words": float("nan"),
            "std_words": float("nan"),
            "min_words": float("nan"),
            "p50_words": float("nan"),
            "p90_words": float("nan"),
            "max_words": float("nan"),
            "mean_chars": float("nan"),
            "std_chars": float("nan"),
        }

    word_lens = np.array([_word_len(t) for t in texts], dtype=float)
    char_lens = np.array([len(t) for t in texts], dtype=float)

    return {
        "n": int(len(texts)),
        "mean_words": _safe_float(word_lens.mean()),
        "std_words": _safe_float(word_lens.std(ddof=0)),
        "min_words": _safe_float(word_lens.min()),
        "p50_words": _safe_float(np.percentile(word_lens, 50)),
        "p90_words": _safe_float(np.percentile(word_lens, 90)),
        "max_words": _safe_float(word_lens.max()),
        "mean_chars": _safe_float(char_lens.mean()),
        "std_chars": _safe_float(char_lens.std(ddof=0)),
    }


def _class_table(
    y: np.ndarray, label_names: list[str] | None
) -> list[dict[str, object]]:
    c = Counter([int(v) for v in y.tolist()])
    total = int(len(y))
    rows: list[dict[str, object]] = []
    for label_id in sorted(c.keys()):
        name = (
            label_names[label_id]
            if label_names is not None and 0 <= label_id < len(label_names)
            else str(label_id)
        )
        count = int(c[label_id])
        pct = (count / total * 100.0) if total else float("nan")
        rows.append({"label_id": label_id, "label": name, "count": count, "pct": pct})
    return rows


def _print_length_stats(prefix: str, stats: dict[str, float]) -> None:
    print(f"{prefix}_n={stats['n']}")
    print(
        f"{prefix}_len_words_mean={stats['mean_words']:.2f} "
        f"{prefix}_len_words_std={stats['std_words']:.2f} "
        f"{prefix}_len_words_p50={stats['p50_words']:.0f} "
        f"{prefix}_len_words_p90={stats['p90_words']:.0f}"
    )
    print(
        f"{prefix}_len_chars_mean={stats['mean_chars']:.2f} "
        f"{prefix}_len_chars_std={stats['std_chars']:.2f}"
    )


def _print_class_rows(prefix: str, rows: list[dict[str, object]]) -> None:
    for r in rows:
        print(
            f"{prefix}_class={r['label']} "
            f"count={r['count']} pct={float(r['pct']):.2f}"
        )


def _load_cached_splits(output_dir: Path):
    train_texts_path = output_dir / "train_texts.pkl"
    test_texts_path = output_dir / "test_texts.pkl"
    train_labels_path = output_dir / "train_labels.npy"
    test_labels_path = output_dir / "test_labels.npy"

    if not (
        train_texts_path.exists()
        and test_texts_path.exists()
        and train_labels_path.exists()
        and test_labels_path.exists()
    ):
        return None

    with train_texts_path.open("rb") as f:
        X_train = pickle.load(f)
    with test_texts_path.open("rb") as f:
        X_test = pickle.load(f)
    y_train = np.load(train_labels_path)
    y_test = np.load(test_labels_path)
    return X_train, X_test, y_train, y_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Financial PhraseBank train/test splits for the FinBERT pipeline, "
            "including text length stats and class percentages."
        )
    )
    parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prefer-cached",
        action="store_true",
        help="Prefer cached train/test artifacts in output-dir when present.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    label_names: list[str] | None = None
    label_path = output_dir / "label_names.json"
    if label_path.exists():
        try:
            label_names = _load_json(label_path)
        except Exception:
            label_names = None

    cached = _load_cached_splits(output_dir) if args.prefer_cached else None
    if cached is not None:
        X_train, X_test, y_train, y_test = cached
        dataset_label_names = None
    else:
        texts, labels, dataset_label_names = load_phrasebank(args.subset)
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )
        y_train = np.array(y_train)
        y_test = np.array(y_test)

    if label_names is None:
        label_names = dataset_label_names

    total_n = int(len(X_train) + len(X_test))
    print("FinBERT Dataset Summary")
    print(f"subset={args.subset}")
    print(f"split=train/test ({(1.0 - args.test_size) * 100:.1f}%/{args.test_size * 100:.1f}%)")
    print(f"total_samples={total_n}")
    print(f"train_samples={len(X_train)} test_samples={len(X_test)}")
    print("validation_samples=0 (CV folds are used during grid search)")
    if label_names is not None:
        print(f"label_names={label_names}")

    train_len = _length_stats(list(X_train))
    test_len = _length_stats(list(X_test))
    _print_length_stats("train", train_len)
    _print_length_stats("test", test_len)

    train_rows = _class_table(y_train, label_names)
    test_rows = _class_table(y_test, label_names)
    _print_class_rows("train", train_rows)
    _print_class_rows("test", test_rows)


if __name__ == "__main__":
    main()

