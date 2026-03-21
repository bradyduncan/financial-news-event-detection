import argparse
import json
import pickle
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from sklearn.model_selection import train_test_split

from config import (
    DEFAULT_LEMMATIZE,
    DEFAULT_MAX_FEATURES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_NGRAM_RANGE,
    DEFAULT_SEED,
    DEFAULT_SUBSET,
    DEFAULT_TEST_SIZE,
    DEFAULT_USE_NEGATION,
)
from preprocessing.load_data import load_phrasebank
from preprocessing.preprocess_text import build_tfidf_pipeline


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Financial PhraseBank.")
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        help="PhraseBank subset (e.g., sentences_allagree, sentences_75agree).",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lemmatize", action="store_true", default=DEFAULT_LEMMATIZE)
    parser.add_argument(
        "--negation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_NEGATION,
        help="Enable/disable negation handling (default: enabled).",
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument(
        "--ngram-min", type=int, default=DEFAULT_NGRAM_RANGE[0], help="Minimum n-gram."
    )
    parser.add_argument(
        "--ngram-max", type=int, default=DEFAULT_NGRAM_RANGE[1], help="Maximum n-gram."
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    texts, labels, label_names = load_phrasebank(args.subset)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    pipeline = build_tfidf_pipeline(
        lemmatize=args.lemmatize,
        use_negation=args.negation,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
    X_train_tfidf = pipeline.fit_transform(X_train)
    X_test_tfidf = pipeline.transform(X_test)

    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")
    if label_names:
        print(f"Labels: {label_names}")

    if args.output_dir:
        repo_root = Path(__file__).resolve().parents[1]
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = repo_root / out
        out.mkdir(parents=True, exist_ok=True)
        with (out / "tfidf_pipeline.pkl").open("wb") as f:
            pickle.dump(pipeline, f)
        with (out / "train_tfidf.pkl").open("wb") as f:
            pickle.dump((X_train_tfidf, y_train), f)
        with (out / "test_tfidf.pkl").open("wb") as f:
            pickle.dump((X_test_tfidf, y_test), f)
        if label_names:
            with (out / "label_names.json").open("w", encoding="utf-8") as f:
                json.dump(label_names, f, indent=2)


if __name__ == "__main__":
    main()
