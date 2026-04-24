import json
from pathlib import Path
import sys
import pickle

import numpy as np
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from preprocessing.load_data import load_phrasebank


def load_label_names(output_dir: Path):
    # Use saved label order when available
    label_path = output_dir / "label_names.json"
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_saved_splits(output_dir: Path):
    # Load cached train test splits if present
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


def get_splits(subset: str, test_size: float, seed: int, output_dir: Path | None = None):
    # Prefer cached splits for reproducibility
    if output_dir is not None:
        saved = load_saved_splits(output_dir)
        if saved is not None:
            X_train, X_test, y_train, y_test = saved
            return X_train, X_test, np.array(y_train), np.array(y_test), None

    texts, labels, label_names = load_phrasebank(subset)
    if isinstance(labels[0], str):
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels).tolist()
        if label_names is None:
            label_names = list(encoder.classes_)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return X_train, X_test, np.array(y_train), np.array(y_test), label_names
