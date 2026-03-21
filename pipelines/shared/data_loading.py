import json
from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from preprocessing.load_data import load_phrasebank


def load_label_names(output_dir: Path):
    label_path = output_dir / "label_names.json"
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_splits(subset: str, test_size: float, seed: int):
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
