from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _infer_label_score_map(label_names: Sequence[str]) -> dict[str, float]:
    normalized = [str(name).strip().lower().replace("_", " ") for name in label_names]

    if len(normalized) == 3 and set(normalized) >= {"negative", "neutral", "positive"}:
        return {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

    score_map: dict[str, float] = {}
    for original, norm in zip(label_names, normalized):
        score = _score_for_label(norm)
        if score is None:
            raise ValueError(
                "Could not infer sentiment score mapping for label: "
                f"{original!r}. Provide label_score_map explicitly."
            )
        score_map[str(original)] = float(score)
    return score_map


def _score_for_label(label: str) -> float | None:
    if "neutral" in label:
        return 0.0

    is_negative = "neg" in label or "bear" in label or "down" in label
    is_positive = "pos" in label or "bull" in label or "up" in label

    if is_negative:
        if "strong" in label or "very" in label:
            return -2.0
        if "weak" in label or "slight" in label:
            return -1.0
        return -1.0

    if is_positive:
        if "strong" in label or "very" in label:
            return 2.0
        if "weak" in label or "slight" in label:
            return 1.0
        return 1.0

    return None


def sentiment_score_from_proba(
    proba: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    label_names: Sequence[str],
    label_score_map: Mapping[str, float] | None = None,
) -> np.ndarray | float:
    """
    Convert class probabilities to a continuous sentiment score via expected value.

    For a 3-class setup (negative, neutral, positive), the default mapping is:
    negative -> -1, neutral -> 0, positive -> +1.
    """
    proba_arr = np.asarray(proba, dtype=float)
    single = False
    if proba_arr.ndim == 1:
        proba_arr = proba_arr.reshape(1, -1)
        single = True

    if proba_arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D proba array, got shape {proba_arr.shape}.")

    if len(label_names) != proba_arr.shape[1]:
        raise ValueError(
            "label_names length does not match proba columns: "
            f"{len(label_names)} vs {proba_arr.shape[1]}"
        )

    if label_score_map is None:
        label_score_map = _infer_label_score_map(label_names)

    scores = np.array([label_score_map[str(name)] for name in label_names], dtype=float)
    result = proba_arr @ scores
    return float(result[0]) if single else result
