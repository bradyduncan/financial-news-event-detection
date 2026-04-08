from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from preprocessing.sentiment_score import sentiment_score_from_proba


def add_continuous_sentiment_score(
    articles: pd.DataFrame,
    proba: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    label_names: Sequence[str],
    column_name: str = "sentiment_score",
    label_score_map: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """
    Add a continuous sentiment score to an articles dataframe.

    proba should align with the rows of articles and the order of label_names.
    """
    # Keep this pure so it is easy to test and reuse.
    scores = sentiment_score_from_proba(
        proba, label_names=label_names, label_score_map=label_score_map
    )

    out = articles.copy()
    out[column_name] = scores
    return out
