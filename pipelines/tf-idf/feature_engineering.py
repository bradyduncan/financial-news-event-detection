"""Handcrafted feature extraction for Financial PhraseBank sentences."""

import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?%?|[$%]|[+-]")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?%")

FINANCIAL_KEYWORDS = {
    "profit", "loss", "growth", "decline", "revenue", "earnings",
    "exceeded", "rose", "fell", "increased", "decreased", "gain",
    "drop", "surged", "plunged", "dividend", "forecast", "margin",
    "debt", "sales", "cost", "income", "expense", "deficit", "surplus",
}

POSITIVE_KEYWORDS = {
    "profit", "growth", "rose", "increased", "gain", "surged",
    "exceeded", "surplus", "dividend",
}

NEGATIVE_KEYWORDS = {
    "loss", "decline", "fell", "decreased", "drop", "plunged",
    "debt", "deficit", "expense", "cost",
}

NEGATION_WORDS = {"no", "not", "nor", "never", "n't", "neither", "nobody", "nothing"}


def extract_features(texts):
    """
    Extract handcrafted features from raw sentences.

    Returns a sparse matrix with columns:
        0 - sentence length (token count)
        1 - financial keyword count
        2 - positive keyword count
        3 - negative keyword count
        4 - has number (binary)
        5 - has percentage (binary)
        6 - number count
        7 - negation word count
    """
    n = len(texts)
    features = np.zeros((n, 8), dtype=np.float64)

    for i, text in enumerate(texts):
        lower = text.lower()
        tokens = TOKEN_RE.findall(lower)
        token_set = set(tokens)

        # sentence length
        features[i, 0] = len(tokens)

        # financial keyword count
        features[i, 1] = len(token_set & FINANCIAL_KEYWORDS)

        # positive keyword count
        features[i, 2] = len(token_set & POSITIVE_KEYWORDS)

        # negative keyword count
        features[i, 3] = len(token_set & NEGATIVE_KEYWORDS)

        # has number (binary)
        features[i, 4] = 1.0 if NUMBER_RE.search(text) else 0.0

        # has percentage (binary)
        features[i, 5] = 1.0 if PERCENT_RE.search(text) else 0.0

        # number count
        features[i, 6] = len(NUMBER_RE.findall(text))

        # negation word count
        features[i, 7] = len(token_set & NEGATION_WORDS)

    return csr_matrix(features)


FEATURE_NAMES = [
    "sentence_length",
    "financial_kw_count",
    "positive_kw_count",
    "negative_kw_count",
    "has_number",
    "has_percentage",
    "number_count",
    "negation_count",
]


def combine_features(tfidf_matrix, texts):
    """Combine TF-IDF matrix with handcrafted features."""
    handcrafted = extract_features(texts)
    return hstack([tfidf_matrix, handcrafted])