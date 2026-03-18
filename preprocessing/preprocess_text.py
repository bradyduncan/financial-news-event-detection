import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?%?|[$%]|[+-]")
NEGATORS = {"no", "not", "nor", "never", "n't"}
STOPWORDS = ENGLISH_STOP_WORDS.difference({"no", "not", "nor", "never"})


def get_lemmatizer(enabled: bool):
    if not enabled:
        return None
    try:
        from nltk import download
        from nltk.corpus import wordnet  # noqa: F401
        from nltk.stem import WordNetLemmatizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Lemmatization requires nltk. Install it with `pip install nltk`."
        ) from exc

    try:
        _ = wordnet.synsets("bank")
    except LookupError:  # pragma: no cover - environment dependent
        download("wordnet")
        download("omw-1.4")

    return WordNetLemmatizer()


def _expand_contractions(text: str) -> str:
    return re.sub(r"n\'t\b", " not", text)


def _apply_negation(tokens):
    out = []
    skip_next = False
    for idx, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if tok in NEGATORS and idx + 1 < len(tokens):
            out.append(f"not_{tokens[idx + 1]}")
            skip_next = True
        else:
            out.append(tok)
    return out


def preprocess_text(
    text: str,
    *,
    lemmatize: bool = False,
    lemmatizer=None,
    use_negation: bool = False,
) -> str:
    text = text.lower()
    text = _expand_contractions(text)
    tokens = TOKEN_RE.findall(text)
    if use_negation:
        tokens = _apply_negation(tokens)
    tokens = [t for t in tokens if t not in STOPWORDS]
    if lemmatize:
        if lemmatizer is None:
            lemmatizer = get_lemmatizer(True)
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_corpus(texts, *, lemmatize: bool = False, use_negation: bool = False):
    lemmatizer = get_lemmatizer(lemmatize)
    return [
        preprocess_text(
            t, lemmatize=lemmatize, lemmatizer=lemmatizer, use_negation=use_negation
        )
        for t in texts
    ]


def preprocess_corpus_transform(texts, *, lemmatize: bool = False, use_negation: bool = False):
    return preprocess_corpus(texts, lemmatize=lemmatize, use_negation=use_negation)


def build_tfidf_pipeline(
    *,
    lemmatize: bool = False,
    use_negation: bool = False,
    max_features: int | None = None,
    ngram_range: tuple[int, int] = (1, 1),
):
    return Pipeline(
        steps=[
            (
                "preprocess",
                FunctionTransformer(
                    preprocess_corpus_transform,
                    validate=False,
                    kw_args={"lemmatize": lemmatize, "use_negation": use_negation},
                ),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    token_pattern=r"(?u)[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?%?|[$%]|[+-]",
                    max_features=max_features,
                    ngram_range=ngram_range,
                ),
            ),
        ]
    )
