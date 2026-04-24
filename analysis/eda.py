"""Exploratory Data Analysis for Financial PhraseBank:
This script is used for examining the dataser before any training for models.
It examines the dataset from six angles: label distribution to understand class imbalance,
sentence length statistics to find differences across sentiment classes, top words per class
to identify discriminative vocabulary, financial keyword frequency to measure domain-specific
term presence, number and percentage analysis to quantify numeric content across classes,
and sample sentences for qualitative inspection. The findings from this analysis directly
inform the handcrafted feature engineering and modeling decisions in the baseline pipeline.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

from preprocessing.load_data import load_phrasebank
from preprocessing.preprocess_text import TOKEN_RE

"""Gathering financial keywords for analysis:"""

FINANCIAL_KEYWORDS = {
    "profit", "loss", "growth", "decline", "revenue", "earnings",
    "exceeded", "rose", "fell", "increased", "decreased", "gain",
    "drop", "surged", "plunged", "dividend", "forecast", "margin",
    "debt", "sales", "cost", "income", "expense", "deficit", "surplus",
}

"""Print count and percentage per class."""

def label_distribution(labels, label_names):

    counts = Counter(labels)
    total = len(labels)
    print("=" * 50)
    print("LABEL DISTRIBUTION")
    print("=" * 50)
    for idx in sorted(counts):
        name = label_names[idx] if label_names else str(idx)
        count = counts[idx]
        pct = count / total * 100
        print(f"  {name:>10s}: {count:5d}  ({pct:.1f}%)")
    print(f"  {'TOTAL':>10s}: {total:5d}")
    print()

"""Print word-count statistics per class."""
def sentence_length_stats(texts, labels, label_names):

    from collections import defaultdict

    lengths_by_class = defaultdict(list)
    for text, label in zip(texts, labels):
        token_count = len(TOKEN_RE.findall(text))
        lengths_by_class[label].append(token_count)

    print("=" * 50)
    print("SENTENCE LENGTH (token count) PER CLASS")
    print("=" * 50)
    for idx in sorted(lengths_by_class):
        name = label_names[idx] if label_names else str(idx)
        lens = sorted(lengths_by_class[idx])
        avg = sum(lens) / len(lens)
        median = lens[len(lens) // 2]
        print(
            f"  {name:>10s}:  avg={avg:.1f}  median={median}  "
            f"min={lens[0]}  max={lens[-1]}"
        )
    print()

"""Print the most common words per class after lowercasing."""

def top_words_per_class(texts, labels, label_names, top_n=15):

    from collections import defaultdict

    word_counts = defaultdict(Counter)
    for text, label in zip(texts, labels):
        tokens = TOKEN_RE.findall(text.lower())
        word_counts[label].update(tokens)

    print("=" * 50)
    print(f"TOP {top_n} WORDS PER CLASS")
    print("=" * 50)
    for idx in sorted(word_counts):
        name = label_names[idx] if label_names else str(idx)
        print(f"\n  [{name}]")
        for word, count in word_counts[idx].most_common(top_n):
            print(f"    {word:20s} {count}")
    print()



"""Check how often financial keywords appear per class."""

def financial_keyword_analysis(texts, labels, label_names):

    from collections import defaultdict

    hits = defaultdict(lambda: Counter())
    class_totals = Counter()

    for text, label in zip(texts, labels):
        tokens = set(TOKEN_RE.findall(text.lower()))
        class_totals[label] += 1
        for kw in FINANCIAL_KEYWORDS:
            if kw in tokens:
                hits[label][kw] += 1

    print("=" * 50)
    print("FINANCIAL KEYWORD FREQUENCY PER CLASS")
    print("=" * 50)
    for idx in sorted(hits):
        name = label_names[idx] if label_names else str(idx)
        total = class_totals[idx]
        print(f"\n  [{name}] ({total} sentences)")
        for kw, count in hits[idx].most_common(10):
            pct = count / total * 100
            print(f"    {kw:20s} {count:4d}  ({pct:.1f}%)")
    print()


"""Count sentences containing numbers or percentages per class."""

def number_percentage_analysis(texts, labels, label_names):

    import re

    num_re = re.compile(r"\d+(?:\.\d+)?%?")
    pct_re = re.compile(r"\d+(?:\.\d+)?%")

    from collections import defaultdict

    stats = defaultdict(lambda: {"total": 0, "has_number": 0, "has_percent": 0})

    for text, label in zip(texts, labels):
        stats[label]["total"] += 1
        if num_re.search(text):
            stats[label]["has_number"] += 1
        if pct_re.search(text):
            stats[label]["has_percent"] += 1

    print("=" * 50)
    print("NUMBER / PERCENTAGE PRESENCE PER CLASS")
    print("=" * 50)
    for idx in sorted(stats):
        name = label_names[idx] if label_names else str(idx)
        s = stats[idx]
        total = s["total"]
        num_pct = s["has_number"] / total * 100
        pct_pct = s["has_percent"] / total * 100
        print(
            f"  {name:>10s}:  has_number={s['has_number']:4d} ({num_pct:.1f}%)  "
            f"has_percent={s['has_percent']:4d} ({pct_pct:.1f}%)"
        )
    print()


"""Print a few example sentences per class."""

def sample_sentences(texts, labels, label_names, n=3):

    from collections import defaultdict

    examples = defaultdict(list)
    for text, label in zip(texts, labels):
        if len(examples[label]) < n:
            examples[label].append(text)

    print("=" * 50)
    print(f"SAMPLE SENTENCES ({n} per class)")
    print("=" * 50)
    for idx in sorted(examples):
        name = label_names[idx] if label_names else str(idx)
        print(f"\n  [{name}]")
        for sent in examples[idx]:
            print(f"    - {sent}")
    print()


def main():
    subset = "sentences_allagree"
    print(f"Loading PhraseBank subset: {subset}\n")
    texts, labels, label_names = load_phrasebank(subset)

    label_distribution(labels, label_names)
    sentence_length_stats(texts, labels, label_names)
    top_words_per_class(texts, labels, label_names)
    financial_keyword_analysis(texts, labels, label_names)
    number_percentage_analysis(texts, labels, label_names)
    sample_sentences(texts, labels, label_names)


if __name__ == "__main__":
    main()