from pathlib import Path

import numpy as np
import torch


MODEL_NAME = "ProsusAI/finbert"


def compute_embeddings(
    texts,
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
):
    # Use the CLS token embedding as a sentence vector
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)
    return np.vstack(embeddings)


def load_or_create_embeddings(
    texts,
    cache_path: Path,
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    expected_len: int | None = None,
):
    # Reuse cached embeddings when the expected size matches
    if cache_path.exists():
        cached = np.load(cache_path)
        if expected_len is None or cached.shape[0] == expected_len:
            return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = compute_embeddings(
        texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
    )
    np.save(cache_path, embeddings)
    return embeddings
