from datasets import load_dataset

from preprocessing.config import DATASET_NAME

def load_phrasebank(subset: str):
    ds = load_dataset(DATASET_NAME, subset, trust_remote_code=True)
    if "train" in ds:
        data = ds["train"]
    else:
        data = ds
    texts = data["sentence"]
    labels = data["label"]
    label_names = None
    try:
        label_names = data.features["label"].names
    except Exception:
        label_names = None
    return texts, labels, label_names
