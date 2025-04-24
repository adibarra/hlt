from __future__ import annotations

import csv
import random
import re
import string
from pathlib import Path

SEED = 42

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def tokenize(text: str) -> list[str]:
    return text.split()

def load_sentiment_csv(path: str) -> list[tuple[str, int]]:
    data = []
    with Path.open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:  # noqa: PLR2004
                continue
            raw_text, label = row[0], row[1].strip().lower()
            if label not in LABEL_MAP:
                continue
            cleaned_text = preprocess_text(raw_text)
            data.append((cleaned_text, LABEL_MAP[label]))
    return data

def split_data(data: list) -> tuple[list, list, list]:
    random.shuffle(data)
    total = len(data)

    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.2)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data
