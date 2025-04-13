from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np
import torch


def process_input(input_words: list[str], word_embedding: dict) -> torch.Tensor:
    input_words = " ".join(input_words)
    input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

    embedding_dim = len(next(iter(word_embedding.values()), None))
    unk_embedding = word_embedding.get("<UNK>", torch.zeros(embedding_dim))

    vectors = [
        torch.tensor(word_embedding.get(i.lower(), unk_embedding)) if isinstance(word_embedding.get(i.lower(), unk_embedding), np.ndarray)
        else word_embedding.get(i.lower(), unk_embedding).clone().detach()
        for i in input_words
    ]
    vectors_tensor = torch.stack(vectors)

    return vectors_tensor.view(len(vectors_tensor), 1, -1)

def load_data(train_path: str, val_path: str) -> tuple[list[tuple[list[str], int]], list[tuple[list[str], int]]]:
    with Path(train_path).open() as train_file:
        training = json.load(train_file)
    with Path(val_path).open() as val_file:
        validation = json.load(val_file)

    train = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]

    return train, val
