from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np
import torch

UNK = "<UNK>"
def process_input(input_words: list[str], word_embedding: dict) -> torch.Tensor:
    input_words = " ".join(input_words)
    input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

    embedding_dim = len(next(iter(word_embedding.values()), None))
    unk_embedding = word_embedding.get(UNK, torch.zeros(embedding_dim))

    vectors = [
        torch.tensor(word_embedding.get(i.lower(), unk_embedding)) if isinstance(word_embedding.get(i.lower(), unk_embedding), np.ndarray)
        else word_embedding.get(i.lower(), unk_embedding).clone().detach()
        for i in input_words
    ]

    return torch.stack(vectors)

def load_data(train_path: str, val_path: str) -> tuple[list[tuple[list[str], int]], list[tuple[list[str], int]]]:
    with Path(train_path).open() as train_file:
        training = json.load(train_file)
    with Path(val_path).open() as val_file:
        validation = json.load(val_file)

    train = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]

    return train, val

def make_vocab(data: list[tuple[list[str], int]]) -> set[str]:
    vocab = set()
    for document, _ in data:
        vocab.update(document)
    vocab.add(UNK)
    return vocab


def make_indices(vocab: set[str]) -> tuple[set[str], dict[str, int], dict[int, str]]:
    vocab_list = sorted(vocab)
    word2index = {word: idx for idx, word in enumerate(vocab_list)}
    index2word = {idx: word for word, idx in word2index.items()}
    return vocab, word2index, index2word


def convert_to_vector_representation(
    data: list[tuple[list[str], int]],
    word2index: dict[str, int],
) -> list[tuple[torch.Tensor, int]]:
    vectorized_data = []
    for document, label in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            idx = word2index.get(word, word2index[UNK])
            vector[idx] += 1
        vectorized_data.append((vector, label))
    return vectorized_data
