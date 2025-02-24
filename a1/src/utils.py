from __future__ import annotations

import collections
import math
import re
from pathlib import Path
from typing import Literal


def load_corpus(file_path: str) -> list[str]:
    with Path(file_path).open(encoding="utf-8") as f:
        return f.readlines()

def preprocess(corpus: list[str]) -> list[str]:
    """Preprocess the text: lowercase, remove punctuation, and tokenize."""
    processed_corpus = []
    for line in corpus:
        line = line.lower()  # noqa: PLW2901
        line = re.sub(r"[^a-zA-Z0-9\s]", "", line)  # noqa: PLW2901
        tokens = line.split()
        processed_corpus.extend(tokens)
    return processed_corpus

def handle_unknown_words(tokens: list[str], known_vocab: set, method: Literal["replacement", "deletion"] | None) -> list[str]:
    """Replace unknown words with the <UNK> token."""
    match method:
        case "replacement":
            return [token if token in known_vocab else "<UNK>" for token in tokens]
        case "deletion":
            return [token for token in tokens if token in known_vocab]
        case _:
            return tokens

def build_unigram_model(tokens: list[str], smoothing: Literal["laplace", "add-k"] | None, k: int = 1, *, debug: bool = False) -> tuple[dict[str, float], collections.Counter, int]:
    """Build a unigram model with optional smoothing methods."""
    unigram_counts = collections.Counter(tokens)
    vocab_size = len(unigram_counts) + 1 # add 1 for <UNK> token
    total_tokens = len(tokens)

    # calculate probabilities
    unigram_probs = {}
    for word, count in unigram_counts.items():
        match smoothing:
            case "laplace":
                prob = (count + 1) / (total_tokens + vocab_size)
            case "add-k":
                prob = (count + k) / (total_tokens + k * vocab_size)
            case _:
                prob = count / total_tokens

        unigram_probs[word] = prob

    # handle <UNK>
    match smoothing:
        case "laplace":
            unigram_probs["<UNK>"] = 1 / (total_tokens + vocab_size)
        case "add-k":
            unigram_probs["<UNK>"] = k / (total_tokens + k * vocab_size)
        case _:
            unigram_probs["<UNK>"] = 0

    # print debug info
    if (debug):
        print(f"{'Word':<15s} {'Count':<15} {'Probability':<15s}")
        print("-" * 50)
        for i, (word, prob) in enumerate(unigram_probs.items()):
            if i >= 30:  # noqa: PLR2004
                break
            print(f"{word:<15s} {unigram_counts[word]:<15} {prob:.6f}")
        print()

    return unigram_probs, unigram_counts, total_tokens

def calculate_unigram_perplexity(tokens: list[str], unigram_probs: dict[str, float]) -> float:
    """Calculate the perplexity of a dataset using the unigram model."""
    log_sum = 0
    for token in tokens:
        prob = unigram_probs.get(token, unigram_probs["<UNK>"])
        log_sum += math.log(prob)

    return math.exp(-log_sum / len(tokens))
