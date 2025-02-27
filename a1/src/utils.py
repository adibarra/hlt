from __future__ import annotations

import collections
import math
import random
import re
from pathlib import Path
from typing import Literal


def load_corpus(file_path: str) -> list[str]:
    with Path(file_path).open(encoding="utf-8") as f:
        return f.readlines()

def preprocess(corpus: list[str]) -> list[str]:
    """Preprocess the text: lowercase, remove punctuation, and tokenize."""
    processed_corpus = []
    regex = re.compile(r"[^a-zA-Z0-9\s]")
    for line in corpus:
        line = line.lower()  # noqa: PLW2901
        line = regex.sub(" ", line)  # noqa: PLW2901
        tokens = line.split()
        processed_corpus.extend(tokens) # preferred over append() to add ind. words
    return processed_corpus 

def preprocess_n_grams(corpus: List[str]) -> List[str]: 
    """Preprocess the text: split sentences, lowercase, remove punctuation, and tokenize into unigrams."""
    processed_corpus = []
    unigrams = []
    for line in corpus: # list of lines -> list of unigrams
        line = line.lower()
        words = line.split()
        unigrams.extend(words)
    
    curr_sentence = []
    sentences = []
    for token in unigrams: # list of unigrams -> list of sentences (each sentence is list of words)
        curr_sentence.append(token)
        if re.match(r'[.!?]', token):
            sentences.append(curr_sentence)
            curr_sentence = []
    
    for sentence in sentences: # use sentence boundaries to add <s> and </s>
        cleaned_sentence = ["<s>"]
        for word in sentence:
            cleaned_word = re.sub(r'[^a-zA-Z0-9\s]', '', word) # regex to remove special chars
            if cleaned_word != "":
                cleaned_sentence.append(cleaned_word)
        cleaned_sentence.append("</s>")
        processed_corpus.extend(cleaned_sentence)
    
    return processed_corpus

def unigrams_to_bigrams(unigrams: List[str]) -> List[str]:
    return list(zip(unigrams[:-1], unigrams[1:]))

def handle_unknown_words(tokens: List[str], known_vocab: set, method: Literal['replacement', 'deletion'] = None) -> List[str]:
    """Replace unknown words with the <UNK> token."""
    match method:
        case 'replacement':
            return [token if token in known_vocab else '<UNK>' for token in tokens]
        case 'deletion': # Delete unknown words instead
            return [token for token in tokens if token in known_vocab]
        case _:
            return tokens

def build_unigram_model(tokens: list[str], cutoff: int | None, smoothing: Literal["laplace", "add-k"] | None, k: int = 1, *, debug: bool = False) -> tuple[dict[str, float], collections.Counter, int]:
    """Build a unigram model with optional smoothing methods."""
    unigram_counts = collections.Counter(tokens)
    unigram_counts = {word: count for word, count in unigram_counts.items() if count > cutoff}
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

def build_bigram_model(bigram_tokens: List[str], unigrams: List[str], smoothing: Literal['laplace', 'add-k'] = None, k: int = 1, debug: bool = False) -> Tuple[Dict[str, float], collections.Counter, int]:
    """Build a bigram model with optional smoothing methods."""
    bigram_counts = collections.Counter(bigram_tokens) 
    unigram_counts = collections.Counter(unigrams)
    vocab_size = len(unigram_counts) + 1 # add 1 for <UNK> token

    # calculate probabilities
    bigram_probs = {}
    for bigram, count in bigram_counts.items(): # formatted as dict bigram: count
        w_1 = bigram[0] # w_(n-1)
        match smoothing:
            case 'laplace':
                prob = (count + 1) / (unigram_counts[w_1] + vocab_size)
            case 'add-k':
                prob = (count + k) / (unigram_counts[w_1] + k * vocab_size)
            case _:
                prob = count / unigram_counts[w_1]

        bigram_probs[bigram] = prob

    # handle <UNK>
    match smoothing: # not sure how to handle <UNK> in bigrams (sep words, whole bigrams, etc.)
        case 'laplace': # w_n-1 = 0 for now
            bigram_probs['<UNK>'] = 1 / (vocab_size)
        case 'add-k':
            bigram_probs['<UNK>'] = k / (k * vocab_size)
        case _:
            bigram_probs['<UNK>'] = 0

    # print debug info
    if (debug):
        print(f"{'Word':<15s} {'Count':<15} {'Probability':<15s}")
        print('-' * 50)
        for i, (word, prob) in enumerate(bigram_probs.items()):
            if i >= 30:
                break
            print(f"{word:<15s} {unigram_counts[word]:<15} {prob:.6f}")
        print()

    return bigram_probs, bigram_counts, unigram_counts, len(bigram_tokens)

def calculate_perplexity(tokens: List[str], token_probs: Dict[str, float]) -> float:
    """Calculate the perplexity of a dataset using the unigram model."""
    log_sum = 0
    for token in tokens:
        prob = token_probs.get(token, token_probs['<UNK>']) # uses <UNK> if token not found
        log_sum += math.log(prob)

    return math.exp(-log_sum / len(tokens))

def sample_unigram(unigram_probs: dict[str, float], n: int) -> list[str]:
    """Sample n tokens from the unigram model based on their probabilities."""
    words = list(unigram_probs.keys())
    probabilities = list(unigram_probs.values())

    sampled_tokens = []
    for _ in range(n):
        sampled_token = random.choices(words, probabilities)[0]
        sampled_tokens.append(sampled_token)

    return sampled_tokens
