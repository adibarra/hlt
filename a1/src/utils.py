import collections
import math
import re
from typing import List, Literal, Tuple, Dict

def load_corpus(file_path: str) -> List[str]:
    """Load the dataset from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def preprocess_unigrams(corpus: List[str]) -> List[str]:
    # Preprocess the text: lowercase, remove punctuation, and tokenize.
    processed_corpus = []
    for line in corpus:
        line = line.lower()
        line = re.sub(r'[^a-zA-Z0-9\s]', '', line) # regex to remove special chars
        tokens = line.split()
        processed_corpus.extend(tokens) # preferred over append() to add ind. words
    return processed_corpus 

def preprocess_n_grams(corpus: List[str]) -> List[str]:
    """Preprocess the text: split sentences, lowercase, remove punctuation, and tokenize into unigrams."""
    processed_corpus = []
    for line in corpus:
        start_pattern = r'((?<=^)|(?<=[.!?]\s+))([A-Z])'
        end_pattern = r'(?<=[.!?])\s+' 
        line = re.sub(start_pattern, rf'{'<s>'}\2', line)
        line = re.sub(end_pattern, rf'{'</s>'}\2', line)

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence) # regex to remove special chars
        tokens = ["<s>"] + sentence.split() + ["</s>"] # add start and end tokens
        processed_corpus.append(tokens)
    processed_corpus = [token for sentence in processed_corpus for token in sentence] # convert sentences to unigram tokens
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

def build_unigram_model(tokens: List[str], smoothing: Literal['laplace', 'add-k'] = None, k: int = 1, debug: bool = False) -> Tuple[Dict[str, float], collections.Counter, int]:
    """Build a unigram model with optional smoothing methods."""
    unigram_counts = collections.Counter(tokens)
    vocab_size = len(unigram_counts) + 1 # add 1 for <UNK> token
    total_tokens = len(tokens)

    # calculate probabilities
    unigram_probs = {}
    for word, count in unigram_counts.items():
        match smoothing:
            case 'laplace':
                prob = (count + 1) / (total_tokens + vocab_size)
            case 'add-k':
                prob = (count + k) / (total_tokens + k * vocab_size)
            case _:
                prob = count / total_tokens

        unigram_probs[word] = prob

    # handle <UNK>
    match smoothing:
        case 'laplace':
            unigram_probs['<UNK>'] = 1 / (total_tokens + vocab_size)
        case 'add-k':
            unigram_probs['<UNK>'] = k / (total_tokens + k * vocab_size)
        case _:
            unigram_probs['<UNK>'] = 0

    # print debug info
    if (debug):
        print(f"{'Word':<15s} {'Count':<15} {'Probability':<15s}")
        print('-' * 50)
        for i, (word, prob) in enumerate(unigram_probs.items()):
            if i >= 30:
                break
            print(f"{word:<15s} {unigram_counts[word]:<15} {prob:.6f}")
        print()

    return unigram_probs, unigram_counts, total_tokens

def build_bigram_model(tokens: List[str], unigrams: List[str], smoothing: Literal['laplace', 'add-k'] = None, k: int = 1, debug: bool = False) -> Tuple[Dict[str, float], collections.Counter, int]:
    """Build a bigram model with optional smoothing methods."""
    bigram_counts = collections.Counter(tokens) 
    unigram_counts = collections.Counter(unigrams)
    vocab_size = len(unigram_counts) + 1 # add 1 for <UNK> token

    # calculate probabilities
    bigram_probs = {}
    for bigram, count in bigram_counts.items(): # formatted as dict bigram: count
        w_1 = bigram[0] # word_1
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
            unigram_probs['<UNK>'] = 0

    # print debug info
    if (debug):
        print(f"{'Word':<15s} {'Count':<15} {'Probability':<15s}")
        print('-' * 50)
        for i, (word, prob) in enumerate(bigram_probs.items()):
            if i >= 30:
                break
            print(f"{word:<15s} {unigram_counts[word]:<15} {prob:.6f}")
        print()

    return bigram_probs, bigram_counts, len(tokens)

def calculate_perplexity(tokens: List[str], token_probs: Dict[str, float]) -> float:
    """Calculate the perplexity of a dataset using the unigram model."""
    log_sum = 0
    for token in tokens:
        prob = token_probs.get(token, token_probs['<UNK>']) # uses <UNK> if token not found
        log_sum += math.log(prob)

    return math.exp(-log_sum / len(tokens))
