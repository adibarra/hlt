from utils import (
    build_bigram_model,
    calculate_perplexity,
    handle_unknown_words,
    load_corpus,
    preprocess_n_grams,
    unigrams_to_bigrams,
)


def main() -> None:
    # load the data
    train_corpus = load_corpus("a1/src/data/train.txt")
    validation_corpus = load_corpus("a1/src/data/val.txt")

    # preprocess data into unigrams
    train_unigrams = preprocess_n_grams(train_corpus)
    validation_unigrams = preprocess_n_grams(validation_corpus)

    # preprocess unigrams into bigrams
    train_tokens = unigrams_to_bigrams(train_unigrams)
    validation_tokens = unigrams_to_bigrams(validation_unigrams)

    # set configuration
    k_value = 0.5
    smoothing_method = "add-k"
    unk_method = "deletion"

    # build the bigram model
    bigram_probs, bigram_counts, unigram_counts, total_tokens = build_bigram_model(
        train_tokens, train_unigrams, smoothing=smoothing_method, k=k_value,
    )

    # handle unknown words in the validation set
    validation_tokens = handle_unknown_words(validation_tokens, bigram_counts, unk_method)

    # print statistics
    print(f"\n{'Bigram Model Statistics':^125s}")
    print("-" * 125)
    print(f"{'Total Tokens':<20s}: {total_tokens}")
    print(f"{'Unique Tokens':<20s}: {len(bigram_probs)}")
    print(f"{'Smoothing Method':<20s}: {smoothing_method if smoothing_method else 'unsmoothed'}", end="")
    print(f"{' (k=' + str(k_value) + ')' if smoothing_method == 'add-k' else ''}")
    print(f"{'Unknown Method':<20s}: {unk_method}")
    print(f"{'Perplexity':<20s}: {calculate_perplexity(validation_tokens, bigram_probs):.4f}")
    print("-" * 125, "\n")

if __name__ == "__main__":
    main()
