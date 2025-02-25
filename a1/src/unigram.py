from utils import (
    build_unigram_model,
    calculate_unigram_perplexity,
    handle_unknown_words,
    load_corpus,
    preprocess,
    sample_unigram,
)


def main() -> None:
    # load the data
    train_corpus = load_corpus("a1/src/A1_DATASET/train.txt")
    validation_corpus = load_corpus("a1/src/A1_DATASET/val.txt")

    # preprocess the data
    train_tokens = preprocess(train_corpus)
    validation_tokens = preprocess(validation_corpus)

    # set configuration
    unk_method = "replacement"
    smoothing_method = "add-k"
    k_value = 0.5

    # build the unigram model
    unigram_probs, unigram_counts, total_tokens = build_unigram_model(
        train_tokens, smoothing=smoothing_method, k=k_value,
    )

    # handle unknown words in the validation set
    validation_tokens = handle_unknown_words(validation_tokens, unigram_probs, unk_method)

    # print statistics
    print(f"\n{'Unigram Model Statistics':^125s}")
    print("-" * 125)
    print(f"{'Total Tokens':<20s}: {total_tokens}")
    print(f"{'Unique Tokens':<20s}: {len(unigram_probs)}")
    print(f"{'Smoothing Method':<20s}: {smoothing_method if smoothing_method else 'unsmoothed'}", end="")
    print(f"{' (k=' + str(k_value) + ')' if smoothing_method == 'add-k' else ''}")
    print(f"{'Unknown Method':<20s}: {unk_method}")
    print(f"{'Perplexity':<20s}: {calculate_unigram_perplexity(validation_tokens, unigram_probs):.4f}")
    print(f"{'Sample':<20s}: {' '.join(sample_unigram(unigram_probs, 15))}")
    print("-" * 125, "\n")

if __name__ == "__main__":
    main()
