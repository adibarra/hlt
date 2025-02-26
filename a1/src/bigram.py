from utils import load_corpus, preprocess_n_grams, unigrams_to_bigrams, build_bigram_model, handle_unknown_words, calculate_perplexity

def main(): # TODO: print
    # load the data
    train_corpus = load_corpus('a1/src/A1_DATASET/train.txt')
    validation_corpus = load_corpus('a1/src/A1_DATASET/val.txt')

    # preprocess data into unigrams
    train_unigrams = preprocess_n_grams(train_corpus)
    validation_unigrams = preprocess_n_grams(validation_corpus)

    # preprocess unigrams into bigrams
    train_tokens = unigrams_to_bigrams(train_unigrams)
    validation_tokens = unigrams_to_bigrams(validation_unigrams)
    
    # set configuration
    unk_method = 'replacement'
    smoothing_method = 'add-k'
    k_value = 0.5
    
    # build the bigram model
    bigram_probs, bigram_counts, total_tokens = build_bigram_model(
        train_tokens, train_unigrams, smoothing=smoothing_method, k=k_value
    )

    # handle unknown words in the validation set
    validation_tokens = handle_unknown_words(validation_tokens, bigram_probs, unk_method)

    # calculate perplexity
    perplexity = calculate_perplexity(validation_tokens, bigram_probs)

    # print statistics
    print(f"\n{'Bigram Model Statistics':^50s}")
    print('-' * 50)
    print(f"{'Total Tokens':<20s}: {total_tokens}")
    print(f"{'Unique Tokens':<20s}: {len(bigram_probs)}")
    print(f"{'Smoothing Method':<20s}: {smoothing_method if smoothing_method else 'unsmoothed'}", end='')
    print(f"{' (k=' + str(k_value) + ')' if smoothing_method == 'add-k' else ''}")
    print(f"{'Perplexity':<20s}: {perplexity:.4f}")
    print('-' * 50, '\n') 
    
if __name__ == '__main__':
    main()