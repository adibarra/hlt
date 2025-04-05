# CS 4395 - Human Language Technologies (NLP)
**Assignment 1 - Unigram and Bigram Language Models**

This directory contains our implementation for **Assignment 1**.

## Team Members
- Alec Ibarra (adi220000)
- Annette Llanas (ajl200006)
- Ashlee Kang (ajk200003)
- Syed Kabir (snk210004)

## Overview
In **Assignment 1**, we implemented two primary types of language models:
- **Unigram Model**: A probabilistic model that estimates the likelihood of individual words, without considering any word context.
- **Bigram Model**: A probabilistic model that estimates the likelihood of a word given its previous word, incorporating word pair dependencies.

Additionally, we applied several techniques to improve the models:
- **Smoothing**: We utilized Laplace (Add-One) and Add-k smoothing methods to address the issue of zero probabilities for unseen words.
- **Unknown Word Handling**: We applied strategies such as replacement, deletion, and retention for handling unseen words during inference.
- **Perplexity Evaluation**: We calculated the perplexity of both models to evaluate their performance.

This assignment also served as an introduction to probabilistic language models and their evaluation techniques.

## Report
The report for this assignment can be found in the `docs` directory.

## Run the Models
All code should be run from the root directory of the repository.

- **To run the Unigram model**:
    ```bash
    python ./a1/src/unigram.py
    ```
- **To run the Bigram model**:
    ```bash
    python ./a1/src/bigram.py
    ```
