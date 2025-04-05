# CS 4395 - Human Language Technologies (NLP)

This repository contains our team's code for CS 4395.

[Tested with Python 3.12.3]

## Repository Setup (VSCode)

1. `Ctrl+Shift+P` ➜ `Git: Clone` ➜ `adibarra/hlt`
2. `Ctrl+Shift+P` ➜ `Python: Create Environment`
3. Install dependencies if prompted.
4. Thats it!

## Running the Models

Each assignment has its own set of instructions for running the models.

## Assignment 1 - Unigram and Bigram Language Models

[[View the report, implementation, and instructions]](./a1/)

In **Assignment 1**, we implemented two primary types of language models:
- **Unigram Model**: A probabilistic model that estimates the likelihood of individual words, without considering any word context.
- **Bigram Model**: A probabilistic model that estimates the likelihood of a word given its previous word, incorporating word pair dependencies.

Additionally, we applied several techniques to improve the models:
- **Smoothing**: We utilized Laplace (Add-One) and Add-k smoothing methods to address the issue of zero probabilities for unseen words.
- **Unknown Word Handling**: We applied strategies such as replacement, deletion, and retention for handling unseen words during inference.
- **Perplexity Evaluation**: We calculated the perplexity of both models to evaluate their performance.

This assignment also served as an introduction to probabilistic language models and their evaluation techniques.

## Assignment 2 - Neural Networks for Sentiment Analysis

[[View the report, implementation, and instructions]](./a2/)

In **Assignment 2**, we implement neural network models for a 5-class sentiment analysis task using Yelp reviews. The two models are:
- **Feedforward Neural Network (FFNN)**: A simple architecture for classifying reviews based on bag-of-words vectors.
- **Recurrent Neural Network (RNN)**: A more advanced model that uses word embeddings to process sequences of words, capturing dependencies across words in the review.

The forward pass was completed for both models, and they were trained on the provided datasets to predict sentiment ratings. This assignment also provided an introduction to PyTorch and neural network techniques for text classification tasks.
