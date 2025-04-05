# Assignment 2 - Neural Networks for Sentiment Analysis

This directory contains our implementation for **Assignment 2**.

## Team Members
- Alec Ibarra (adi220000)

## Overview
In **Assignment 2**, we implement neural network models for a 5-class sentiment analysis task using Yelp reviews. The two models are:
- **Feedforward Neural Network (FFNN)**: A simple architecture for classifying reviews based on bag-of-words vectors.
- **Recurrent Neural Network (RNN)**: A more advanced model that uses word embeddings to process sequences of words, capturing dependencies across words in the review.

The forward pass was completed for both models, and they were trained on the provided datasets to predict sentiment ratings. This assignment also provided an introduction to PyTorch and neural network techniques for text classification tasks.

## Report and documentation
The report for this assignment can be found in the `docs` directory. It includes details on the models, training process, and evaluation metrics.

## Run the Models
All code should be run from the root directory of the repository.

- **To run the FFNN model**:
    ```bash
    python ./a2/src/ffnn.py
    ```
- **To run the RNN model**:
    ```bash
    python ./a2/src/rnn.py
    ```
