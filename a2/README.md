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

Follow the instructions [here](../#repository-setup-vscode) for repository setup.

**Note: Make sure to unzip the word_embeddings.zip file and put the resulting .pkl file in the src directory.**

- **To run the FFNN model**:
    ```bash
    python a2/src/ffnn.py -hd 128 -e 30 -p 5 --train-data a2/src/data/train.json --val-data a2/src/data/val.json --test-data a2/src/data/test.json --do-train

    # or to use defaults
    python a2/src/ffnn.py --do-train
    ```
- **To run the RNN model**:
    ```bash
    python a2/src/rnn.py -hd 128 -e 30 -p 5 --train-data a2/src/data/train.json --val-data a2/src/data/val.json --test-data a2/src/data/test.json --do-train

    # or to use defaults
    python a2/src/rnn.py --do-train
    ```
