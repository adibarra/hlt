# Final Project – Comparison of Model Architectures for Sentiment Analysis

This directory contains our implementation for the **Final Project**, focused on the comparison of multiple model architectures for sentiment analysis.

## Team Members
- Alec Ibarra (adi220000)
- Annette Llanas (ajl200006)
- Ashlee Kang (ajk200003)
- Syed Kabir (snk210004)

## Overview

In our **Final Project**, we implemented and compared four distinct model architectures for **sentiment analysis**, each representing a different class of machine learning or deep learning approaches:

- **Logistic Regression with TF-IDF**: A traditional linear model using term frequency–inverse document frequency to convert text into numerical features. It serves as a strong baseline for sentiment classification tasks.

- **LSTM with Pretrained Embeddings**: A recurrent neural network that leverages GloVe embeddings to model sequential dependencies in text. This architecture captures word order and context, making it well-suited for sentiment detection.

- **Text CNN**: This model uses 1D convolutional filters to capture local patterns in text, such as common word combinations or n-grams. It's a fast and effective approach for sentence-level classification tasks like sentiment analysis.

- **BERT (Transformer-based Model)**: A fine-tuned version of `bert-base-uncased` from the Hugging Face Transformers library. BERT's contextual embeddings and bidirectional attention make it one of the most powerful tools for NLP tasks, including sentiment analysis.

These models were trained and evaluated on real-world sentiment datasets, and their performances were compared using standard classification metrics. Our goal was to understand the trade-offs between model complexity, training time, and accuracy in sentiment prediction.

## Report and Documentation
The report for this assignment can be found in the `docs` directory. It includes details on the models, training process, and evaluation metrics.

## Run the Models
All code should be run from the root directory of the repository.

Follow the instructions [here](../#repository-setup-vscode) for repository setup.

<!-- TODO: Add usage instructions or example command-line usage if needed -->
