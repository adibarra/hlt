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

- **TinyBERT (Transformer-based Model)**: A distilled version of BERT from 'huawei-noah/TinyBERT_General_4L_312D' of the Hugging Face Transformers Library. It is 7.5x smaller and 9.4x faster, while retaining comparable performance to its predecessor, as per the paper <a href="https://arxiv.org/abs/1909.10351"> TinyBERT: Distilling BERT for Natural Language Understanding </a>

These models were trained and evaluated on real-world sentiment datasets, and their performances were compared using standard classification metrics. Our goal was to understand the trade-offs between model complexity, training time, and accuracy in sentiment prediction.

## Report and Documentation
The report for this assignment can be found in the `docs` directory. It includes details on the models, training process, and evaluation metrics.

## Run the Models
All code should be run from the root directory of the repository.

Follow the instructions [here](../#repository-setup-vscode) for repository setup.

- **To run the Linear Regression model**:
    ```bash
    python project/src/lr.py
    ```
