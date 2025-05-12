import random
import time
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from utils import (
    SEED,
    load_dataset,
    load_glove,
    print_reports,
    split_data,
)

GLOVE_PATH = Path("project/src/data/glove.6B.50d.txt")
GLOVE_ZIP_PATH = Path("project/src/data/glove.6B.50d.txt.zip")

def build_lstm_model(
    vocab_size: int,
    embedding_dim: int,
    embedding_matrix: np.ndarray,
) -> Sequential:
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=True,
        ),
        Bidirectional(
            LSTM(
                64,
                dropout=0.3,
                recurrent_dropout=0.3,
                return_sequences=False,
        )),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_sentiment_analysis(data: list) -> tuple[dict, str, int]:
    print(">>> Tokenizing and Vectorizing Dataset")
    train_data, val_data, test_data = split_data(data)

    X_train_texts, y_train = zip(*train_data)  # noqa: N806
    X_val_texts, y_val = zip(*val_data)  # noqa: N806
    X_test_texts, y_test = zip(*test_data)  # noqa: N806

    max_length = 100
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # noqa: S106
    tokenizer.fit_on_texts(X_train_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_texts), maxlen=max_length)  # noqa: N806
    y_train = np.array(list(y_train))

    X_val = pad_sequences(tokenizer.texts_to_sequences(X_val_texts), maxlen=max_length)  # noqa: N806
    y_val = np.array(list(y_val))

    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_texts), maxlen=max_length)  # noqa: N806
    y_test = np.array(list(y_test))

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    if not Path.exists(GLOVE_PATH):
        with zipfile.ZipFile(GLOVE_ZIP_PATH, "r") as zip_ref:
            zip_ref.extract("glove.6B.50d.txt", GLOVE_PATH.parent)

    glove = load_glove(str(GLOVE_PATH))
    embedding_dim = 50
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    rng = np.random.default_rng(SEED)
    for word, i in tokenizer.word_index.items():
        embedding_matrix[i] = glove.get(word, rng.normal(size=(embedding_dim,)))

    model = build_lstm_model(vocab_size, embedding_dim, embedding_matrix)

    print(">>> Training LSTM Model")
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6),
        ],
        verbose=2,
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    classification = classification_report(y_test, y_pred)

    params = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "epochs": 10,
        "batch_size": 128,
        "early_stopping_patience": 3,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 2,
        "optimizer": model.optimizer.__class__.__name__,
        "learning_rate": float(model.optimizer.learning_rate.numpy()),
        "loss": model.loss,
        "metrics": model.metrics_names,
    }

    return params, classification, 1


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    tf.random.set_seed(SEED)
    start = time.perf_counter()
    reports = {}

    datasets = [
        ("Amazon Reviews", "project/src/data/amazon.csv"),
        ("Airline Tweets", "project/src/data/tweets.csv"),
        ("YouTube Comments", "project/src/data/youtube.csv"),
    ]

    print(">>> Running LSTM Sentiment Analysis with GloVe Embeddings")
    for name, path in datasets:
        print(f"\n>>> Loading dataset: {name}")
        data = load_dataset(path)
        params, classification, num_fits = run_sentiment_analysis(data)
        reports[name] = {"params": params, "classification": classification, "num_fits": num_fits}

    print("\n>>> Final Statistics")
    print_reports(reports)

    sum_all_fits = sum(report["num_fits"] for report in reports.values())
    print(f"Average F1 Score: {sum([float(report['classification'].split()[-2]) for report in reports.values()]) / len(reports):.4f}")
    print(f"Total Time: {time.perf_counter() - start:.2f} seconds for {sum_all_fits} fits")
