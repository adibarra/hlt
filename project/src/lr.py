import random
import warnings

import torch
from joblib import parallel_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from utils import SEED, load_sentiment_csv, preprocess_text, split_data, tokenize


def run_sentiment_analysis(data: list) -> tuple[float, str]:
    print(">>> Running sentiment analysis")

    train_data, val_data, test_data = split_data(data)
    test_data.extend(val_data)

    param_grid = {
        "vec__max_features": list(range(100, 10001, 100)),
        "lr__max_iter": list(range(1, 11, 1)),
        "lr__solver": ["saga"],
    }

    pipeline = Pipeline([
        ("vec", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, preprocessor=preprocess_text)),
        ("lr", LogisticRegression(random_state=SEED)),
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    train_texts, train_labels = zip(*train_data)

    with parallel_backend("multiprocessing"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        grid_search.fit(train_texts, train_labels)

    print(">>> Reporting Results")
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    test_texts, test_labels = zip(*test_data)
    test_pred = best_model.predict(test_texts)

    return classification_report(test_labels, test_pred)


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    final_statistics = {}

    print(">>> Loading first dataset (YouTube comments)")
    data1 = load_sentiment_csv("project/src/data/youtube.csv")
    class_report1 = run_sentiment_analysis(data1)
    final_statistics["YouTube Comments"] = {
        "classification_report": class_report1,
    }

    print("\n>>> Loading second dataset (Tweets)")
    data2 = load_sentiment_csv("project/src/data/tweets.csv")
    class_report2 = run_sentiment_analysis(data2)
    final_statistics["Tweets"] = {
        "classification_report": class_report2,
    }

    print("\n>>> Final Statistics:")
    for dataset_name, stats in final_statistics.items():
        print(f"{dataset_name}:")
        print("Classification Report:")
        print(stats["classification_report"])
