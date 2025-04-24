import random

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from utils import (
    SEED,
    load_dataset,
    preprocess_text,
    print_reports,
    split_data,
    tokenize,
)


def run_sentiment_analysis(data: list) -> tuple[dict, str]:
    print(">>> Vectorizing Dataset")

    train_data, val_data, test_data = split_data(data)
    test_data.extend(val_data)

    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, preprocessor=preprocess_text, max_features=5000)
    train_texts, train_labels = zip(*train_data)
    X_train = vectorizer.fit_transform(train_texts)  # noqa: N806

    print(">>> Training Models with Hyperparameter Tuning")
    param_grid = {                             # Tested hyperparameters
        "C": [0.1, 0.5, 1, 1.5, 2, 2.5, 3.0],  # [1e-6, 1e-4, 1e-2, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 10, 25, 50, 100]
        "penalty": ["l2"],                     # ['l1', 'l2', 'elasticnet', None]
        "solver": ["saga"],                    # ['saga', 'lbfgs', 'liblinear']
        "l1_ratio": [None],                    # [0, 0.3, 0.5, 0.7, 1] note: only used with 'elasticnet' penalty
        "class_weight": [None],                # [None, 'balanced']
        "max_iter": [100_000],                 # np.linspace(10, 100000, 10).astype(int) note: seems mostly converged at 15
        "tol": [1e-1, 1e-2, 1e-3, 1e-4],       # [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    }

    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=SEED, warm_start=True),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, train_labels)

    best_model = grid_search.best_estimator_
    best_params = best_model.get_params()
    test_texts, test_labels = zip(*test_data)
    X_test = vectorizer.transform(test_texts)  # noqa: N806
    test_pred = best_model.predict(X_test)

    return best_params, classification_report(test_labels, test_pred)


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    reports = {}

    datasets = [
        ("Airline Tweets", "project/src/data/tweets.csv"),
        ("YouTube Comments", "project/src/data/youtube.csv"),
        ("Amazon Reviews", "project/src/data/amazon.csv"),
    ]

    for name, path in datasets:
        print(f"\n>>> Loading dataset: {name}")
        data = load_dataset(path)
        params, classification = run_sentiment_analysis(data)
        reports[name] = {"params": params, "classification": classification}

    print("\n>>> Final Statistics")
    print_reports(reports)

    print("\n>>> Finished")
    print(f"Average F1 Score: {sum([float(report['classification'].split()[-2]) for report in reports.values()]) / len(reports):.4f}")
