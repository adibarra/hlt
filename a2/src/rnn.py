from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils import load_data, process_input


class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 5) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])

    def compute_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, labels)

    def train_epoch(
        self,
        train_data: list[tuple[list[str], int]],
        word_embedding: dict[str, torch.Tensor],
        optimizer: optim.Optimizer,
        minibatch_size: int = 16,
    ) -> float:
        self.train()
        correct = 0
        total = 0
        N = len(train_data)  # noqa: N806

        for minibatch_index in tqdm(range(0, N, minibatch_size), desc="Training"):
            optimizer.zero_grad()

            input_data, labels = [], []
            for example_index in range(minibatch_index, min(minibatch_index + minibatch_size, N)):
                input_words, gold_label = train_data[example_index]
                vectors_tensor = process_input(input_words, word_embedding)
                input_data.append(vectors_tensor)
                labels.append(gold_label)

            input_data = pad_sequence(input_data, batch_first=True, padding_value=0)
            input_data = input_data.view(input_data.size(0), input_data.size(1), -1)
            labels = torch.tensor(labels)

            output = self(input_data)
            loss = self.compute_loss(output, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return correct / total

    def evaluate(
        self,
        val_data: list[tuple[list[str], int]],
        word_embedding: dict[str, torch.Tensor],
    ) -> float:
        self.eval()
        correct = 0
        total = 0

        input_data, labels = [], []
        for input_words, gold_label in val_data:
            vectors_tensor = process_input(input_words, word_embedding)
            input_data.append(vectors_tensor)
            labels.append(gold_label)

        input_data = pad_sequence(input_data, batch_first=True, padding_value=0)
        input_data = input_data.view(input_data.size(0), input_data.size(1), -1)
        labels = torch.tensor(labels)

        with torch.no_grad():
            output = self(input_data)
            _, predicted = torch.max(output, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return correct / total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, required=True, help="hidden dimension size")
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("--train-data", required=True, help="path to training data")
    parser.add_argument("--val-data", required=True, help="path to validation data")
    parser.add_argument("--test-data", default="to fill", help="path to test data")
    parser.add_argument("--do-train", action="store_true", help="whether to train the model")
    args = parser.parse_args()

    print(">>> Loading data")
    train_data, val_data = load_data(args.train_data, args.val_data)
    with Path.open(Path(__file__).parent / "word_embedding.pkl", "rb") as f:
        word_embedding = pickle.load(f)  # noqa: S301

    print(">>> Building model")
    model = RNN(input_dim=50, hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    patience = 3
    patience_counter = 0
    stopping_condition = False

    best = {"epoch": 0, "train_acc": 0, "val_acc": 0}
    last = {"epoch": 0, "train_acc": 0, "val_acc": 0}

    print(">>> Training")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_acc = model.train_epoch(train_data, word_embedding, optimizer)
        val_acc = model.evaluate(val_data, word_embedding)
        print(f"Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > best["val_acc"]:
            best["epoch"] = epoch + 1
            best["train_acc"] = train_acc
            best["val_acc"] = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No validation accuracy improvement. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            stopping_condition = True
            print("Early stopping triggered")
            break

        last["epoch"] = epoch + 1
        last["train_acc"] = train_acc
        last["val_acc"] = val_acc

    print(f"Best validation accuracy: {best['val_acc']:.4f} at epoch {best['epoch']}")
