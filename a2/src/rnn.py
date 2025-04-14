from __future__ import annotations

import pickle
import platform
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils import load_data, process_input


class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 5) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        dropped = self.dropout(hidden_combined)
        return self.fc(dropped)

    def compute_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, labels)

    def train_epoch(
        self,
        train_data: list[tuple[list[str], int]],
        word_embedding: dict[str, torch.Tensor],
        optimizer: optim.Optimizer,
        minibatch_size: int = 16,
    ) -> tuple[float, float]:
        self.train()
        correct = 0
        total = 0
        epoch_loss = 0.0
        N = len(train_data)  # noqa: N806

        random.shuffle(train_data)
        for minibatch_index in tqdm(range(0, N, minibatch_size), desc="Training"):
            optimizer.zero_grad()

            input_data, labels = [], []
            for example_index in range(minibatch_index, min(minibatch_index + minibatch_size, N)):
                input_words, gold_label = train_data[example_index]
                vectors_tensor = process_input(input_words, word_embedding).to(device)
                input_data.append(vectors_tensor)
                labels.append(gold_label)

            input_data = pad_sequence(input_data, batch_first=True, padding_value=0).to(device)
            labels = torch.tensor(labels).to(device)

            output = self(input_data)
            loss = self.compute_loss(output, labels)
            epoch_loss += loss.item() * labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return correct / total, epoch_loss / total

    def evaluate(self, val_data: list[tuple[list[str], int]], word_embedding: dict[str, torch.Tensor]) -> tuple[float, float]:
        self.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        input_data, labels = [], []
        for input_words, gold_label in val_data:
            vectors_tensor = process_input(input_words, word_embedding).to(device)
            input_data.append(vectors_tensor)
            labels.append(gold_label)

        input_data = pad_sequence(input_data, batch_first=True, padding_value=0).to(device)
        labels = torch.tensor(labels).to(device)

        with torch.no_grad():
            output = self(input_data)
            loss = self.compute_loss(output, labels)
            val_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return correct / total, val_loss / total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd",  "--hidden-dim",  type=int,  required=True,        help="hidden dimension size")
    parser.add_argument("-e",   "--epochs",      type=int,  required=True,        help="number of epochs")
    parser.add_argument("-p",   "--patience",    type=int,  default=5,            help="patience for early stopping")
    parser.add_argument(        "--train-data",  type=str,  required=True,        help="path to training data")
    parser.add_argument(        "--val-data",    type=str,  required=True,        help="path to validation data")
    parser.add_argument(        "--test-data",   type=str,  default="to fill",    help="path to test data")
    parser.add_argument(        "--do-train",               action="store_true",  help="whether to train the model")
    args = parser.parse_args()

    print(">>> Setting up environment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor
    print(f"Using device: {device_name} {'(GPU)' if device.type == 'cuda' else '(CPU)'}")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    random.seed(42)

    print(">>> Loading data")
    train_data, val_data = load_data(args.train_data, args.val_data)
    with Path.open(Path(__file__).parent / "word_embedding.pkl", "rb") as f:
        word_embedding = pickle.load(f)  # noqa: S301

    print(">>> Building model")
    model = RNN(input_dim=50, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.75)

    print("=" * 125)
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Scheduler: {scheduler.__class__.__name__}")
    print(f"Scheduler step size: {scheduler.step_size}")
    print(f"Scheduler gamma: {scheduler.gamma}")
    print(f"Early stopping patience: {args.patience}")
    print("=" * 125)

    patience_counter = 0
    stopping_condition = False
    best = {"epoch": 0, "train_acc": 0, "val_acc": 0, "train_loss": float("inf"), "val_loss": float("inf")}

    print(">>> Training")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_acc, train_loss = model.train_epoch(train_data, word_embedding, optimizer)
        val_acc, val_loss = model.evaluate(val_data, word_embedding)
        print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_acc > best["val_acc"]:
            best = { "epoch": epoch + 1, "train_acc": train_acc, "val_acc": val_acc, "train_loss": train_loss, "val_loss": val_loss}
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No validation accuracy improvement. Patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            stopping_condition = True
            print("Early stopping triggered")
            break

        scheduler.step()

    print("\n>>> Reporting Results")
    print("=" * 125)
    print(f"Best Epoch: {best['epoch']}")
    print(f"Train Accuracy: {best['train_acc']:.4f}")
    print(f"Val Accuracy: {best['val_acc']:.4f}")
    print(f"Train Loss: {best['train_loss']:.4f}")
    print(f"Val Loss: {best['val_loss']:.4f}")
    print("=" * 125)
