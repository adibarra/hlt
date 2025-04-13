from __future__ import annotations

import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from tqdm import tqdm
from utils import load_data, process_input


class RNN(nn.Module):
    def __init__(self, input_dim: int, h: int, output_dim: int = 5) -> None:
        super().__init__()
        self.h: int = h
        self.numOfLayer: int = 1
        self.rnn: nn.RNN = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity="tanh")
        self.W: nn.Linear = nn.Linear(h, output_dim)
        self.softmax: nn.LogSoftmax = nn.LogSoftmax(dim=1)
        self.loss: nn.NLLLoss = nn.NLLLoss()

    def compute_loss(self, predicted_vector: Tensor, gold_label: Tensor) -> Tensor:
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs: Tensor) -> Tensor:
        _, hidden = self.rnn(inputs)
        final_hidden = hidden[-1]
        logits = self.W(final_hidden)

        return self.softmax(logits)

    def train_epoch(self, data: list[tuple[list[str], int]], word_embedding: dict, optimizer: optim.Optimizer, minibatch_size: int = 16) -> float:
        self.train()
        correct = 0
        total = 0
        N = len(data)  # noqa: N806

        for minibatch_index in tqdm(range(N // minibatch_size), desc="Training"):
            optimizer.zero_grad()
            loss: Tensor = None

            for example_index in range(minibatch_size):
                input_words, gold_label = data[minibatch_index * minibatch_size + example_index]
                vectors_tensor: Tensor = process_input(input_words, word_embedding)
                output: Tensor = self(vectors_tensor)

                example_loss: Tensor = self.compute_loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label: Tensor = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                loss = example_loss if loss is None else loss + example_loss

            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()

        return correct / total

    def evaluate(self, data: list[tuple[list[str], int]], word_embedding: dict) -> float:
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for input_words, gold_label in tqdm(data, desc="Validating"):
                vectors_tensor: Tensor = process_input(input_words, word_embedding)
                output: Tensor = self(vectors_tensor)
                predicted_label: Tensor = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1

        return correct / total


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument("--do_train", action="store_true")
    args: Namespace = parser.parse_args()

    print(">>> Loading data")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    with (Path(__file__).parent / "word_embedding.pkl").open("rb") as f:
        word_embedding: dict = pickle.load(f)  # noqa: S301

    print(">>> Building model")
    model: RNN = RNN(50, args.hidden_dim)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    last = {"epoch": 0, "train_acc": 0, "val_acc": 0}
    best = {"epoch": 0, "train_acc": 0, "val_acc": 0}
    patience: int = 3
    patience_counter: int = 0
    stopping_condition: bool = False

    print("\n" + "=" * 125)
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Scheduler step size: {scheduler.step_size}")
    print(f"Scheduler gamma: {scheduler.gamma}")
    print(f"Early stopping patience: {patience}")
    print("=" * 125)

    print("\n>>> Training")
    while not stopping_condition:
        print(f"\nEpoch {last["epoch"] + 1}")
        train_acc = model.train_epoch(train_data, word_embedding, optimizer)
        val_acc = model.evaluate(valid_data, word_embedding)
        print(f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > last["val_acc"]:
            patience_counter = 0
            best["epoch"] = last["epoch"] + 1
            best["train_acc"] = train_acc
            best["val_acc"] = val_acc
        else:
            patience_counter += 1
            print(f"No validation accuracy improvement. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            stopping_condition = True
            print("Early stopping triggered")

        last["epoch"] = last["epoch"] + 1
        last["val_acc"] = val_acc
        last["train_acc"] = train_acc

        last["epoch"] += 1
        scheduler.step()

    print("\n>>> Reporting results")
    print(f"Best epoch: {best["epoch"]}, training accuracy: {best["train_acc"]:.4f}, validation accuracy: {best["val_acc"]:.4f}")
