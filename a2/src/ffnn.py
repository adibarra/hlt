from __future__ import annotations

import platform
import random
from argparse import ArgumentParser

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import convert_to_vector_representation, load_data, make_indices, make_vocab


class FFNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 5) -> None:
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        return self.ffnn(input_vector)

    def compute_loss(self, predicted_vector: torch.Tensor, gold_label: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predicted_vector, gold_label)

    def train_epoch(
        self,
        train_data: list[tuple[torch.Tensor, int]],
        optimizer: optim.Optimizer,
        minibatch_size: int = 16,
    ) -> tuple[float, float]:
        self.train()
        correct, total, epoch_loss = 0, 0, 0.0
        N = len(train_data)  # noqa: N806

        random.shuffle(train_data)
        for minibatch_index in tqdm(range(0, N, minibatch_size), desc="Training"):
            optimizer.zero_grad()

            batch = train_data[minibatch_index : min(minibatch_index + minibatch_size, N)]
            input_batch = torch.stack([ex[0] for ex in batch]).to(device)
            label_batch = torch.tensor([ex[1] for ex in batch]).to(device)

            output = self(input_batch)
            loss = self.compute_loss(output, label_batch)
            epoch_loss += loss.item() * label_batch.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label_batch).sum().item()
            total += label_batch.size(0)

        return correct / total, epoch_loss / total

    def evaluate(
        self,
        val_data: list[tuple[torch.Tensor, int]],
        minibatch_size: int = 16,
    ) -> tuple[float, float]:
        self.eval()
        correct, total, val_loss = 0, 0, 0.0
        N = len(val_data)  # noqa: N806

        with torch.no_grad():
            for minibatch_index in tqdm(range(0, N, minibatch_size), desc="Validation"):
                batch = val_data[minibatch_index : min(minibatch_index + minibatch_size, N)]
                input_batch = torch.stack([ex[0] for ex in batch]).to(device)
                label_batch = torch.tensor([ex[1] for ex in batch]).to(device)

                output = self(input_batch)
                loss = self.compute_loss(output, label_batch)
                val_loss += loss.item() * label_batch.size(0)

                _, predicted = torch.max(output, 1)
                correct += (predicted == label_batch).sum().item()
                total += label_batch.size(0)

        return correct / total, val_loss / total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd",  "--hidden-dim",  type=int,  default=128,                       help="hidden dimension size")
    parser.add_argument("-e",   "--epochs",      type=int,  default=30,                        help="number of epochs")
    parser.add_argument("-p",   "--patience",    type=int,  default=5,                         help="patience for early stopping")
    parser.add_argument(        "--train-data",  type=str,  default="a2/src/data/train.json",  help="path to training data")
    parser.add_argument(        "--val-data",    type=str,  default="a2/src/data/val.json",    help="path to validation data")
    parser.add_argument(        "--test-data",   type=str,  default="a2/src/data/test.json",   help="path to test data")
    parser.add_argument(        "--do-train",               action="store_true",               help="whether to train the model")
    args = parser.parse_args()

    print(">>> Setting up environment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    print(f"Using device: {device_name} {'(GPU)' if device.type == 'cuda' else '(CPU)'}")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    random.seed(42)

    print(">>> Loading data")
    train_raw, val_raw = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_raw)
    vocab, word2index, index2word = make_indices(vocab)

    print(">>> Vectorizing data")
    train_data = convert_to_vector_representation(train_raw, word2index)
    val_data = convert_to_vector_representation(val_raw, word2index)

    print(">>> Building model")
    model = FFNN(input_dim=len(vocab), hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", min_lr=1e-6, factor=0.5, patience=3)

    print("=" * 50)
    print(f"{'Hidden dimension':<25s}: {args.hidden_dim}")
    print(f"{'Epochs':<25s}: {args.epochs}")
    print(f"{'Optimizer':<25s}: {optimizer.__class__.__name__}")
    print(f"{'Optimizer learning rate':<25s}: {optimizer.param_groups[0]['lr']}")
    print(f"{'Optimizer weight decay':<25s}: {optimizer.param_groups[0]['weight_decay']}")
    print(f"{'Scheduler':<25s}: {scheduler.__class__.__name__}")
    print(f"{'Scheduler mode':<25s}: {scheduler.mode}")
    print(f"{'Scheduler factor':<25s}: {scheduler.factor}")
    print(f"{'Scheduler patience':<25s}: {scheduler.patience}")
    print(f"{'Early stopping patience':<25s}: {args.patience}")
    print("=" * 50)

    patience_counter = 0
    best = {"epoch": 0, "train_acc": 0, "val_acc": 0, "comb_acc": 0, "train_loss": float("inf"), "val_loss": float("inf")}

    print(">>> Training")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_acc, train_loss = model.train_epoch(train_data, optimizer)
        val_acc, val_loss = model.evaluate(val_data)
        comb_acc = 0.95 * val_acc + 0.05 * train_acc
        print(f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, comb_acc: {comb_acc:.4f}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if comb_acc > best["comb_acc"]:
            best = { "epoch": epoch + 1, "train_acc": train_acc, "comb_acc": comb_acc, "val_acc": val_acc, "train_loss": train_loss, "val_loss": val_loss}
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No comb_acc improvement. Patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

        scheduler.step(comb_acc)

    print("\n>>> Reporting Results")
    print("=" * 50)
    print(f"{'best epoch':<12s}: {best['epoch']}")
    print(f"{'train_acc':<12s}: {best['train_acc']:.4f}")
    print(f"{'val_acc':<12s}: {best['val_acc']:.4f}")
    print(f"{'train_loss':<12s}: {best['train_loss']:.4f}")
    print(f"{'val_loss':<12s}: {best['val_loss']:.4f}")
    print("=" * 50)
