import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
from torch.optim.lr_scheduler import StepLR


unk = "<UNK>"
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h) -> None:
        super().__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        h = self.activation(self.W1(input_vector))

        # [to fill] obtain output layer representation
        z = self.W2(h)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(z)

        return predicted_vector


# Returns:
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


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
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print(">>> Vectorizing data")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(val_data, word2index)

    print(">>> Building model")
    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
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
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Training started for epoch {epoch + 1}")
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print(f"Training completed for epoch {epoch + 1}")
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")
        print(f"Training time for this epoch: {time.time() - start_time}")


        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Validation started for epoch {epoch + 1}")
        minibatch_size = 16
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
        print(f"Validation completed for epoch {epoch + 1}")
        print(f"Validation accuracy for epoch {epoch + 1}: {correct / total}")
        print(f"Validation time for this epoch: {time.time() - start_time}")

    # write out to results/test.out
