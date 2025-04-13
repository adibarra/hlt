from __future__ import annotations

import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from tqdm import tqdm
from utils import load_data, process_input


# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
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
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        output, hidden = self.rnn(inputs)

        # [to fill] obtain output layer representations
        final_hidden = hidden[-1]

        # [to fill] sum over output
        summed_output = torch.sum(output, dim=0)  # Sum over the sequence length

        # [to fill] obtain probability dist.
        logits = self.W(summed_output)
        predicted_vector = self.softmax(logits)

        return predicted_vector

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument("--do_train", action="store_true")
    args: Namespace = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)  # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model: RNN = RNN(50, args.hidden_dim)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    with (Path(__file__).parent / "word_embedding.pkl").open("rb") as f:
        word_embedding: dict = pickle.load(f)

    epoch: int = 0
    last_train_accuracy: float = 0
    last_validation_accuracy: float = 0
    patience: int = 2
    patience_counter: int = 0
    stopping_condition: bool = False

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print(f"Training started for epoch {epoch + 1}")
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)
        loss_total = 0
        loss_count = 0

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss: Tensor = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                vectors_tensor: Tensor = process_input(input_words, word_embedding)
                output: Tensor = model(vectors_tensor)

                # Get loss
                example_loss: Tensor = model.compute_loss(output.view(1, -1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label: Tensor = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.item()
            loss_count += 1
            loss.backward()
            optimizer.step()

        print(loss_total / loss_count)
        print(f"Training completed for epoch {epoch + 1}")
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")
        training_accuracy: float = correct / total

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print(f"Validation started for epoch {epoch + 1}")

        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data):
                vectors_tensor: Tensor = process_input(input_words, word_embedding)
                output: Tensor = model(vectors_tensor)
                predicted_label: Tensor = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                # print(predicted_label, gold_label)
        print(f"Validation completed for epoch {epoch + 1}")
        print(f"Validation accuracy for epoch {epoch + 1}: {correct / total}")
        validation_accuracy: float = correct / total

        if validation_accuracy > last_validation_accuracy:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            stopping_condition = True
            print("Early stopping triggered due to no improvement in validation accuracy!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy

        epoch += 1
        scheduler.step()

    # You may find it beneficial to keep track of training accuracy or training loss;
    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
