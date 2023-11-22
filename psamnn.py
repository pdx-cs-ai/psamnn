# Neural Net for binary CSV data — tuned for PSAM
# Bart Massey 2023
# Taken mostly from https://pytorch.org/tutorials/beginner/basics

import argparse, csv, sys

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--learning-rate", type=float, default=0.005)
ap.add_argument("-e", "--epochs", type=int)
ap.add_argument("-b", "--batch_size", type=int, default=1)
ap.add_argument("-s", "--skip", type=int, default=1)
ap.add_argument("-r", "--report-interval", type=int, default=0)
ap.add_argument("-q", "--quiet", action="store_true")
ap.add_argument("-d", "--acc-delta", type=float, default=0.0001)
ap.add_argument("--place")
ap.add_argument("csvfile")
args = ap.parse_args()

import numpy as np

import torch
from torch import nn
from torch.utils import data

device = "cpu"
if args.place == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
elif args.place:
    device = args.place

def eprint(*argv, **pargv):
    if args.quiet:
        return
    pargv["file"]=sys.stderr
    print(*argv, **pargv)
    

eprint(f"Using {device} device")

def read_csv(instances_file, skip=args.skip, shuffle=True):
    reader = csv.reader(open(instances_file, "r"))
    instances = np.array([[float(x) for x in row[skip:]] for row in reader], dtype=np.float32)
    if shuffle:
        np.random.shuffle(instances)
    return torch.from_numpy(instances).to(device)


class CustomCSVDataset(data.Dataset):
    def __init__(self, instances, transform=None, target_transform=None):
        super().__init__()
        self.instances = instances
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        row = self.instances[idx]
        label = row[:1]
        features = row[1:]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label

csvdata = read_csv(args.csvfile)

test_fraction = 0.2
ncsvdata = len(csvdata)
ntrain = int(ncsvdata * test_fraction)
train_data = CustomCSVDataset(csvdata[:ntrain])
test_data = CustomCSVDataset(csvdata[ntrain:])

train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
test_dataloader = data.DataLoader(test_data, batch_size=1)

class NeuralNetwork(nn.Module):
    def __init__(self, train_data):
        super().__init__()
        self.flatten = nn.Flatten()
        w = len(train_data[0][0])
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork(train_data).to(device)
eprint(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer, report=args.report_interval):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if report > 0 and batch % report == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            eprint(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).sum().item()

    test_loss /= num_batches
    correct /= size
    eprint(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


eprint("Training:")
t = 0
old_acc = None
acc_delta = args.acc_delta
while True:
    eprint(f"Epoch {t+1}")
    eprint("-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    acc = test_loop(test_dataloader, model, loss_fn)
    if args.epochs and t >= args.epochs:
        break
    if old_acc and acc < old_acc + acc_delta:
        break
    old_acc = acc
    t += 1
eprint("Done!")
