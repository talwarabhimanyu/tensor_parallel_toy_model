import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import numpy as np
from model import ToyModel

class ToyDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]

def get_dataloaders(data_dir, batch_size):
    X_train = np.load(os.path.join(data_dir, "xtrain.npy"))
    y_train = np.load(os.path.join(data_dir, "ytrain.npy"))
    X_valid = np.load(os.path.join(data_dir, "xvalid.npy"))
    y_valid = np.load(os.path.join(data_dir, "yvalid.npy"))
    train_loader = DataLoader(dataset=ToyDataset(
                                            X_train,
                                            y_train
                                    ),
                            batch_size=batch_size,
                            shuffle=False)
    valid_loader = DataLoader(dataset=ToyDataset(
                                            X_valid,
                                            y_valid
                                    ),
                            batch_size=batch_size,
                            shuffle=False)
    return train_loader, valid_loader

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--init_lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--num_iterations", type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--update_freq", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = ToyModel(input_size=args.input_size,
                    hidden_size=32*args.input_size)
    model.to(device);

    optimizer = torch.optim.SGD(model.parameters(),
                    lr=args.init_lr,
                    momentum=args.momentum
                )
    loss_fn = nn.MSELoss()

    train_loader, valid_loader = get_dataloaders(data_dir=args.data_dir,
                                    batch_size=args.batch_size,
                                    )

    model.train()
    running_train_loss = 0.
    for i in range(args.num_iterations):
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs[:,0], targets)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        if i % args.update_freq == args.update_freq-1:
            print(f"Iter {i+1} Train Loss {running_train_loss:.3f}")
            running_train_loss = 0.

train()
