"""Script for model training."""
# pylint: disable=import-error

import torch
import torchmetrics

from torch import nn, optim
from torch.utils.data import random_split, DataLoader

from model import NeedleNet
from engine import train_model
from data import AudioDataset


DATASET_PATH = "./new_data"
BATCH_SIZE = 64
EPOCHS = 100
LERANING_RATE = 1e-3
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if __name__ == "__main__":
    # set manual seed for experiments
    torch.manual_seed(0)
    # get dataset
    ds = AudioDataset(root=DATASET_PATH, extensions=("wav"), sample_rate=22050)
    train_ds, test_ds = random_split(ds, (0.8, 0.2))
    # prepare dataloader
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    # create model
    model = NeedleNet(num_classes=5)
    model = model.to(DEVICE)
    # prepare training
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=5)
    optimizer = optim.SGD(model.parameters(), lr=LERANING_RATE)
    train_model(model, EPOCHS, loss_fn, acc_fn, train_dl, test_dl, optimizer, DEVICE)
