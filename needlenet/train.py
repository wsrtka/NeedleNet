"""Script for model training."""
# pylint: disable=import-error

from datetime import date

import torch

from torch import nn, optim
from torch.utils.data import random_split, DataLoader

from model import NeedleNetV1
from engine import train_model
from data import AudioDatasetV1


DATASET_PATH = "./new_data"
BATCH_SIZE = 32
EPOCHS = 300
LERANING_RATE = 1e-2
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
    ds = AudioDatasetV1(root=DATASET_PATH, extensions=("wav"), sample_rate=22050)
    train_ds, test_ds = random_split(ds, (0.8, 0.2))
    # prepare dataloader
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    # create model
    model = NeedleNetV1(num_classes=5)
    model = model.to(DEVICE)
    # prepare training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LERANING_RATE, momentum=0.9)
    train_model(model, EPOCHS, loss_fn, train_dl, test_dl, optimizer, DEVICE)
    torch.save(model.state_dict(), f"./models/v1/model{date.today()}.pt")
