"""Script for model training."""
# pylint: disable=import-error

import torch

from torch import nn, optim
from torch.utils.data import DataLoader

from model import NeedleNetV2
from engine import train_model
from data import CWTDataset, file_length_split


DATASET_PATH = "./cwt_processed"
BATCH_SIZE = 32
EPOCHS = 150
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
    ds = CWTDataset(root=DATASET_PATH)
    train_ds, test_ds = file_length_split(ds, 0.8)
    # prepare dataloader
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    # create model
    model = NeedleNetV2(num_classes=4)
    model = model.to(DEVICE)
    # prepare training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LERANING_RATE, momentum=0.9)
    train_model(model, EPOCHS, loss_fn, train_dl, test_dl, optimizer, DEVICE)
