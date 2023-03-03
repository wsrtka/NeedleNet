"""Script for model training."""
# pylint: disable=import-error

import torch
import torchmetrics

from torchaudio import load, transforms, pipelines
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import DatasetFolder

from model import Average
from engine import train_model


DATASET_PATH = "./data"
BATCH_SIZE = 32
EPOCHS = 3
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
    ds = DatasetFolder(
        DATASET_PATH,
        loader=lambda filepath: load(filepath)[0],
        extensions=("wav"),
        transform=transforms.Resample(44500, 16000),
    )
    train_ds, test_ds = random_split(ds, (0.8, 0.2))
    # prepare dataloader
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    # get and prepare model
    bundle = pipelines.WAV2VEC2_ASR_BASE_10M
    model = bundle.get_model()
    model.aux = nn.Sequential(Average(axis=-2), nn.Linear(768, 5))
    # prepare training
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=5)
    optimizer = optim.SGD(model.parameters(), lr=LERANING_RATE)
    train_model(model, EPOCHS, loss_fn, acc_fn, train_dl, test_dl, optimizer)
