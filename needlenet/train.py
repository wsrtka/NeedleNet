"""Script for model training."""
# pylint: disable=import-error

import torch

from torchaudio import load, transforms, pipelines
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import DatasetFolder

from model import Average


DATASET_PATH = "./data"
BATCH_SIZE = 64


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
