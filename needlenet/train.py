"""Script for model training."""
# pylint: disable=import-error

import pprint
import json

from argparse import ArgumentParser
from collections import Counter

import torch
import torchmetrics

from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset

from model import NeedleNetV1
from engine import train_model
from data import CWTDataset, file_length_split, AudioDatasetV1


PARSER = ArgumentParser(
    prog="NeedleNet Training",
    description="Training script for audio classification tasks associated with the Vibronav project.",
)
PARSER.add_argument("-cv", action="store_true")


CWT_DATASET_PATH = "./cwt_dataset"
CWT_DENOISED_DATASET_PATH = "./cwt_denoised_dataset"
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-2
NUM_CLASSES = 5
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
METRICS = [
    torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
    torchmetrics.classification.MulticlassF1Score(NUM_CLASSES),
    torchmetrics.classification.MulticlassPrecision(NUM_CLASSES),
]
CV_FOLDS = 5


if __name__ == "__main__":
    # set manual seed for experiments
    torch.manual_seed(0)

    # go through both remaining datasets
    for ds_path in [CWT_DATASET_PATH, CWT_DENOISED_DATASET_PATH]:
        # initialize dataset
        ds = CWTDataset(root=ds_path)
        ys = []
        for _, y in ds:
            ys.append(y)
        class_counts = Counter(ys)
        print(f"Classes in dataset:")
        for class_name, count in class_counts.items():
            print(
                f"\tClass {class_name} (idx {ds.idx_to_class[class_name]}): {count} instances."
            )

        # initialize loss function
        loss_fn = nn.CrossEntropyLoss()

        # this is for k fold crossval
        ratios = [1 / CV_FOLDS for _ in range(CV_FOLDS)]
        ratios[0] += 1 - sum(ratios)
        subsets = file_length_split(ds, ratios)
        report = {str(metric): {"best": 0, "average": []} for metric in METRICS}
        for val_idx in range(CV_FOLDS):
            iter_subsets = subsets.copy()
            test_subset = iter_subsets.pop(val_idx)
            train_subset = ConcatDataset(iter_subsets)
            train_dl = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            test_dl = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)
            model = NeedleNetV1(NUM_CLASSES)
            model = model.to(DEVICE)
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
            result = train_model(
                model,
                EPOCHS,
                loss_fn,
                train_dl,
                test_dl,
                optimizer,
                DEVICE,
                NUM_CLASSES,
                METRICS,
            )
            for metric_name, values in result.items():
                print(metric_name)
                print(f"Final value: {values[-1]}, best: {max(values)}")
                report[metric_name]["average"].append(values[-1])
                if max(values) > report[metric_name]["best"]:
                    report[metric_name]["best"] = max(values)
        for metric in report.keys():
            report[metric]["average"] = sum(report[metric]["average"]) / len(
                report[metric]["average"]
            )
            report[metric]["average"] = round(report[metric]["average"], 5)
            report[metric]["best"] = round(report[metric]["best"], 5)

        pprint.pprint(report)
        with open(f"{ds_path}_nn1.json", "w") as file:
            json.dump(report, file)
        report = {str(metric): {"best": 0, "average": []} for metric in METRICS}

        for val_idx in range(CV_FOLDS):
            iter_subsets = subsets.copy()
            test_subset = iter_subsets.pop(val_idx)
            train_subset = ConcatDataset(iter_subsets)
            train_dl = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            test_dl = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet34", pretrained=True
            )
            model.fc = torch.nn.Linear(in_features=512, out_features=NUM_CLASSES)
            model = model.to(DEVICE)
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
            result = train_model(
                model,
                EPOCHS,
                loss_fn,
                train_dl,
                test_dl,
                optimizer,
                DEVICE,
                NUM_CLASSES,
                METRICS,
            )
            for metric_name, values in result.items():
                print(metric_name)
                print(f"Final value: {values[-1]}, best: {max(values)}")
                report[metric_name]["average"].append(values[-1])
                if max(values) > report[metric_name]["best"]:
                    report[metric_name]["best"] = max(values)
        for metric in report.keys():
            report[metric]["average"] = sum(report[metric]["average"]) / len(
                report[metric]["average"]
            )
            report[metric]["average"] = round(report[metric]["average"], 5)
            report[metric]["best"] = round(report[metric]["best"], 5)
        pprint.pprint(report)
        with open(f"{ds_path}_resnet.json", "w") as file:
            json.dump(report, file)
