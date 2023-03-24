"""File containing training functions."""
# pylint: disable=import-error

import torch
import torchmetrics

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


# pylint: disable=invalid-name,too-many-arguments,too-many-locals
def train_model(model, epochs, loss_fn, train_dl, test_dl, optimizer, device):
    """Function to train model."""
    writer = SummaryWriter()

    acc_fn = torchmetrics.classification.MulticlassAccuracy(5)
    f1_fn = torchmetrics.classification.MulticlassF1Score(5)
    recall_fn = torchmetrics.classification.MulticlassRecall(5)
    precision_fn = torchmetrics.classification.MulticlassPrecision(5)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch #{epoch}/{epochs}")
        train_loss = 0
        model.train()
        for _, (X, y) in enumerate(train_dl):
            X = X.to(device)
            y_pred = model(X)
            y_pred = y_pred.to("cpu")
            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch % 10 == 0:
            #     print(
            #         f"{round(100 * batch * len(X) / len(train_dl.dataset), 2)}%, loss: {loss}"
            #     )

        train_loss /= len(train_dl)
        test_loss = 0
        test_acc = 0
        test_f1 = 0
        test_recall = 0
        test_precision = 0

        model.eval()
        with torch.inference_mode():
            for X, y in test_dl:
                X = X.to(device)
                test_pred = model(X)
                test_pred = test_pred.to("cpu")
                test_loss += loss_fn(test_pred, y)
                test_acc += acc_fn(test_pred, y)
                test_f1 += f1_fn(test_pred, y)
                test_recall += recall_fn(test_pred, y)
                test_precision += precision_fn(test_pred, y)

            test_loss /= len(test_dl)
            test_acc /= len(test_dl)
            test_f1 /= len(test_dl)
            test_recall /= len(test_dl)
            test_precision /= len(test_dl)
            writer.add_scalars(
                "Test metrics",
                {
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "test_recall": test_recall,
                    "test_precision": test_precision,
                },
                epoch,
            )

        writer.add_scalars(
            "Loss", {"train_loss": train_loss, "test_loss": test_loss}, epoch
        )

        print(
            f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
        )

    writer.close()
