"""File containing training functions."""
# pylint: disable=import-error

import os
from datetime import date

import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


# pylint: disable=invalid-name,too-many-arguments,too-many-locals
def train_model(model, epochs, loss_fn, train_dl, test_dl, optimizer, device, metrics):
    """Function to train model."""
    # set up metrics logging
    model_class = type(model).__name__
    os.makedirs(f"./runs/{model_class}", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{model_class}/{date.today()}/e150Magdeburg")
    # dictionary for logging metrics across epochs
    metrics_values = {str(k): [] for k in metrics}

    # training
    for epoch in tqdm(range(epochs)):
        print(f"Epoch #{epoch}/{epochs}")
        # set model mode to training
        model.train()
        train_loss = 0
        for _, (X, y) in enumerate(train_dl):
            # forward
            X = X.to(device)
            y_pred = model(X)
            # backpropagation
            y_pred = y_pred.to("cpu")
            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # average out train loss
        train_loss /= len(train_dl)
        # setup test metrics
        test_loss = 0
        test_metrics = {str(k): 0 for k in metrics}

        # set mode to evaluation mode
        model.eval()
        with torch.inference_mode():
            for X, y in test_dl:
                X = X.to(device)
                test_pred = model(X)
                test_pred = test_pred.to("cpu")
                # record averaged metrics
                test_loss += loss_fn(test_pred, y) / len(test_dl)
                for metric in metrics:
                    test_metrics[str(metric)] += metric(test_pred, y) / len(test_dl)

        # record metrics for epoch
        for metric_name, metric_val in test_metrics.items():
            metrics_values[metric_name].append(metric_val.item())

        # add metrics to tensorboard
        writer.add_scalars(
            "Test metrics",
            test_metrics,
            epoch,
        )
        writer.add_scalars(
            "Loss", {"train_loss": train_loss, "test_loss": test_loss}, epoch
        )

        print(f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")

        os.makedirs(f"./models/{model_class}", exist_ok=True)
        torch.save(model.state_dict(), f"./models/v2/model{date.today()}.pt")

    writer.close()
    return metrics_values
