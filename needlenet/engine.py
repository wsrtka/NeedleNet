"""File containing training functions."""
# pylint: disable=import-error

import torch

from tqdm.auto import tqdm


# pylint: disable=invalid-name,too-many-arguments,too-many-locals
def train_model(model, epochs, loss_fn, acc_fn, train_dl, test_dl, optimizer):
    """Function to train model."""
    for epoch in tqdm(range(epochs)):
        print(f"Epoch #{epoch}/{epochs}")
        train_loss = 0
        model.train()
        for batch, (X, y) in enumerate(train_dl):
            print(X.shape, y.shape)
            y_pred, _ = model(X)
            print(y_pred, X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"{batch * len(X) / len(train_dl.dataset)} samples, loss: {loss}")

        train_loss /= len(train_dl)
        test_loss = 0
        test_acc = 0

        model.eval()
        with torch.inference_mode():
            for X, y in test_dl:
                test_pred, _ = model(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += acc_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(test_dl)
            test_acc /= len(test_dl)

        print(
            f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
        )
