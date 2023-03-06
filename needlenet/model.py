"""Module containing audio classification model."""
# pylint: disable=import-error

import torch

from torch import nn


# pylint: disable=too-few-public-methods, invalid-name
class Average(nn.Module):
    """Module to get average of elements along specified axis."""

    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        """Average input along specified axis."""
        output = x.mean(dim=self.axis)
        output = torch.squeeze(output, self.axis)
        return output


# pylint: disable=too-few-public-methods, invalid-name
class NeedleNet(nn.Module):
    """All convolutional CNN for audio file classification."""

    def __init__(self, num_classes):
        super().__init__()
        conv1 = nn.Conv1d(1, 256, 1)
        conv2 = nn.Conv1d(256, 256, 3, 2)
        global_pool = nn.AdaptiveAvgPool3d((1, 1, 256))
        self.feature_extractor = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            global_pool,
        )
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        """Predict audio file class."""
        output = self.feature_extractor(x)
        output = self.linear(output)
        return output


if __name__ == "__main__":
    net = NeedleNet(5)
    print(net)
