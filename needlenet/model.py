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


class NeedleNet(nn.Module):
    """All convolutional CNN for audio file classification."""
