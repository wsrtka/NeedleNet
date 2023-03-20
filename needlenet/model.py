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


# pylint: disable=too-few-public-methods, invalid-name,too-many-instance-attributes
class NeedleNet(nn.Module):
    """All convolutional CNN for audio file classification."""

    def __init__(self, num_classes):
        super().__init__()
        feature_extractor = []
        classifier_head = []

        # first convolution block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        feature_extractor += [self.conv1, self.relu1, self.bn1]

        # second convolution block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        feature_extractor += [self.conv2, self.relu2, self.bn2]

        # third convolutional block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        feature_extractor += [self.conv3, self.relu3, self.bn3]

        # fourth convolutional block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        feature_extractor += [self.conv4, self.relu4, self.bn4]

        # linear classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(64, num_classes)

        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.classifier_head = nn.Sequential(*classifier_head)

    def forward(self, x):
        """Predict audio file class."""
        output = self.feature_extractor(x)
        output = self.classifier_head(output)
        return output


if __name__ == "__main__":
    net = NeedleNet(5)
    print(net)
