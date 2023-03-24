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


class ConvBlock(nn.Module):
    """Module implementing a convolution block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        conv_block = []
        conv_block.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        conv_block.append(nn.BatchNorm2d(out_channels))
        conv_block.append(nn.ReLU())
        conv_block.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        conv_block.append(nn.ReLU())
        conv_block.append(nn.Dropout2d(0.2))
        conv_block.append(nn.AvgPool2d(2))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward pass for convolution block."""
        return self.conv_block(x)


# pylint: disable=too-few-public-methods, invalid-name
class NeedleNetV2(nn.Module):
    """All convolutional CNN for audio file classification."""

    def __init__(self, num_classes):
        super().__init__()
        feature_extractor = []
        classifier_head = []

        # first convolution block
        feature_extractor += [ConvBlock(1, 64)]

        # second convolution block
        feature_extractor += [ConvBlock(64, 128)]

        # third convolutional block
        feature_extractor += [ConvBlock(128, 256)]

        # fourth convolutional block
        feature_extractor += [ConvBlock(256, 512)]

        # linear classifier
        classifier_head.append(nn.AdaptiveAvgPool2d(output_size=1))
        classifier_head.append(nn.Flatten())
        classifier_head.append(nn.Dropout1d(0.5))
        classifier_head.append(nn.Linear(512, 128))
        classifier_head.append(nn.PReLU())
        classifier_head.append(nn.BatchNorm1d(128))
        classifier_head.append(nn.Dropout1d(0.5))
        classifier_head.append(nn.Linear(128, num_classes))

        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.classifier_head = nn.Sequential(*classifier_head)

    def forward(self, x):
        """Predict audio file class."""
        output = self.feature_extractor(x)
        output = self.classifier_head(output)
        return output


# pylint: disable=too-few-public-methods, invalid-name,too-many-instance-attributes
class NeedleNetV1(nn.Module):
    """All convolutional CNN for audio file classification."""

    def __init__(self, num_classes):
        super().__init__()
        feature_extractor = []
        classifier_head = []

        # first convolution block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(0.2)
        feature_extractor += [self.conv1, self.relu1, self.dropout1, self.bn1]

        # second convolution block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(0.2)
        feature_extractor += [self.conv2, self.relu2, self.dropout2, self.bn2]

        # third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(0.2)
        self.bn3 = nn.BatchNorm2d(256)
        feature_extractor += [self.conv3, self.relu3, self.dropout3, self.bn3]

        # fourth convolutional block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout2d(0.2)
        self.bn4 = nn.BatchNorm2d(512)
        feature_extractor += [self.conv4, self.relu4, self.dropout4, self.bn4]

        # linear classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fl = nn.Flatten()
        self.lin = nn.Linear(512, num_classes)
        classifier_head += [self.ap, self.fl, self.lin]

        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.classifier_head = nn.Sequential(*classifier_head)

    def forward(self, x):
        """Predict audio file class."""
        output = self.feature_extractor(x)
        output = self.classifier_head(output)
        return output


if __name__ == "__main__":
    net = NeedleNetV1(5)
    print(net)
