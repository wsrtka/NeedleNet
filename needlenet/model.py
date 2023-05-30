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
class NeedleNetV2(nn.Module):
    """All convolutional CNN for audio file classification."""

    def __init__(self, num_classes):
        super().__init__()
        feature_extractor = []
        classifier_head = []

        feature_extractor.append(nn.Conv2d(1, 12, 3, padding="same"))
        feature_extractor.append(nn.BatchNorm2d(12))
        feature_extractor.append(nn.ReLU())
        feature_extractor.append(nn.MaxPool2d(3, 2))

        feature_extractor.append(nn.Conv2d(12, 24, 3, padding="same"))
        feature_extractor.append(nn.BatchNorm2d(24))
        feature_extractor.append(nn.ReLU())
        feature_extractor.append(nn.MaxPool2d(3, 2))

        feature_extractor.append(nn.Conv2d(24, 48, 3, padding="same"))
        feature_extractor.append(nn.BatchNorm2d(48))
        feature_extractor.append(nn.ReLU())
        feature_extractor.append(nn.MaxPool2d(3, 2))

        feature_extractor.append(nn.Conv2d(48, 48, 3, padding="same"))
        feature_extractor.append(nn.BatchNorm2d(48))
        feature_extractor.append(nn.ReLU())

        feature_extractor.append(nn.Conv2d(48, 48, 3, padding="same"))
        feature_extractor.append(nn.BatchNorm2d(48))
        feature_extractor.append(nn.ReLU())
        feature_extractor.append(nn.MaxPool2d(3, 2))

        classifier_head.append(nn.Linear(1344, num_classes))

        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.flatten = nn.Flatten()
        self.classifier_head = nn.Sequential(*classifier_head)

    def forward(self, x):
        """Predict audio file class."""
        cwt, dwt, emd = x
        cnn_features = self.feature_extractor(cwt)
        cnn_features = self.flatten(cnn_features)
        dwt = self.flatten(dwt)
        emd = self.flatten(emd)
        print(cnn_features.shape, dwt.shape, emd.shape)
        features = torch.cat([cnn_features, dwt, emd], dim=0)
        output = self.classifier_head(features)
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
    torch.onnx.export(
        net,
        torch.rand(1, 1, 64, 44),
        "model.onnx",
        input_names=["Mel Spectrogram"],
        output_names=["Prediction"],
    )
    print(net)
