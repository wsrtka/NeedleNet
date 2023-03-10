"""Module containing data-related functions and classes."""
# pylint: disable=import-error

from torchaudio import load, functional
from torchvision.datasets import DatasetFolder


class AudioDataset(DatasetFolder):
    """Datset used for loading audio files."""

    def __init__(self, root, extensions, sample_rate):
        self.classes, self.class_to_idx = self.find_classes(root)
        self.file_to_class = DatasetFolder.make_dataset(
            root, self.class_to_idx, extensions=extensions
        )
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.file_to_class)

    # pylint: disable=invalid-name
    def __getitem__(self, index):
        audio_path = self.file_to_class[index][0]
        signal, sr = load(audio_path)
        if sr != self.sample_rate:
            signal = functional.resample(signal, sr, self.sample_rate)
        label = self.file_to_class[index][1]
        return signal, label


# used for testing
if __name__ == "__main__":
    nd = AudioDataset("./data", ("wav"), 44500)
    print(nd.classes)
    print(len(nd))
    print(nd[0])
