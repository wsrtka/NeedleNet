"""Module containing data-related functions and classes."""
# pylint: disable=import-error

import torch
import numpy as np

from torchaudio import load, functional
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image, ImageReadMode


class AudioDatasetV1(DatasetFolder):
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
        spec = self._transform_signal(signal)
        label = self.file_to_class[index][1]
        return spec, label

    def _transform_signal(self, signal):
        signal = signal.numpy()
        signal = self._time_shift(signal)
        signal = signal / np.max(np.abs(signal))
        signal = torch.from_numpy(signal)
        spec = self._convert_to_spectogram(signal)
        return spec

    def _time_shift(self, signal):
        _, sig_len = signal.shape
        shift = int(np.random.rand(1) * sig_len)
        return np.roll(signal, shift)

    def _convert_to_spectogram(self, signal):
        spec = MelSpectrogram(self.sample_rate, n_fft=1024, hop_length=None, n_mels=64)(
            signal
        )
        spec = AmplitudeToDB(top_db=80)(spec)
        return spec


class CWTDataset(DatasetFolder):
    def __init__(self, root, extensions=(".png"), dwt_dir="DWT", emd_dir="EMD"):
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.file_to_class = DatasetFolder.make_dataset(
            self.root, self.class_to_idx, extensions=extensions
        )
        self.dwt_dir = dwt_dir
        self.emd_dir = emd_dir
        self._link_files()

    def __len__(self):
        return len(self.file_to_class)

    def __getitem__(self, index):
        # read file path and label
        file_path, label = self.file_to_class[index]
        cwt_spec = read_image(file_path, ImageReadMode.GRAY)
        # read corresponding dwt

        # read corresponding emd

        return cwt_spec, label

    def _link_files(self):
        self.file_to_dwt = []
        self.file_to_emd = []
        for file, idx in self.file_to_class:
            base_name = file.split("/")[-1]
            base_name = base_name.split("_")[0]
            dwt_link = f"{self.root}/{self.idx_to_class[idx]}/{self.dwt_dir}/{base_name}_dwt_scales.csv"
            emd_link = f"{self.root}/{self.idx_to_class[idx]}/{self.emd_dir}/{base_name}_emd_imfs.csv"
            self.file_to_dwt.append(dwt_link)
            self.file_to_emd.append(emd_link)


# used for testing
if __name__ == "__main__":
    nd = CWTDataset("./cwt_data")
    print(nd.classes)
    print(len(nd))
    print(nd[0], nd[0][0].shape)
    print(nd.file_to_class[0])
