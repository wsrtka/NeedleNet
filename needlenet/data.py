"""Module containing data-related functions and classes."""
# pylint: disable=import-error

import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchaudio import load, functional
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image, write_png, ImageReadMode


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
    def __init__(self, root, extensions=(".png")):
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.file_to_class = DatasetFolder.make_dataset(
            self.root, self.class_to_idx, extensions=extensions
        )
        self._link_files()

    def __len__(self):
        return len(self.file_to_class)

    def __getitem__(self, index):
        # read file path and label
        file_path, label = self.file_to_class[index]
        cwt_spec = read_image(file_path, ImageReadMode.GRAY)
        # read corresponding dwt
        dwt = torch.load(self.file_to_dwt[index])
        # read corresponding emd
        emd = torch.load(self.file_to_emd[index])
        # apply data transformations
        cwt_spec, dwt, emd = self._transform_data(cwt_spec, dwt, emd)
        return cwt_spec, dwt, emd, label

    def _link_files(self):
        self.file_to_dwt = []
        self.file_to_emd = []
        for file, idx in self.file_to_class:
            rec_name = file.split("/")[-2]
            part = file.split("/")[-1].split(".")[0]
            part = part[-1]
            dwt_link = f"{self.root}/{self.idx_to_class[idx]}/{rec_name}/dwt{part}.pt"
            emd_link = f"{self.root}/{self.idx_to_class[idx]}/{rec_name}/emd{part}.pt"
            self.file_to_dwt.append(dwt_link)
            self.file_to_emd.append(emd_link)

    def _transform_data(self, cwt, dwt, emd):
        # cut frequencies lower than 300Hz from cwt
        cwt = cwt[:, :625]
        return cwt, dwt, emd


def split_cwt_data(dataset_path, target_data_path):
    os.makedirs(target_data_path, exist_ok=True)

    dataset = CWTDataset(dataset_path)

    for idx in range(len(dataset)):
        # get file names
        cwt_file, _ = dataset.file_to_class[idx]
        dwt_file = dataset.file_to_dwt[idx]
        emd_file = dataset.file_to_emd[idx]
        # get target directory name
        target_subdir = cwt_file.split("/")
        target_subdir[1] = target_data_path
        target_subdir[-1] = target_subdir[-1].split("_")[0]
        target_subdir.pop(-2)
        target_subdir = "/".join(target_subdir)
        os.makedirs(target_subdir, exist_ok=True)
        # read cwt
        cwt = read_image(cwt_file, ImageReadMode.GRAY)
        # read dwt
        dwt = pd.read_csv(dwt_file)
        dwt = torch.tensor(dwt.values).T
        # read emd
        emd = pd.read_csv(emd_file)
        emd = emd.iloc[:, :10]
        emd = torch.tensor(emd.values).T
        # cut the first and last .5s from data
        offset = ((cwt.shape[-1] % 445) // 2) + 222
        cwt = cwt[:, :, offset:-offset]
        cwt = cwt[:, :, cwt.shape[-1] % 445 :]
        offset = ((dwt.shape[-1] % 44500) // 2) + 22200
        offset -= offset % 100
        dwt = dwt[:, offset:-offset]
        dwt = dwt[:, dwt.shape[-1] % 44500 :]
        emd = emd[:, offset:-offset]
        emd = emd[:, emd.shape[-1] % 44500 :]
        # cut and save data for every 1s
        for idx in range(cwt.shape[-1] // 445):
            cwt_part = cwt[:, :, idx * 445 : (idx + 1) * 445]
            write_png(cwt_part, f"{target_subdir}/cwt{idx}.png")
            dwt_part = dwt[:, idx * 44500 : (idx + 1) * 44500]
            torch.save(dwt_part, f"{target_subdir}/dwt{idx}.pt")
            emd_part = emd[:, idx * 44500 : (idx + 1) * 44500]
            torch.save(emd_part, f"{target_subdir}/emd{idx}.pt")


# used for testing
if __name__ == "__main__":
    nd = CWTDataset("./cwt_processed")
    # print(nd.classes)
    # print(len(nd))
    print(nd[0])
    print(nd[0][0].shape, nd[0][1].shape, nd[0][2].shape)
    # print(nd.file_to_class[0])
    # plt.imshow(nd[0][0][0])
    # plt.show()

    ### *
    old_folder = "./cwt_data"
    new_folder = "./cwt_data_modified"

    # split_cwt_data('./cwt_data', './cwt_processed')
