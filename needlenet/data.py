"""Module containing data-related functions and classes."""
# pylint: disable=import-error

import os

from random import shuffle
from collections import defaultdict
from pickle import HIGHEST_PROTOCOL

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from torch.utils.data import Subset
from torchaudio import load, save, functional
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Normalize, Resize
from torchvision.io import read_image, write_png, ImageReadMode


class AudioDataset(DatasetFolder):
    """Datset used for loading audio files."""

    def __init__(self, root, extensions=(".wav"), sample_rate=44500):
        self.classes, self.class_to_idx = self.find_classes(root)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
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
        # todo: remove average of signal
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
        self.resize_cwt = Resize((224, 160))

    def __len__(self):
        return len(self.file_to_class)

    def __getitem__(self, index):
        # read file path and label
        file_path, label = self.file_to_class[index]
        cwt_spec = read_image(file_path, ImageReadMode.GRAY).to(torch.float32)
        # read corresponding dwt
        # dwt = torch.load(self.file_to_dwt[index])
        dwt = torch.zeros(12)
        # read corresponding emd
        # emd = torch.load(self.file_to_emd[index])
        emd = torch.zeros(12)
        # apply data transformations
        cwt_spec, dwt, emd = self._transform_data(cwt_spec, dwt, emd)
        # cwt_spec = torch.cat([cwt_spec, cwt_spec, cwt_spec])
        # cwt_spec = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        #     cwt_spec
        # )
        return cwt_spec, label

    def _link_files(self):
        self.file_to_dwt = []
        self.file_to_emd = []
        for file, idx in self.file_to_class:
            # rec_name = file.split("/")[-2]
            rec_name = file.split("/")[-1].split("_")[0]
            part = file.split("/")[-1].split(".")[0]
            part = part[-1]
            dwt_link = f"{self.root}/{self.idx_to_class[idx]}/{rec_name}/dwt{part}.pt"
            emd_link = f"{self.root}/{self.idx_to_class[idx]}/{rec_name}/emd{part}.pt"
            # dwt_link = f"{self.root}/{self.idx_to_class[idx]}/DWT/{rec_name}_dwt_scales.csv"
            # emd_link = f"{self.root}/{self.idx_to_class[idx]}/EMD/{rec_name}_emd_imfs.csv"
            self.file_to_dwt.append(dwt_link)
            self.file_to_emd.append(emd_link)

    def _transform_data(self, cwt, dwt, emd):
        # cut frequencies lower than 300Hz from cwt
        cwt = cwt[:, :625]
        # normalize spectrogram data
        cwt = cwt / 255
        # resize spectrogram
        cwt = self.resize_cwt(cwt)
        # convert to dtype compatible with mps
        cwt = cwt.to(torch.float32)
        # dwt = dwt.to(torch.float32)
        # emd = emd.to(torch.float32)
        return cwt, dwt, emd


def split_data(dataset_path, target_data_path):
    """Split CWT, EMD and DWT data from dataset_path into 1 second parts
    and save them into target_data_path.
    """
    os.makedirs(target_data_path, exist_ok=True)

    dataset = AudioDataset(dataset_path)
    sr = dataset.sample_rate

    for idx in range(len(dataset)):
        # get file names
        data_file, _ = dataset.file_to_class[idx]
        # get target directory name
        target_subdir = data_file.split("/")
        target_subdir[1] = target_data_path
        target_subdir[-1] = target_subdir[-1].split(".")[0]
        target_subdir.pop(0)
        target_subdir = "/".join(target_subdir)
        os.makedirs(target_subdir, exist_ok=True)
        # read cwt
        signal, _ = load(data_file)
        # cut the first and last .5s from data
        offset = ((signal.shape[-1] % sr) // 2) + sr // 2
        signal = signal[:, offset:-offset]
        signal = signal[:, signal.shape[-1] % sr :]
        print(signal.shape)
        # cut and save data for every 1s
        for signal_idx in range(signal.shape[-1] // sr):
            signal_part = signal[:, signal_idx * sr : (signal_idx + 1) * sr]
            print(signal_part.shape)
            save(f"{target_subdir}/part{signal_idx}.wav", signal_part, sr)


def file_length_split(dataset, ratios):
    """Split dataset into n subsets so that each subset has its assigned ratio of samples.
    The functions takes into account a specific use case: the raw audio files are split
    into 1 second parts. In order to prevent data leakage, all samples resulting from one
    raw audio file split should be in the same subset.

    Args:
        dataset (DatasetFolder): The dataset to be split.
        ratios (iterable(float)): Ratio of training samples to testing samples.

    Returns:
        tuple(Subset): Two subsets resulting in the dataset split.
    """
    assert sum(ratios) == 1, "Ratios do not sum to 1."
    # variable for holding the number of files per recording split
    data_count = {dataset.class_to_idx[c]: defaultdict(int) for c in dataset.classes}
    # variable for holding the number of examples per class
    class_count = {dataset.class_to_idx[c]: 0 for c in dataset.classes}
    # calculate the number of files per split and examples per class
    for file, idx in dataset.file_to_class:
        rec_name = file.split("/")[-2]
        data_count[idx][rec_name] += 1
        class_count[idx] += 1
    # calculate the number of instances per class that should be in the training subset
    split_classes_count = []
    for ratio in ratios:
        split_classes_count.append(
            {k: int(count * ratio) for k, count in class_count.items()}
        )
    # variable for holding decision on subset membership of recording
    data_split = data_count.copy()
    # split the dataset
    for class_idx, files in data_count.items():
        dirs = list(files.keys())
        # ensure randomness of split
        shuffle(dirs)
        for d in dirs:
            for split_idx, cls_count in enumerate(split_classes_count):
                # put example into train dataset
                if cls_count[class_idx] > 0:
                    cls_count[class_idx] -= data_count[class_idx][d]
                    data_split[class_idx][d] = split_idx
                    break
    # extract indices of dataset samples that belong in each subset
    indices = [list() for _ in split_classes_count]
    for idx, (file, class_idx) in enumerate(dataset.file_to_class):
        rec_name = file.split("/")[-2]
        try:
            indices[data_split[class_idx][rec_name]].append(idx)
        except IndexError:
            pass
    # create subsets
    subsets = [Subset(dataset, idxs) for idxs in indices]
    return subsets


# used for testing
if __name__ == "__main__":
    # nd = CWTDataset("./cwt_processed")
    # print(len(nd))
    # print(nd[0][0][0].shape)
    # print(len(nd[0]))

    # print(nd[0][0].shape, nd[0][1].shape, nd[0][2].shape)
    # print(nd.file_to_class[0])
    # img = nd[0][0][0]
    # img = img.squeeze(0)
    # plt.imshow(img)
    # plt.show()

    # ds = file_length_split(nd, (0.2, 0.2, 0.2, 0.2, 0.2))
    # for d in ds:
    #     print(len(d), end=", ")

    ### *
    old_folder = "./data"
    new_folder = "./new_data"

    split_data(old_folder, new_folder)
