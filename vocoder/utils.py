import os
import numpy as np
import scipy.io
import torch
import random

from audio import normalize_signal
from config import AudioConfig

class SigMelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_split_type: str = None, dump_dir: str = "dump", max_frames: int = 300, audio_config: AudioConfig = AudioConfig()):
        assert os.path.isdir(dump_dir), f"dump ({dump_dir}) directory does not exist"
        if dataset_split_type == None:
            dataset_path = "data.csv"
        else:
            assert dataset_split_type in ["train", "validation"], f"invalid dataset_split_type ({dataset_split_type}) given"
            dataset_path = f"data_{dataset_split_type}.csv"
        self.audio_config = audio_config
        self.max_frames = max_frames
        self.max_signal_length = self.max_frames * self.audio_config.hop_length
        with open(os.path.join(dump_dir, dataset_path)) as f:
            self.raw_data = f.readlines()

    def __getitem__(self, index):
        # NOTE: random noise adding should be tested
        raw_value = self.raw_data[index].strip().split("|")
        signal_path, feats_path = raw_value[1], raw_value[2]
        _, signal = scipy.io.wavfile.read(signal_path) # (signal_length, )
        signal = normalize_signal(signal)
        feats = np.load(feats_path) # (n_mels, n_frames)
        if (feats.shape[1] < self.max_frames): # pad zeros to get the max_frames and max_signal_length shape
            frames_pad = self.max_frames - feats.shape[1]
            feats = np.pad(feats, ((0, 0), (0, frames_pad)), mode="constant", constant_values=0.0)
            if signal.shape[0] < self.max_signal_length:
                signal_pad = self.max_signal_length - signal.shape[0]
                signal = np.pad(signal, ((0, signal_pad)), mode="constant", constant_values=0.0)
            else:
                signal = signal[: self.max_signal_length]
        else: # randomly sample a part of the mel and signal to get the max_frames and max_signal_length shape
            random_start_frame = random.randint(0, feats.shape[1] - self.max_frames)
            feats = feats[:, random_start_frame: random_start_frame + self.max_frames]
            random_start_signal = random_start_frame * self.audio_config.hop_length
            signal = signal[random_start_signal: random_start_signal + self.max_signal_length]
        signal = torch.FloatTensor(signal).unsqueeze(0) # (1, signal_length)
        feats = torch.FloatTensor(feats)
        return signal, feats

    def __len__(self):
        return len(self.raw_data)
