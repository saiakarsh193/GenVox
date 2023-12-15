import os
import random
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from configs import AudioConfig, BaseConfig
from models import BaseModel, _DATASET_SPLIT_TYPE
from utils.audio import normalize_signal

class WavMelDataset(torch.utils.data.Dataset):
    def __init__(self, audio_config: AudioConfig, max_frames: int = 300, dataset_split_type: _DATASET_SPLIT_TYPE = "train", dump_dir: str = "dump"):
        dataset_path = os.path.join(dump_dir, f"data_{dataset_split_type}.csv")
        with open(dataset_path) as f:
            self.raw_data = f.readlines()
        self.audio_config = audio_config
        self.max_frames = max_frames
        self.max_wav_len = self.max_frames * self.audio_config.hop_length
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        unique_id, audio_path, feature_path, tokens_ind = self.raw_data[index].strip().split("|")
        fs, wav = scipy.io.wavfile.read(audio_path)
        wav = normalize_signal(wav)
        feats = np.load(feature_path)
        if (feats.shape[1] < self.max_frames): # pad zeros to get the max_frames and max_wav_len shape
            frames_pad = self.max_frames - feats.shape[1]
            feats = np.pad(feats, ((0, 0), (0, frames_pad)), mode="constant", constant_values=0.0)
            if wav.shape[0] < self.max_wav_len:
                wav_pad = self.max_wav_len - wav.shape[0]
                wav = np.pad(wav, ((0, wav_pad)), mode="constant", constant_values=0.0)
            else:
                wav = wav[: self.max_wav_len]
        else: # randomly sample a part of the mel and wav to get the max_frames and max_wav_len shape
            frame_start = random.randint(0, feats.shape[1] - self.max_frames)
            feats = feats[:, frame_start: frame_start + self.max_frames]
            wav_start = frame_start * self.audio_config.hop_length
            wav = wav[wav_start: wav_start + self.max_wav_len]
        wav = torch.FloatTensor(wav).unsqueeze(0) # [1, max_wav_len]
        feats = torch.FloatTensor(feats) # [n_mels, n_frames]
        return {
            "wav": wav,
            "feature": feats
        }
    
    def __len__(self) -> int:
        return len(self.raw_data)

class WavMelCollateFn:
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # we need to use collate function, otherwise torch will check the batch dimensions and will throw error
        wavs = torch.FloatTensor(len(batch), 1, batch[0]["wav"].shape[1])
        feats = torch.FloatTensor(len(batch), batch[0]["feature"].shape[0], batch[0]["feature"].shape[1])
        for ind in range(len(batch)):
            wavs[ind, :] = batch[ind]["wav"]
            feats[ind, :] = batch[ind]["feature"]
        return {
            "wavs": wavs,
            "features": feats
        }

class VocoderModel(BaseModel):
    def __init__(self, model_config: BaseConfig, audio_config: AudioConfig) -> None:
        super().__init__(model_config=model_config, audio_config=audio_config)

    def get_train_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        return DataLoader(
            WavMelDataset(
                audio_config=self.audio_config,
                max_frames=self.model_config.max_frames,
                dataset_split_type="train",
                dump_dir=dump_dir
            ),
            num_workers=num_loader_workers,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=WavMelCollateFn()
        )
    
    def get_eval_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        return DataLoader(
            WavMelDataset(
                audio_config=self.audio_config,
                max_frames=self.model_config.max_frames,
                dataset_split_type="eval",
                dump_dir=dump_dir
            ),
            num_workers=num_loader_workers,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=WavMelCollateFn(),
            drop_last=True
        )
    
    def prepare_batch(self, batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        for key, val in batch.items():
            # NOTE: might need to do val.contiguous()
            batch[key] = val.to(device=device)
        return batch

    def get_wandb_metrics(self) -> List[Tuple[str, str]]:
        return [
            ("loss", "min"),
            ("loss_eval", "min")
        ]
