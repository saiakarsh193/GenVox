import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from configs import AudioConfig, BaseConfig, TextConfig
from models import BaseModel, _DATASET_SPLIT_TYPE

class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_split_type: _DATASET_SPLIT_TYPE = "train", dump_dir: str = "dump"):
        dataset_path = os.path.join(dump_dir, f"data_{dataset_split_type}.csv")
        with open(dataset_path) as f:
            self.raw_data = f.readlines()
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        unique_id, audio_path, feature_path, tokens_ind = self.raw_data[index].strip().split("|")
        tokens = torch.IntTensor([int(tk) for tk in tokens_ind.split()])
        feats = torch.FloatTensor(np.load(feature_path))
        return {
            "tokens": tokens,
            "features": feats
        }
    
    def __len__(self) -> int:
        return len(self.raw_data)

class TextMelCollateFn:
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # we need to use collate function, otherwise torch will check the batch dimensions and will throw error

        token_ind_sorted = np.argsort([x["tokens"].shape[0] for x in batch])[::-1] # np.argsort sorts in ascending order
        max_token_length = batch[token_ind_sorted[0]]["tokens"].shape[0]
        n_mels = batch[0]["features"].shape[0] # features (mel): [n_mels, n_frames]
        max_n_frames = max([x["features"].shape[1] for x in batch])

        token_padded = torch.LongTensor(len(batch), max_token_length)
        token_padded.zero_()
        token_lengths = torch.LongTensor(len(batch))
        mel_padded = torch.FloatTensor(len(batch), n_mels, max_n_frames)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_n_frames)
        gate_padded.zero_()
        mel_lengths = torch.LongTensor(len(batch))
        for ind in range(len(batch)):
            tokens = batch[token_ind_sorted[ind]]["tokens"]
            token_padded[ind, : tokens.shape[0]] = tokens
            token_lengths[ind] = tokens.shape[0]
            mel = batch[token_ind_sorted[ind]]["features"]
            mel_padded[ind, :, : mel.shape[1]] = mel
            # marking which is the last step
            # for largest -> 000001 (1 in last frame marks that this is the end which is used in decoding inference)
            gate_padded[ind, mel.shape[1] - 1: ] = 1
            mel_lengths[ind] = mel.shape[1]

        return {
            "token_padded": token_padded,
            "token_lengths": token_lengths,
            "mel_padded": mel_padded,
            "gate_padded": gate_padded,
            "mel_lengths": mel_lengths
        }

class TTSModel(BaseModel):
    def __init__(self, model_config: BaseConfig, audio_config: AudioConfig, text_config: TextConfig) -> None:
        super().__init__(model_config=model_config, audio_config=audio_config)
        self.text_config = text_config

    def get_train_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        return DataLoader(
            TextMelDataset(dataset_split_type="train", dump_dir=dump_dir),
            num_workers=num_loader_workers,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=TextMelCollateFn()
        )
    
    def get_eval_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        return DataLoader(
            TextMelDataset(dataset_split_type="eval", dump_dir=dump_dir),
            num_workers=num_loader_workers,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=TextMelCollateFn(),
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
