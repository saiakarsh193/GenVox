import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any

from models import BaseModel, _DATASET_SPLIT_TYPE

class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_split_type: _DATASET_SPLIT_TYPE = "train", dump_dir: str = "dump"):
        assert os.path.isdir(dump_dir), f"dump ({dump_dir}) directory does not exist"
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

class TTSModel(BaseModel):
    def get_train_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        return DataLoader(
            TextMelDataset(dataset_split_type="train", dump_dir=dump_dir),
            num_workers=num_loader_workers,
            shuffle=True,
            batch_size=batch_size
        )
    
    def get_eval_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        return DataLoader(
            TextMelDataset(dataset_split_type="eval", dump_dir=dump_dir),
            num_workers=num_loader_workers,
            shuffle=True,
            batch_size=batch_size
        )
