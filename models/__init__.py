import torch
from torch.utils.data import DataLoader
from typing import Literal, Dict

_DATASET_SPLIT_TYPE = Literal[
    "train",
    "eval"   
]

class BaseModel:
    def __init__(self) -> None:
        ...

    def get_train_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        raise NotImplementedError

    def get_eval_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        raise NotImplementedError
    
    def get_checkpoint_statedicts(self, save_optimizer_dict: bool) -> Dict:
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
    
    # def