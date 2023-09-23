import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Literal, Dict, Callable, Union

from configs import TrainerConfig, AudioConfig, TextConfig, BaseConfig

_DATASET_SPLIT_TYPE = Literal[
    "train",
    "eval"   
]

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_train_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        raise NotImplementedError

    def get_eval_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        raise NotImplementedError
    
    def get_criterion(self) -> Dict[str, Union[Callable, nn.Module]]:
        raise NotImplementedError
    
    def get_optimizer(self) -> Dict[str, torch.optim.Optimizer]:
        raise NotImplementedError
    
    def get_checkpoint_statedicts(self, save_optimizer_dict: bool) -> Dict:
        raise NotImplementedError
    
    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def train_step(self, batch: Dict, criterion: Dict, optimizer: Dict) -> None:
        raise NotImplementedError
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_train_step_logs(self) -> Dict:
        raise NotImplementedError
    
    def get_eval_step_logs(self) -> Dict:
        raise NotImplementedError
