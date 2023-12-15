from __future__ import annotations
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Literal, Dict, Callable, Union, List, Tuple, Optional

from configs import BaseConfig, AudioConfig

_DATASET_SPLIT_TYPE = Literal[
    "train",
    "eval"   
]

class BaseModel(nn.Module):
    def __init__(self, model_config: BaseConfig, audio_config: AudioConfig) -> None:
        super().__init__()
        self.model_name = self.__class__.__name__
        self.model_config = model_config
        self.audio_config = audio_config

    def get_train_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        raise NotImplementedError

    def get_eval_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        raise NotImplementedError

    def get_criterion(self) -> Dict[str, Union[Callable, nn.Module]]:
        raise NotImplementedError

    def get_optimizer(self) -> Dict[str, torch.optim.Optimizer]:
        raise NotImplementedError

    def get_wandb_metrics(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def get_checkpoint_statedicts(self, optimizer: Optional[Dict[str, torch.optim.Optimizer]] = None) -> Dict:
        raise NotImplementedError

    def load_checkpoint_statedicts(self, statedicts: Dict, save_optimizer_dict: bool, optimizer: Dict[str, torch.optim.Optimizer]) -> None:
        raise NotImplementedError

    def prepare_batch(self, batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def train_step(self, batch: Dict, criterion: Dict, optimizer: Dict) -> None:
        raise NotImplementedError

    def eval_step(self, batch: Dict, criterion: Dict, eval_outdir: Optional[str] = None) -> None:
        raise NotImplementedError
    
    def get_eval_priority(self) -> float:
        raise NotImplementedError

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_train_step_logs(self) -> Dict:
        raise NotImplementedError

    def get_eval_step_logs(self, wandb_logger) -> Dict:
        raise NotImplementedError

    @staticmethod
    def load_from_config(config_path: str) -> BaseModel:
        raise NotImplementedError
