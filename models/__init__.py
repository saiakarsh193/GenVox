import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Literal, Dict

from configs import TrainerConfig, AudioConfig, TextConfig, BaseConfig

_DATASET_SPLIT_TYPE = Literal[
    "train",
    "eval"   
]

class BaseModel(nn.Module):
    def __init__(self, model_config: BaseConfig, audio_config: AudioConfig, text_config: TextConfig, trainer_config: TrainerConfig) -> None:
        super().__init__()
        self.model_name = self.__class__.__name__
        self.model_config = model_config
        self.audio_config = audio_config
        self.text_config = text_config
        self.trainer_config = trainer_config
        self.use_cuda = self.trainer_config.use_cuda & torch.cuda.is_available()

    def get_train_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        raise NotImplementedError

    def get_eval_dataloader(self, dump_dir: str, num_loader_workers: int, batch_size: int) -> DataLoader:
        raise NotImplementedError
    
    def get_checkpoint_statedicts(self, save_optimizer_dict: bool) -> Dict:
        raise NotImplementedError
    
    def prepare_batch(self, batch: Dict) -> Dict:
        raise NotImplementedError
    
    def train_step(self, batch: Dict) -> None:
        raise NotImplementedError

    def get_train_step_logs(self) -> Dict:
        raise NotImplementedError
    
    def get_eval_step_logs(self) -> Dict:
        raise NotImplementedError