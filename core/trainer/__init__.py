import os
import torch
from torch.utils.data import DataLoader
from typing import Optional

from utils import load_json
from configs import BaseConfig, TextConfig, AudioConfig, TrainerConfig
from models import BaseModel
from .checkpoint_manager import CheckpointManager

class Trainer:
    """
    Trainer class for the data loading and training stage, along with checkpointing and logging
    """
    def __init__(
            self,
            model: BaseModel,
            trainer_config: TrainerConfig,
            text_config: Optional[TextConfig] = None,
            audio_config: Optional[AudioConfig] = None,
            dump_dir: str = "dump",
            exp_dir: str = "exp"
        ) -> None:
        self.model = model
        self.config = trainer_config
        self.audio_config = audio_config
        self.text_config = text_config

        self.dump_dir = dump_dir
        self.exp_dir = exp_dir

        # loading token_map
        if self.text_config != None:
            if self.text_config.token_map == None:
                self.text_config.token_map = load_json(os.path.join(self.dump_dir, "token_list.json"))

        # checking for exp directory (depending on whether we are resuming training or not)
        if (self.config.checkpoint_path != None):
            print(f"overriding exp_dir ({self.exp_dir}) with checkpoint_path ({self.config.checkpoint_path}) to resume training")
            self.exp_dir = self.config.checkpoint_path
            assert os.path.isdir(self.exp_dir), f"checkpoint_path ({self.exp_dir}) does not exist"
        else:
            assert not os.path.isdir(self.exp_dir), f"exp_dir ({self.exp_dir}) already exists"
            os.mkdir(self.exp_dir)

        # write all the configs
        # NOTE: only if no checkpoint path, else load it from there?
        self.config_yaml_path = os.path.join(self.exp_dir, "config.yaml")
        BaseConfig.write_configs_to_file(
            path=self.config_yaml_path,
            configs={
                "trainer_config": self.config,
                "text_config": self.text_config,
                "audio_config": self.audio_config,
                # "model_config": self.model_config,
            }
        )

        # setting device
        if (self.config.use_cuda):
            if not torch.cuda.is_available():
                print("torch CUDA is not available, using CPU instead")
                self.device = "cpu"
            else:
                self.device = "cuda:0"
        else:
            self.device = "cpu"

        # setting flags correctly
        if self.config.debug_run:
            self.config.run_eval = False
            self.config.use_wandb = False

        # setting dataloaders
        self.train_dataloader: DataLoader = model.get_train_dataloader(
            dump_dir=self.dump_dir,
            num_loader_workers=self.config.num_loader_workers,
            batch_size=self.config.batch_size
        )
        if self.config.run_eval:
            self.eval_dataloader: DataLoader = model.get_eval_dataloader(
                dump_dir=self.dump_dir,
                num_loader_workers=self.config.num_loader_workers,
                batch_size=self.config.eval_batch_size
            )
            self.eval_outdir = os.path.join(self.exp_dir, "eval_outputs")
            if not os.path.isdir(self.eval_outdir):
                os.mkdir(self.eval_outdir)

        # setting up helper classes
        self.checkpoint_manager = CheckpointManager(
            exp_dir=self.exp_dir,
            max_best_models=self.config.max_best_models,
            save_optimizer_dict=self.config.save_optimizer_dict
        )
        # if (self.config.wandb_logger):
        #     self.wandb = WandbLogger(self)

    def run(self):
        ...