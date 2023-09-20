from typing import Optional
import os
from utils import load_json
from configs import BaseConfig, TextConfig, AudioConfig, TrainerConfig
from models import BaseModel

class Trainer:
    """
    Trainer class for the data loading and training stage, along with checkpointing and logging
    """
    def __init__(
            self,
            trainer_config: TrainerConfig,
            model: BaseModel,
            text_config: Optional[TextConfig] = None,
            audio_config: Optional[AudioConfig] = None
        ) -> None:
        # preparing exp directory (nothing if checkpoint resuming)
        self.config = trainer_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.audio_config = audio_config
        self.text_config = text_config
        self.dataset_config = dataset_config

        self.task = self.model_config.task
        self.exp_dir = self.config.exp_dir
        self.dump_dir = self.config.dump_dir

        # checking for exp directory (depending on whether we are resuming training or not)
        if (self.config.resume_from_checkpoint):
            assert os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory does not exist (resume_from_checkpoint was set True)"
        else:
            assert not os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory already exists"
            os.mkdir(self.exp_dir)

        # loading symbols from token_list.txt if TTS model and saving the config file in exp directories
        if text_config != None:
            if text_config.token_map == None:
                text_config.token_map = load_json(os.path.join(self.dump_dir, "token_list.json"))
        
        if self.task == "TTS":
            self.model_config.load_symbols(self.dump_dir)
        self.config_yaml_path = os.path.join(self.exp_dir, "config.yaml")
        BaseConfig.write_configs_to_file(
            path=self.config_yaml_path,
            configs={
                "text_config": self.text_config,
                "audio_config": self.audio_config,
                "trainer_config": self.config,
                # "model_config": self.model_config,
                # "optimizer_config": self.optimizer_config
            }
        )

        # setting up trainer variables
        if (self.config.use_cuda):
            assert torch.cuda.is_available(), "torch CUDA is not available"
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # loading training data
        if (self.task == "TTS"):
            self.collate_fn = tts.utils.TextMelCollate()
            self.train_dataset = tts.utils.TextMelDataset(dataset_split_type="train", dump_dir=self.dump_dir)
        else:
            self.collate_fn = None
            self.train_dataset = vocoder.utils.SigMelDataset(dataset_split_type="train", dump_dir=self.dump_dir, audio_config=self.audio_config, max_frames=self.model_config.max_frames)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.batch_size, collate_fn=self.collate_fn)
        self.iters_per_epoch = len(self.train_dataloader)

        # loading validation data
        if (self.config.run_validation):
            print("run_validation is set to True")
            if (self.task == "TTS"):
                self.validation_dataset = tts.utils.TextMelDataset(dataset_split_type="validation", dump_dir=self.dump_dir)
            else:
                self.validation_dataset = vocoder.utils.SigMelDataset(dataset_split_type="validation", dump_dir=self.dump_dir, audio_config=self.audio_config, max_frames=self.model_config.max_frames)
            assert len(self.validation_dataset) > 0, "run_validation was set True, but validation data ({val_csv}) is empty".format(val_csv=os.path.join(self.dump_dir, "data_validation.csv"))
            self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.validation_batch_size, collate_fn=self.collate_fn, drop_last=True)
            self.validation_dir = os.path.join(self.exp_dir, "validation_runs")
            if not os.path.isdir(self.validation_dir):
                os.mkdir(self.validation_dir)

        # setting up helper classes
        self.checkpoint_manager = CheckpointManager(self.exp_dir, self.config.max_best_models, self.config.save_optimizer_dict, self.task)
        if (self.config.wandb_logger):
            self.wandb = WandbLogger(self)