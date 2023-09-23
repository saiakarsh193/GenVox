import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional

from utils import load_json, current_formatted_time, center_print, log_print, sec_to_formatted_time, print_parameter_count
from configs import BaseConfig, TextConfig, AudioConfig, TrainerConfig
from models import BaseModel
from .checkpoint_manager import CheckpointManager
from .wandb_logger import WandbLogger

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
        ):
        self.model = model
        self.config = trainer_config
        self.text_config = text_config
        self.audio_config = audio_config
        self.dump_dir = dump_dir
        self.exp_dir = exp_dir

        # setting device
        if (self.config.use_cuda):
            if not torch.cuda.is_available():
                print("torch CUDA is not available. switching from GPU to CPU (will be relatively very slow)")
                self.device = "cpu"
            else:
                self.device = "cuda:0"
        else:
            self.device = "cpu"

        # setting flags correctly
        if self.config.debug_run:
            self.config.run_eval = False
        if not self.config.run_eval:
            self.config.use_wandb = False

    def _pre_run_setup(self):
        center_print(f"TRAINING PREPARATION ({current_formatted_time()})", space_factor=0.35)

        if self.config.debug_run:
            print("debug_run set to True, skipping unnecessary prints, asserts")

        # checking for exp directory (depending on whether we are resuming training or not, and debug_run is True or not)
        if (self.config.checkpoint_path != None):
            print(f"overriding exp_dir ({self.exp_dir}) with checkpoint_path ({self.config.checkpoint_path}) to resume training")
            self.exp_dir = self.config.checkpoint_path
            assert os.path.isdir(self.exp_dir), f"checkpoint_path ({self.exp_dir}) does not exist"
        else:
            if not self.config.debug_run: # if debug_run, dont do assert
                assert not os.path.isdir(self.exp_dir), f"exp_dir ({self.exp_dir}) already exists"
            if not os.path.isdir(self.exp_dir): # create exp_dir if not already there
                os.mkdir(self.exp_dir)

        # write all the configs
        # NOTE: only if no checkpoint path, else load it from there?
        print(f"writing configs to {self.exp_dir} directory")
        self.config_yaml_path = os.path.join(self.exp_dir, "config.yaml")
        BaseConfig.write_configs_to_file(
            path=self.config_yaml_path,
            configs={
                "trainer_config": self.config,
                "text_config": self.text_config,
                "audio_config": self.audio_config,
                "model_config": self.model.model_config,
            }
        )

        # printing configs
        if not self.config.debug_run:
            print(self.config)
            print(self.model.model_config)

        # setting up helper classes
        if self.config.run_eval:
            self.checkpoint_manager = CheckpointManager(
                exp_dir=self.exp_dir,
                max_best_models=self.config.max_best_models,
                save_optimizer_dict=self.config.save_optimizer_dict
            )
        if self.config.use_wandb:
            self.wandb_logger = WandbLogger(
                project_name=self.config.project_name,
                experiment_id=self.config.experiment_id,
                notes=self.config.notes,
                model_name=self.model.model_name,
                seed=self.config.seed,
                epochs=self.config.epochs
            )

        # setting all the seeds
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        # setting criterion and optimizers
        self.criterion = self.model.get_criterion()
        self.optimizer = self.model.get_optimizer()

        # data loading and outdir prep
        print("loading train dataloader")
        self.train_dataloader: DataLoader = self.model.get_train_dataloader(
            dump_dir=self.dump_dir,
            num_loader_workers=self.config.num_loader_workers,
            batch_size=self.config.batch_size if not self.config.debug_run else 2
        )
        if self.config.run_eval:
            print("run_eval set as True. loading eval dataloader, preparing output log directory")
            self.eval_dataloader: DataLoader = self.model.get_eval_dataloader(
                dump_dir=self.dump_dir,
                num_loader_workers=self.config.num_loader_workers,
                batch_size=self.config.eval_batch_size
            )
            # setup directory to store eval outputs
            self.eval_outdir = os.path.join(self.exp_dir, "eval_outputs")
            if not os.path.isdir(self.eval_outdir):
                os.mkdir(self.eval_outdir)

        # loading checkpoint state_dict into model to resume training
        # if (self.config.checkpoint_path != None):

        if not self.config.debug_run: # print model details only when NOT in debug_run
            center_print(f"MODEL DETAILS", space_factor=0.1)
            self.model.train()
            print(self.model)
            print_parameter_count(self.model)

        self.epoch_start = 0
        self.iteration_start = 0

    def _eval_loop(self, iteration: int) -> float:
        center_print(f"EVALUATION ({current_formatted_time()})", space_factor=0.35)
        return 0

    def _train_loop(self):
        # setting training variables
        avg_time_epoch = 0
        iteration = self.iteration_start
        total_iterations = len(self.train_dataloader) * self.config.epochs
        start_train = time.time() # start time of training

        # training loop
        for epoch in range(self.epoch_start, self.config.epochs):
            start_epoch = time.time() # start time of epoch
            log_print(f"epoch start: {epoch + 1} / {self.config.epochs}")
            for ind, batch in enumerate(self.train_dataloader):
                start_iter = time.time() # start time of iteration
                batch = self.model.prepare_batch(batch)
                self.model.train_step(
                    batch=batch,
                    criterion=self.criterion,
                    optimizer=self.optimizer
                )
                end_iter = time.time() # end time of iteration
                iteration += 1
                log_print(f"{iteration} ({ind + 1}/{len(self.train_dataloader)}|{epoch + 1}) -> {end_iter - start_iter: .2f} s")

                # evaluation and checkpoint saving
                if (self.config.run_eval and (iteration % self.config.iters_for_checkpoint == 0 or (iteration == total_iterations))): # every iters_per_checkpoint or last iteration
                    priority_value = self._eval_loop(iteration)
                    self.checkpoint_manager.save_model(
                        iteration=iteration, 
                        model=self.model,
                        priority_value=priority_value
                    )

                # logging to wandb
                if (self.config.use_wandb):
                    self.wandb_logger.log(self.model.get_train_step_logs(), epoch=iteration, commit=True)
                
                if self.config.debug_run:
                    break

            end_epoch = time.time() # end time of epoch
            epoch_time = end_epoch - start_epoch
            # update the average epoch time (removed epoch start offset to get correct counts)
            avg_time_epoch = ((avg_time_epoch * (epoch - self.epoch_start)) + epoch_time) / ((epoch - self.epoch_start) + 1)
            log_print(f"epoch end (left: {self.config.epochs - epoch}) -> time_taken: {sec_to_formatted_time(epoch_time)}")
            log_print(f"average time per epoch: {sec_to_formatted_time(avg_time_epoch)}")
            log_print(f"time elapsed: {sec_to_formatted_time(end_epoch - start_train)}")
            # calculate ETA: epochs left * avg time per epoch
            remaining_time = (self.config.epochs - (epoch + 1)) * avg_time_epoch
            log_print(f"estimated time remaining: {sec_to_formatted_time(remaining_time)}")
            log_print(f"training ending at {current_formatted_time(sec_add=remaining_time)}")
            print()

            if self.config.debug_run:
                break

        # training done
        center_print(f"TRAINING END ({current_formatted_time()})", space_factor=0.35)
        end_train = time.time() # end time of training
        print(f"total time of training: {sec_to_formatted_time(end_train - start_train)}")

    def run(self):
        self._pre_run_setup()

        center_print(f"TRAINING START ({current_formatted_time()})", space_factor=0.35)
        print()
        print(f"Model: {self.model.model_name}")
        print(f"Project: {self.config.project_name}")
        print(f"Experiment: {self.config.experiment_id}")
        print(f"Seed: {self.config.seed}")
        print(f"Device: {self.device}")
        print(f"Epochs: {(self.config.epochs - self.epoch_start)} (start: {self.epoch_start})")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Iterations per epoch: {len(self.train_dataloader)}")
        print(f"Total iterations: {len(self.train_dataloader) * (self.config.epochs - self.epoch_start)} (start: {self.iteration_start}, end: {len(self.train_dataloader) * self.config.epochs})")
        if (self.config.checkpoint_path != None):
            print(f"Training resuming from checkpoint ({self.config.checkpoint_path}), epoch start: {self.epoch_start}, iteration start: {self.iteration_start}")
        print()

        self._train_loop()
        
        # stop wandb syncing
        if (self.config.use_wandb):
            self.wandb_logger.finish()
