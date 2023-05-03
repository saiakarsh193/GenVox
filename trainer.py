import os
import random
import numpy as np
from typeguard import typechecked
import torch
import torch.utils.data
import wandb
import time

from config import TextConfig, AudioConfig, DatasetConfig, TrainerConfig, ModelConfig, Tacotron2Config, OptimizerConfig, write_configs
from utils import load_json, dump_json, sec_to_formatted_time, log_print, center_print, current_formatted_time
from utils import saveplot_mel, saveplot_alignment, saveplot_gate
import tts


class CheckpointManager:
    def __init__(self, exp_dir, max_best_models, save_optimizer_dict):
        self.exp_dir = exp_dir
        self.max_best_models = max_best_models
        self.save_optimizer_dict = save_optimizer_dict
        assert os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory does not exist"
        self.manager_path = os.path.join(self.exp_dir, "checkpoint_manager.json")
        if not os.path.isfile(self.manager_path):
            dump_json(self.manager_path, [])

    def save_model(self, iteration, model, optimizer, loss_value):
        # lower the loss_value better the model
        # [(checkpoint_iteration, loss_value)] (increasing order)
        manager_data = load_json(self.manager_path)
        add_at_index = -1
        for index in range(len(manager_data)):
            if (loss_value < manager_data[index][1]):
                add_at_index = index
                break
        if (len(manager_data) == 0):
            add_at_index = 0
        if (add_at_index >= 0):
            checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_{iteration}.pt")
            manager_data.insert(add_at_index, (checkpoint_path, loss_value))
            # for saving the torch model
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict() if self.save_optimizer_dict else None,
                }, checkpoint_path)
        if (len(manager_data) > self.max_best_models):
            model_removed_path = manager_data[-1][0]
            manager_data = manager_data[: -1]
            os.remove(model_removed_path)
        dump_json(self.manager_path, manager_data)


class WandbLogger:
    def __init__(self, trainer):
        wandb.login(key=trainer.config.wandb_auth_key)
        wandb.init(
            project=trainer.config.project_name,
            name=f"exp_{trainer.config.experiment_id}",
            notes=trainer.config.notes,
            config={
                "architecture": trainer.model_config.model_name,
                "task": trainer.model_config.task,
                "epochs": trainer.config.epochs,
                "batch_size": trainer.config.batch_size,
                "seed": trainer.config.seed,
                "device": trainer.device
            },
            tags=["dev", "ljspeech"],
        )

    def define_metric(self, value, summary):
        wandb.define_metric(value, summary=summary)
    
    def Image(self, img, caption=""):
        return wandb.Image(img, caption=caption)

    def log(self, values, epoch, commit=False):
        wandb.log(values, step=epoch, commit=commit)

    def finish(self):
        wandb.finish()


class Trainer:
    """
    Trainer class for the data loading and training stage, along with checkpointing and logging
    """
    @typechecked
    def __init__(
        self,
        text_config: TextConfig,
        audio_config: AudioConfig,
        dataset_config: DatasetConfig,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
    ):
        # preparing exp directory (nothing if checkpoint resuming)
        self.text_config = text_config
        self.audio_config = audio_config
        self.dataset_config = dataset_config
        self.config = trainer_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.task = self.model_config.task
        self.exp_dir = self.config.exp_dir
        self.dump_dir = self.config.dump_dir
        if (self.config.resume_from_checkpoint):
            assert os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory does not exist (resume_from_checkpoint was set True)"
        else:
            assert not os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory already exists"
            os.mkdir(self.exp_dir)

        # loading symbols from token_list.txt if TTS model and saving the config file in exp directorys
        if self.task == "TTS":
            self.model_config.load_symbols(self.dump_dir)
        self.config_yaml_path = os.path.join(self.exp_dir, "config.yaml")
        write_configs(
            self.config_yaml_path,
            text_config=self.text_config,
            audio_config=self.audio_config,
            dataset_config=self.dataset_config,
            trainer_config=self.config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config
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
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.batch_size, collate_fn=self.collate_fn)
        self.iters_per_epoch = len(self.train_dataloader)

        # loading validation data
        if (self.config.run_validation):
            print("run_validation is set to True")
            if (self.task == "TTS"):
                self.validation_dataset = tts.utils.TextMelDataset(dataset_split_type="validation", dump_dir=self.dump_dir)
            assert len(self.validation_dataset) > 0, "run_validation was set True, but validation data ({val_csv}) is empty".format(val_csv=os.path.join(self.dump_dir, "data_validation.csv"))
            self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.validation_batch_size, collate_fn=self.collate_fn, drop_last=True)
            self.validation_dir = os.path.join(self.exp_dir, "validation_runs")
            if not os.path.isdir(self.validation_dir):
                os.mkdir(self.validation_dir)

        # setting up helper classes
        self.checkpoint_manager = CheckpointManager(self.exp_dir, self.config.max_best_models, self.config.save_optimizer_dict)
        if (self.config.wandb_logger):
            self.wandb = WandbLogger(self)

    def prepare_for_training(self):
        center_print(f"TRAINING PREPARATION ({current_formatted_time()})", space_factor=0.35)

        # set seed for all the random modules for experiment repeatability
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        # printing configs
        print(self.config)
        print(self.model_config)
        print(self.optimizer_config)
        print(self.audio_config)

        # loading the model, loss (criterion), and optimizer using hyperparams
        if self.task == "TTS":
            if self.model_config.model_name == "Tacotron2":
                self.model = tts.tacotron2.Tacotron2(self.model_config, self.audio_config, self.config.use_cuda)
                self.criterion = tts.tacotron2.Tacotron2Loss()
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_config.learning_rate, weight_decay=self.optimizer_config.weight_decay)
        self.epoch_start = self.config.epoch_start - 1
        self.iteration_start = 0

        # loading checkpoint state_dict into model to resume training
        if (self.config.resume_from_checkpoint):
            checkpoint_data = torch.load(self.config.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.iteration_start = checkpoint_data['iteration']
            if (self.epoch_start == 0):
                self.epoch_start = self.iteration_start // self.iters_per_epoch
        
        # model ready for training
        self.model.train()
        print(self.model)

        # optimizing torch
        torch.backends.cudnn.enabled = True # speeds up Conv, RNN layers (see dev_log ### 23-04-23)
        # torch.backends.cudnn.benckmark = True # use only if input size is consistent

        # setting up wandb metrics for summarization
        if (self.config.wandb_logger):
            self.wandb.define_metric('loss', summary='min')
            if (self.config.run_validation):
                self.wandb.define_metric('validation_loss', summary='min')

    def validation(self, iteration):
        # validation preparation
        assert self.config.run_validation, "run_validation was set as False"
        log_print(f"validation start -> validation_batch_count: {len(self.validation_dataloader)}")
        start_valid = time.time()
        valid_loss = 0

        # validation loop
        self.model.eval()
        with torch.no_grad():
            for batch in self.validation_dataloader:
                x, y = self.model.parse_batch(batch)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                valid_loss += loss.item()
        self.model.train()

        # logging validation data
        valid_loss /= len(self.validation_dataloader)
        end_valid = time.time()
        log_print(f"validation end -> validation_loss: {valid_loss: .3f}, time_taken: {sec_to_formatted_time(end_valid - start_valid)}")

        # creating validation output directory
        validation_run_path = os.path.join(self.validation_dir, f"iter_{iteration}")
        if not os.path.isdir(validation_run_path):
            os.mkdir(validation_run_path)

        # saving validation outputs for one random sample
        rand_ind = random.randrange(0, len(batch))
        if self.task == "TTS":
            if self.model_config.model_name == "Tacotron2":
                text_length = x[1][rand_ind].item() # x: (text, text_len, max_text_len, mel, mel_len), text_len: (batch)
                mel_length = x[4][rand_ind].item() # x: (text, text_len, max_text_len, mel, mel_len), mel_len: (batch)
                mel_target = y[0][rand_ind, :, : mel_length].cpu().numpy() # y: (mel, gate), mel: (batch, n_mels, max_mel_length)
                gate_target = y[1][rand_ind].cpu().numpy() # y: (mel, gate), gate: (batch, max_mel_length)
                mel_predicted = y_pred[1][rand_ind, :, : mel_length].cpu().numpy() # y_pred: (mel, mel_postnet, gate, align), mel_postnet: (batch, n_mels, max_mel_length)
                gate_predicted = torch.sigmoid(y_pred[2][rand_ind]).cpu().numpy() # y_pred: (mel, mel_postnet, gate, align), gate: (batch, max_mel_length), we take sigmoid to get the actual gate value
                alignments = y_pred[3][rand_ind, : mel_length, : text_length].cpu().numpy() # y_pred: (mel, mel_postnet, gate, align), align: (batch, max_mel_length, max_text_length)
                validation_run_mel_tar_path = os.path.join(validation_run_path, "mel_tar.png")
                validation_run_mel_pred_path = os.path.join(validation_run_path, "mel_pred.png")
                validation_run_gate_path = os.path.join(validation_run_path, "gate.png")
                validation_run_align_path = os.path.join(validation_run_path, "align.png")
                saveplot_mel(mel_target, validation_run_mel_tar_path)
                saveplot_mel(mel_predicted, validation_run_mel_pred_path)
                saveplot_gate(gate_target, gate_predicted, validation_run_gate_path, plot_both=True)
                saveplot_alignment(alignments.T, validation_run_align_path) # we take transpose to get (text_length, mel_length) dimension

        # logging validation data to wandb
        if (self.config.wandb_logger):
            self.wandb.log({'validation_loss': valid_loss}, epoch=iteration)
            if self.task == "TTS":
                if self.model_config.model_name == "Tacotron2":
                    mel_target_img = self.wandb.Image(validation_run_mel_tar_path, caption='mel target')
                    mel_predicted_img = self.wandb.Image(validation_run_mel_pred_path, caption='mel predicted')
                    self.wandb.log({'mel_plots': [mel_target_img, mel_predicted_img]}, epoch=iteration)
                    gate_img = self.wandb.Image(validation_run_gate_path, caption='gate')
                    self.wandb.log({'gate_plot': gate_img}, epoch=iteration)
                    align_img = self.wandb.Image(validation_run_align_path, caption='alignment')
                    self.wandb.log({'alignment_plot': align_img}, epoch=iteration)

        return valid_loss

    def train(self):
        # training preparation
        self.prepare_for_training()

        center_print(f"TRAINING START ({current_formatted_time()})", space_factor=0.35)
        print()
        print(f"Project: {self.config.project_name}")
        print(f"Experiment: {self.config.experiment_id}")
        print(f"Model: {self.model_config.model_name}")
        print(f"Task: {self.task}")
        print(f"Epochs: {(self.config.epochs - self.epoch_start)} (start: {self.epoch_start})")
        print(f"Total iterations: {self.iters_per_epoch * (self.config.epochs - self.epoch_start)} (start: {self.iteration_start})")
        print(f"Batch count: {self.iters_per_epoch}")
        print(f"Device: {self.device}")
        if (self.config.resume_from_checkpoint):
            print(f"Training resuming from checkpoint ({self.config.checkpoint_path}), epoch start: {self.epoch_start}, iteration start: {self.iteration}")
        
        # setting training variables
        start_train = time.time() # start time of training
        iteration = self.iteration_start
        avg_time_epoch = 0

        # training loop
        for epoch in range(self.epoch_start, self.config.epochs):
            start_epoch = time.time() # start time of epoch
            log_print(f"epoch start: {epoch + 1} / {self.config.epochs}")

            for ind, batch in enumerate(self.train_dataloader):
                start_iter = time.time() # start time of iteration
                
                self.optimizer.zero_grad() # same as self.model.zero_grad()
                x, y = self.model.parse_batch(batch)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss_value = loss.item()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimizer_config.grad_clip_thresh)
                self.optimizer.step()
                
                end_iter = time.time() # end time of iteration
                iteration += 1
                log_print(f"({epoch + 1} :: {ind + 1} / {self.iters_per_epoch}) -> iteration: {iteration}, loss: {loss_value: .3f}, grad_norm: {grad_norm: .3f}, time_taken: {end_iter - start_iter: .2f} s")

                if (iteration % self.config.iters_for_checkpoint == 0 or (self.iters_per_epoch * epoch == iteration)): # every iters_per_checkpoint or last iteration
                    # validation
                    validation_loss = self.validation(iteration)
                    # checkpoint saving
                    self.checkpoint_manager.save_model(iteration, self.model, self.optimizer, validation_loss)
                
                # logging to wandb
                if (self.config.wandb_logger):
                    self.wandb.log({'loss': loss_value, 'grad_norm': grad_norm}, epoch=iteration, commit=True)
            
            end_epoch = time.time() # end time of epoch
            epoch_time = end_epoch - start_epoch
            # update the average epoch time (removed epoch start offset to get correct counts)
            avg_time_epoch = ((avg_time_epoch * (epoch - self.epoch_start)) + epoch_time) / ((epoch - self.epoch_start) + 1)
            log_print(f"epoch end (left: {epoch - self.config.epochs}) -> time_taken: {sec_to_formatted_time(epoch_time)}")
            log_print(f"average time per epoch: {sec_to_formatted_time(avg_time_epoch)}")
            log_print(f"time elapsed: {sec_to_formatted_time(end_epoch - start_train)}")
            # calculate ETA: epochs left * avg time per epoch
            remaining_time = (self.config.epochs - (epoch + 1)) * avg_time_epoch
            log_print(f"estimated time remaining: {sec_to_formatted_time(remaining_time)}")
            log_print(f"training ending at {current_formatted_time(sec_add=remaining_time)}")
            print()

        # training done
        center_print(f"TRAINING END ({current_formatted_time()})", space_factor=0.35)
        end_train = time.time() # end time of training
        print(f"total time of training: {sec_to_formatted_time(end_train - start_train)}")

        # stop wandb syncing
        if (self.config.wandb_logger):
            self.wandb.finish()
