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
    def __init__(self, max_best_models, exp_dir):
        self.max_best_models = max_best_models
        self.exp_dir = exp_dir
        assert os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory does not exist"
        self.manager_path = os.path.join(self.exp_dir, "checkpoint_manager.json")
        if not os.path.isfile(self.manager_path):
            dump_json(self.manager_path, [])

    def save_model(self, iteration, model, loss_value):
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
                'model_state_dict': model.state_dict()
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
        self.exp_dir = trainer_config.exp_dir
        self.dump_dir = trainer_config.dump_dir
        if (trainer_config.resume_from_checkpoint):
            assert os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory does not exist (resume_from_checkpoint was set True)"
        else:
            assert not os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory already exists"
            os.mkdir(self.exp_dir)

        model_config.load_symbols(self.dump_dir)
        self.config_yaml_path = os.path.join(self.exp_dir, "config.yaml")
        write_configs(
            self.config_yaml_path,
            text_config=text_config,
            audio_config=audio_config,
            dataset_config=dataset_config,
            trainer_config=trainer_config,
            model_config=model_config,
            optimizer_config=optimizer_config
        )

        self.audio_config = audio_config
        self.config = trainer_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        if (self.config.use_cuda):
            assert torch.cuda.is_available(), "torch CUDA is not available"
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if (self.model_config.task == "TTS"):
            self.collate_fn = tts.utils.TextMelCollate()
            self.train_dataset = tts.utils.TextMelDataset(dataset_split_type="train", dump_dir=self.dump_dir)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.batch_size, collate_fn=self.collate_fn)

        if (self.config.run_validation):
            print("run_validation is set to True")
            if (self.model_config.task == "TTS"):
                self.validation_dataset = tts.utils.TextMelDataset(dataset_split_type="validation", dump_dir=self.dump_dir)
            assert len(self.validation_dataset) > 0, "run_validation was set True, but validation data was not found"
            self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.validation_batch_size, collate_fn=self.collate_fn, drop_last=True)
            self.validation_dir = os.path.join(self.exp_dir, "validation_runs")
            if not os.path.isdir(self.validation_dir):
                os.mkdir(self.validation_dir)

        self.checkpoint_manager = CheckpointManager(self.config.max_best_models, self.exp_dir)
        if (self.config.wandb_logger):
            self.wandb = WandbLogger(self)

    def prepare_for_training(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        center_print(f"TRAINING PREP ({current_formatted_time()})", space_factor=0.35)
        print(self.config)
        print(self.model_config)
        print(self.optimizer_config)
        print(self.audio_config)
        if self.model_config.task == "TTS":
            if self.model_config.model_name == "Tacotron2":
                self.model = tts.tacotron2.Tacotron2(self.model_config, self.audio_config, self.config.use_cuda)
                self.criterion = tts.tacotron2.Tacotron2Loss()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_config.learning_rate, weight_decay=self.optimizer_config.weight_decay)
        self.start_iteration = 0
        self.epoch_start = 0
        # checkpoint loading for resuming the training
        if (self.config.resume_from_checkpoint):
            checkpoint_data = torch.load(self.config.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.start_iteration = checkpoint_data['iteration']
            if (self.config.epoch_start == 1):
                self.epoch_start = self.start_iteration // len(self.train_dataloader)
            else:
                self.epoch_start = self.config.epoch_start - 1
        self.model.train()
        print(self.model)
        if (self.config.wandb_logger):
            self.wandb.define_metric('loss', summary='min')
            if (self.config.run_validation):
                self.wandb.define_metric('validation_loss', summary='min')

    def validation(self, iteration):
        assert self.config.run_validation, "run_validation was set as False"
        self.model.eval()
        log_print(f"validation start -> validation_batch_count: {len(self.validation_dataloader)}")
        start_valid = time.time()
        valid_loss = 0
        with torch.no_grad():
            for ind, batch in enumerate(self.validation_dataloader):
                x, y = self.model.parse_batch(batch)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss_value = loss.item()
                valid_loss += loss_value
        valid_loss /= len(self.validation_dataloader)
        end_valid = time.time()
        log_print(f"validation end -> validation_loss: {valid_loss: .3f}, time_taken: {sec_to_formatted_time(end_valid - start_valid)}")
        # saving validation files
        validation_run_path = os.path.join(self.validation_dir, f"iter_{iteration}")
        if not os.path.isdir(validation_run_path):
            os.mkdir(validation_run_path)
        rand_ind = random.randrange(0, len(batch)) # selecting random sample in batch
        if self.model_config.task == "TTS":
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
        # logging
        if (self.config.wandb_logger):
            self.wandb.log({'validation_loss': valid_loss}, epoch=iteration)
            if self.model_config.task == "TTS":
                mel_target_img = self.wandb.Image(validation_run_mel_tar_path, caption='mel target')
                mel_predicted_img = self.wandb.Image(validation_run_mel_pred_path, caption='mel predicted')
                self.wandb.log({'mel_plots': [mel_target_img, mel_predicted_img]}, epoch=iteration)
                gate_img = self.wandb.Image(validation_run_gate_path, caption='gate')
                self.wandb.log({'gate_plot': gate_img}, epoch=iteration)
                align_img = self.wandb.Image(validation_run_align_path, caption='alignment')
                self.wandb.log({'alignment_plot': align_img}, epoch=iteration)
        self.model.train()
        return valid_loss

    def train(self):
        self.prepare_for_training()
        print()
        print(f"Project: {self.config.project_name}")
        print(f"Experiment: {self.config.experiment_id}")
        print(f"Model: {self.model_config.model_name}")
        print(f"Task: {self.model_config.task}")
        center_print(f"TRAINING START ({current_formatted_time()})", space_factor=0.35)
        start_train = time.time() # start time of training
        log_print(f"epochs: {self.config.epochs}, batch_count: {len(self.train_dataloader)}, device: {self.device}")
        if (self.config.resume_from_checkpoint):
            print(f"Training resuming from checkpoint ({self.config.checkpoint_path}), epoch start: {self.epoch_start}, iteration start: {self.iteration}")
        iteration = self.start_iteration
        avg_time_epoch = 0
        for epoch in range(self.epoch_start, self.config.epochs):
            start_epoch = time.time() # start time of epoch
            log_print(f"epoch start: {epoch + 1} / {self.config.epochs}")
            for ind, batch in enumerate(self.train_dataloader):
                start_iter = time.time() # start time of iteration
                # token_padded, token_lengths, mel_padded, gate_padded, mel_lengths = batch
                self.optimizer.zero_grad() # same as self.model.zero_grad()
                x, y = self.model.parse_batch(batch)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss_value = loss.item()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimizer_config.grad_clip_thresh)
                self.optimizer.step()
                end_iter = time.time() # end time of iteration
                log_print(f"({epoch + 1} :: {ind + 1} / {len(self.train_dataloader)}) -> iteration: {iteration}, loss: {loss_value: .3f}, grad_norm: {grad_norm: .3f}, time_taken: {end_iter - start_iter: .2f} s")
                iteration += 1

                if (iteration % self.config.iters_for_checkpoint == 0):
                    # validation
                    validation_loss = self.validation(iteration)
                    # checkpoint saving
                    self.checkpoint_manager.save_model(iteration, self.model, validation_loss)
                # logging
                if (self.config.wandb_logger):
                    self.wandb.log({'loss': loss_value, 'grad_norm': grad_norm}, epoch=iteration, commit=True)
            end_epoch = time.time() # end time of epoch
            epoch_time = end_epoch - start_epoch
            avg_time_epoch = (avg_time_epoch * (epoch - self.epoch_start) + epoch_time) / ((epoch - self.epoch_start) + 1)  # update the average epoch time (removed epoch start offset to get correct counts)
            log_print(f"epoch end -> time_taken: {sec_to_formatted_time(epoch_time)}")
            remaining_time = (self.config.epochs - (epoch + 1)) * avg_time_epoch # calculate ETA: epochs left * avg time per epoch
            log_print(f"estimated time remaining: {sec_to_formatted_time(remaining_time)} -> training ending at {current_formatted_time(remaining_time)}")
            print()
        if (self.config.wandb_logger):
            self.wandb.finish()
        center_print(f"TRAINING END ({current_formatted_time()})", space_factor=0.35)
        end_train = time.time() # end time of training
        print(f"total time of training: {sec_to_formatted_time(end_train - start_train)}")
