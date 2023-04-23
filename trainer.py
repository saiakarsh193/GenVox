import os
import random
import numpy as np
from typeguard import typechecked
import torch
import torch.utils.data
import wandb
import time

from config import TextConfig, AudioConfig, DatasetConfig, TrainerConfig, Tacotron2Config, OptimizerConfig, write_file_from_config
from utils import load_json, dump_json, sec_to_formatted_time, log_print, center_print, current_formatted_time
from utils import saveplot_mel, saveplot_signal, saveplot_alignment, saveplot_gate
import tacotron2

class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_split_type=None):
        dump_dir = "dump"
        assert os.path.isdir(dump_dir), f"dump ({dump_dir}) directory does not exist"
        if dataset_split_type == None:
            dataset_path = "data.csv"
        else:
            assert dataset_split_type in ["train", "validation"], f"invalid dataset_split_type ({dataset_split_type}) given"
            dataset_path = f"data_{dataset_split_type}.csv"
        with open(os.path.join(dump_dir, dataset_path)) as f:
            self.text_wav_raw = f.readlines()
    
    def __getitem__(self, index):
        wav_path, text_tokens = self.text_wav_raw[index].strip().split("|")
        tokens = torch.IntTensor([int(tk) for tk in text_tokens.split()])
        feats = torch.FloatTensor(np.load(wav_path))
        return tokens, feats
    
    def __len__(self):
        return len(self.text_wav_raw)
    

class TextMelCollate:
    def __call__(self, batch):
        """
        batch: [(text_tokens, mel_feats)] -> (batch_size)
        """
        # np.argsort sorts in ascending order
        token_ind_sorted = np.argsort([x[0].shape[0] for x in batch])[::-1]

        max_token_length = batch[token_ind_sorted[0]][0].shape[0]
        token_padded = torch.IntTensor(len(batch), max_token_length)
        token_padded.zero_()
        token_lengths = torch.IntTensor(len(batch))
        for ind in range(len(batch)):
            tokens = batch[token_ind_sorted[ind]][0]
            token_padded[ind, : tokens.shape[0]] = tokens
            token_lengths[ind] = tokens.shape[0]

        n_mels = batch[0][1].shape[0] # mel: n_mels x frames
        max_mel_length = max([x[1].shape[1] for x in batch])
        mel_padded = torch.FloatTensor(len(batch), n_mels, max_mel_length)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_mel_length)
        gate_padded.zero_()
        mel_lengths = torch.IntTensor(len(batch))
        for ind in range(len(batch)):
            mel = batch[token_ind_sorted[ind]][1]
            mel_padded[ind, :, : mel.shape[1]] = mel
            # marking which is the last step
            # for largest -> 000001 (1 in last frame marks that this is the end which is used in decoding inference)
            gate_padded[ind, mel.shape[1]-1: ] = 1
            mel_lengths[ind] = mel.shape[1]

        return token_padded, token_lengths, mel_padded, gate_padded, mel_lengths


class CheckpointManager:
    def __init__(self, max_best_models, exp_dir):
        self.max_best_models = max_best_models
        self.exp_dir = exp_dir
        assert os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory does not exist"
        self.manager_path = os.path.join(self.exp_dir, "checkpoint_manager.json")
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
                "architecture": trainer.model_config.model_architecture,
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
        tacotron2_config: Tacotron2Config,
        optimizer_config: OptimizerConfig
    ):
        self.exp_dir = "exp"
        assert not os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory already exists"
        os.mkdir(self.exp_dir)
        self.config_yaml_path = os.path.join(self.exp_dir, "config.yaml")
        write_file_from_config(self.config_yaml_path, text_config, audio_config, dataset_config, trainer_config, tacotron2_config, optimizer_config)

        self.config = trainer_config
        self.model_config = tacotron2_config
        self.optimizer_config = optimizer_config
        self.audio_config = audio_config

        if (self.config.use_cuda):
            assert torch.cuda.is_available(), "torch CUDA is not available"
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.collate_fn = TextMelCollate()
        self.train_dataset = TextMelDataset(dataset_split_type="train")
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.batch_size, collate_fn=self.collate_fn)

        if (self.config.run_validation):
            print("run_validation is set to True")
            self.validation_dataset = TextMelDataset(dataset_split_type="validation")
            assert len(self.validation_dataset) > 0, "run_validation was set True, but validation data was not found"
            self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.validation_batch_size, collate_fn=self.collate_fn, drop_last=True)
            self.validation_dir = os.path.join(self.exp_dir, "validation_runs")
            os.mkdir(self.validation_dir)

        self.checkpoint_manager = CheckpointManager(self.config.max_best_models, self.exp_dir)
        if (self.config.wandb_logger):
            self.wandb = WandbLogger(self)

    def prepare_for_training(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        print(self.config)
        print(self.model_config)
        print(self.optimizer_config)
        print(self.audio_config)
        self.model = tacotron2.Tacotron2(self.model_config, self.audio_config, self.config.use_cuda)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_config.learning_rate, weight_decay=self.optimizer_config.weight_decay)
        self.start_iteration = 0
        # checkpoint loading for resuming the training
        if (self.config.resume_from_checkpoint):
            checkpoint_data = torch.load(self.config.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.start_iteration = checkpoint_data['iteration']
        self.criterion = tacotron2.Tacotron2Loss()
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
        os.mkdir(validation_run_path)
        rand_ind = random.randrange(0, len(batch)) # selecting random sample in batch
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
            mel_target_img = self.wandb.Image(validation_run_mel_tar_path, caption='mel target')
            mel_predicted_img = self.wandb.Image(validation_run_mel_pred_path, caption='mel predicted')
            self.wandb.log({'mel_plots': [mel_target_img, mel_predicted_img]}, epoch=iteration)
            gate_img = self.wandb.Image(validation_run_gate_path, caption='gate')
            self.wandb.log({'gate_plot': gate_img}, epoch=iteration)
            align_img = self.wandb.Image(validation_run_align_path, caption='alignment')
            self.wandb.log({'alignment_plot': align_img}, epoch=iteration)
        self.model.train()

    def train(self):
        self.prepare_for_training()
        print(f"Project: {self.config.project_name}")
        print(f"Experiment: {self.config.experiment_id}")
        center_print(f"TRAINING START ({current_formatted_time()})", space_factor=0.35)
        start_train = time.time() # start time of training
        log_print(f"epochs: {self.config.epochs}, batch_count: {len(self.train_dataloader)}, device: {self.device}")
        iteration = self.start_iteration
        avg_time_epoch = 0
        for epoch in range(self.config.epochs):
            start_epoch = time.time() # start time of epoch
            log_print(f"epoch start: {epoch + 1} / {self.config.epochs}")
            for ind, batch in enumerate(self.train_dataloader):
                start_iter = time.time() # start time of iteration
                # token_padded, token_lengths, mel_padded, gate_padded, mel_lengths = batch
                # check for manual reset of learning rate in optimizer param_groups
                for param_group in self.optimizer.param_groups:
                    # param_group['lr'] = self.optimizer_config.learning_rate
                    print(param_group)
                    print(param_group['lr'])
                self.optimizer.zero_grad() # same as self.model.zero_grad()
                x, y = self.model.parse_batch(batch)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss_value = loss.item()
                loss.backward()
                # check for torch.nn.utils.clip_grad_norm_()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimizer_config.grad_clip_thresh)
                self.optimizer.step()
                end_iter = time.time() # end time of iteration
                log_print(f"({epoch + 1} :: {ind + 1} / {len(self.train_dataloader)}) -> iteration: {iteration}, loss: {loss_value: .3f}, grad_norm: {grad_norm: .3f}, time_taken: {end_iter - start_iter: .2f} s")
                iteration += 1

                if (iteration % self.config.iters_for_checkpoint == 0):
                    # checkpoint saving
                    self.checkpoint_manager.save_model(iteration, self.model, loss_value)
                    # validation
                    self.validation(iteration)
                # logging
                if (self.config.wandb_logger):
                    self.wandb.log({'loss': loss_value}, epoch=iteration, commit=True)
                if (iteration == 3):
                    break
            end_epoch = time.time() # end time of epoch
            epoch_time = end_epoch - start_epoch
            avg_time_epoch = (avg_time_epoch * epoch + epoch_time) / (epoch + 1)  # update the average epoch time
            log_print(f"epoch end -> time_taken: {sec_to_formatted_time(end_epoch - start_epoch)}")
            # calculate ETA
            remaining_epochs = self.config.epochs - (epoch + 1)
            remaining_time = remaining_epochs * avg_time_epoch
            log_print(f"estimated time remaining: {sec_to_formatted_time(remaining_time)} -> training ending at {current_formatted_time(remaining_time)}")
            print()
        if (self.config.wandb_logger):
            self.wandb.finish()
        center_print(f"TRAINING END ({current_formatted_time()})", space_factor=0.35)
        end_train = time.time() # end time of training
        print(f"total time of training: {sec_to_formatted_time(end_train - start_train)}")


if __name__ == "__main__":
    twd = TextMelDataset()
    print(twd[0])
