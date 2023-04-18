import os
import random
import numpy as np
from typeguard import typechecked
import torch
import torch.utils.data
import wandb

from config import TrainerConfig, Tacotron2Config, OptimizerConfig, AudioConfig
from utils import load_json, dump_json
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
    def __init__(self, max_best_models):
        self.max_best_models = max_best_models
        self.exp_dir = "exp"
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
            torch.save(model.state_dict(), checkpoint_path)
        if (len(manager_data) > self.max_best_models):
            model_removed_path = manager_data[-1][0]
            manager_data = manager_data[: -1]
            os.remove(model_removed_path)
        dump_json(self.manager_path, manager_data)


class WandbLogger:
    def __init__(self, auth_key, project_name, experiment_id, architecture):
        self.auth_key = auth_key
        self.project_name = project_name
        self.experiment_id = experiment_id
        wandb.login(key=self.auth_key)
        wandb.init(
            project=self.project_name,
            name=f"experiment_{self.experiment_id}",
            config={
                "architecture": architecture
            }
        )

    def define_metric(self, value, summary):
        wandb.define_metric(value, summary=summary)

    def log(self, values, epoch):
        wandb.log(values, step=epoch, commit=True)

    def finish(self):
        wandb.finish()


class Trainer:
    """
    Trainer class for the data loading and training stage, along with checkpointing and logging
    """
    @typechecked
    def __init__(self, config: TrainerConfig, model_config: Tacotron2Config, optimizer_config: OptimizerConfig, audio_config: AudioConfig):
        exp_dir = "exp"
        assert not os.path.isdir(exp_dir), f"experiments ({exp_dir}) directory already exists"
        os.mkdir(exp_dir)
        self.config = config
        print(self.config)
        self.model_config = model_config
        print(self.model_config)
        self.optimizer_config = optimizer_config
        print(self.optimizer_config)
        self.audio_config = audio_config
        print(self.audio_config)

        if (self.config.use_cuda):
            assert torch.cuda.is_available(), "torch CUDA is not available"
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.checkpoint_manager = CheckpointManager(self.config.max_best_models)
        if (self.config.wandb_logger):
            self.wandb = WandbLogger(self.config.wandb_auth_key, self.config.project_name, self.config.experiment_id, self.config.model_architecture)

        self.collate_fn = TextMelCollate()
        self.train_dataset = TextMelDataset(dataset_split_type="train")
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.batch_size, collate_fn=self.collate_fn)

        if (self.config.run_validation):
            print("run_validation is set to True")
            self.validation_dataset = TextMelDataset(dataset_split_type="validation")
            assert len(self.validation_dataset) > 0, "run_validation was set True, but validation data was not found"
            self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.validation_batch_size, collate_fn=self.collate_fn)

    def prepare_for_training(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.model = tacotron2.Tacotron2(self.model_config, self.audio_config, self.config.use_cuda)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_config.learning_rate, weight_decay=self.optimizer_config.weight_decay)
        self.criterion = tacotron2.Tacotron2Loss()
        self.model.train()
        print(self.model)
        if (self.config.wandb_logger):
            self.wandb.define_metric('loss', summary='min')

    def train(self):
        print(f"Project: {self.config.project_name}")
        print(f"Experiment: {self.config.experiment_id}")
        self.prepare_for_training()
        print(f"training is starting (epochs: {self.config.epochs}, batch_count: {len(self.train_dataloader)}, device: {self.device})")
        iteration = 0
        for epoch in range(self.config.epochs):
            print(f"training loop epoch: {epoch + 1} / {self.config.epochs}")
            for ind, batch in enumerate(self.train_dataloader):
                # token_padded, token_lengths, mel_padded, gate_padded, mel_lengths = batch
                # check for manual reset of learning rate in optimizer param_groups
                self.optimizer.zero_grad() # same as self.model.zero_grad()
                x, y = self.model.parse_batch(batch)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss_value = loss.item()
                loss.backward()
                # check for torch.nn.utils.clip_grad_norm_()
                self.optimizer.step()
                print(f"epoch: ({epoch}/{self.config.epochs}::{ind}), iteration: {iteration}, loss: {loss_value}")
                iteration += 1

                # logging
                if (self.config.wandb_logger):
                    self.wandb.log({'loss': loss_value}, epoch=iteration)
                # checkpoint saving
                self.checkpoint_manager.save_model(iteration, self.model, loss_value)
                # validation
        if (self.config.wandb_logger):
            self.wandb.finish()


if __name__ == "__main__":
    twd = TextMelDataset()
    print(twd[0])
