import os
import numpy as np
from typeguard import typechecked
import torch
import torch.utils.data

from config import TrainerConfig

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


class Trainer:
    """
    Trainer class for the data loading and training stage, along with checkpointing and logging
    """
    @typechecked
    def __init__(self, config: TrainerConfig):
        exp_dir = "exp"
        assert not os.path.isdir(exp_dir), f"experiments ({exp_dir}) directory already exists"
        os.mkdir(exp_dir)
        self.config = config
        print(self.config)
        self.collate_fn = TextMelCollate()
        self.train_dataset = TextMelDataset(dataset_split_type="train")
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.batch_size, collate_fn=self.collate_fn)
        if (self.config.run_validation):
            print("run_validation is set to True")
            self.validation_dataset = TextMelDataset(dataset_split_type="validation")
            assert len(self.validation_dataset) > 0, "run_validation was set True, but validation data was not found"
            self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, num_workers=self.config.num_loader_workers, shuffle=True, batch_size=self.config.validation_batch_size, collate_fn=self.collate_fn)

    def train(self):
        print(f"training is starting (epochs: {self.config.epochs})")
        for epoch in range(self.config.epochs):
            print(f"training loop epoch: {epoch + 1} / {self.config.epochs}")
            for ind, batch in enumerate(self.train_dataloader):
                print(ind)
                token_padded, token_lengths, mel_padded, gate_padded, mel_lengths = batch
                print(token_padded.shape, token_lengths.shape, mel_padded.shape, gate_padded.shape, mel_lengths.shape)
                # model forward
                # model backward
                # params optim and grad zero
                # logger
            break


if __name__ == "__main__":
    twd = TextMelDataset()
    print(twd[0])
