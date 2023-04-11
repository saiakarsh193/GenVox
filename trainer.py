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
    def __init__(self):
        pass


class Trainer:
    """
    Trainer class for the data loading and training stage, along with checkpointing and logging
    """
    @typechecked
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.train_dataset = TextMelDataset(dataset_split_type="train")
        if (self.config.run_validation):
            self.validation_dataset = TextMelDataset(dataset_split_type="validation")
            assert len(self.validation_dataset) > 0, "run_validation was set True, but validation data was not found"
        self.collate_fn = TextMelCollate()
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            num_workers=self.config.num_loader_workers,
                                                            shuffle=True,
                                                            batch_size=self.config.batch_size,
                                                            # collate_fn=self.collate_fn
                                                            )
        if (self.config.run_validation):
            self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset,
                                                            num_workers=self.config.num_loader_workers,
                                                            shuffle=True,
                                                            batch_size=self.config.validation_batch_size,
                                                            # collate_fn=self.collate_fn
                                                            )

    def train(self):
        for ind, batch in enumerate(self.train_dataloader):
            print(ind)
            print(batch)
            print(batch.shape)
            break


if __name__ == "__main__":
    twd = TextMelDataset()
    print(twd[0])
