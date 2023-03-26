import os

from config import DatasetConfig, AudioConfig
from processors import DatasetProcessor


dataset_path = "data/LJSpeech_sample"

dataset_config = DatasetConfig(
    delimiter="|",
    transcript_path=os.path.join(dataset_path, "metadata.csv"),
    wavs_path=os.path.join(dataset_path, "wavs")
)

dataset_processor = DatasetProcessor(dataset_config)
dataset_processor()