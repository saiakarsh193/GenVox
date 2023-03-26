import os

from config import DatasetConfig, AudioConfig
from processors import DatasetProcessor


dataset_path = "data/"

dataset_config = DatasetConfig(
    delimiter="|",
    transcript_path=os.path.join(dataset_path, "transcript")
)

dataset_processor = DatasetProcessor(dataset_config)
dataset_processor()