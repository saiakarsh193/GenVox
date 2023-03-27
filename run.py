import os

from config import DatasetConfig, AudioConfig
from processors import DatasetProcessor, AudioProcessor


dataset_path = "data/LJSpeech_sample"

dataset_config = DatasetConfig(
    delimiter="|",
    transcript_path=os.path.join(dataset_path, "metadata.csv"),
    wavs_path=os.path.join(dataset_path, "wavs")
)

audio_config = AudioConfig(
    sampling_rate=16000,
    trim_silence=False
)

dataset_processor = DatasetProcessor(dataset_config)
dataset_processor()

audio_processor = AudioProcessor(audio_config)
audio_processor()