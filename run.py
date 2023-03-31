import os

from config import DownloadConfig, DatasetConfig, AudioConfig
from processors import DownloadProcessor, DatasetProcessor, AudioProcessor

# dataset_path = "data/LJSpeech_sample"
dataset_path = "data/youtube_3b1b"

download_config = DownloadConfig(
    is_youtube=True,
    youtube_link="https://www.youtube.com/watch?v=fRed0Xmc2Wg",
    directory_path=dataset_path,
    create_directory=True,
    speaker_id="3B1B"
)

download_processor = DownloadProcessor(download_config)
download_processor()

dataset_config = DatasetConfig(
    delimiter="|",
    transcript_path=os.path.join(dataset_path, "transcript.txt"),
    # transcript_path=os.path.join(dataset_path, "metadata.csv"),
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
