import os

from config import DownloadConfig, TextConfig, AudioConfig, DatasetConfig, write_file_from_config
from processors import DownloadProcessor, DatasetProcessor

dataset_path = "data/LJSpeech_test"
# dataset_path = "data/youtube_3b1b"

# download_config = DownloadConfig(
#     is_youtube=True,
#     youtube_link="https://www.youtube.com/watch?v=fRed0Xmc2Wg",
#     directory_path=dataset_path,
#     create_directory=True,
#     speaker_id="3B1B"
# )

# download_processor = DownloadProcessor(download_config)
# download_processor()

text_config = TextConfig()

audio_config = AudioConfig(
    sampling_rate=16000,
    trim_silence=True,
    trim_dbfs=-50, # anything below -50 is considered silent
    min_wav_duration=2,
    max_wav_duration=10,
    normalize=True,
    filter_length=1024,
    hop_length=256,
    n_mels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0
)

dataset_config = DatasetConfig(
    text_config=text_config,
    audio_config=audio_config,
    delimiter="|",
    # transcript_path=os.path.join(dataset_path, "transcript.txt"),
    transcript_path=os.path.join(dataset_path, "metadata.csv"),
    wavs_path=os.path.join(dataset_path, "wavs"),
)

# write_file_from_config('config.json', text_config, audio_config, dataset_config)

dataset_processor = DatasetProcessor(dataset_config)
dataset_processor()
