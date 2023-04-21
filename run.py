import os

from config import DownloadConfig, TextConfig, AudioConfig, DatasetConfig, TrainerConfig, Tacotron2Config, OptimizerConfig
from processors import DownloadProcessor, DatasetProcessor
from trainer import Trainer

dataset_path = "data/LJSpeech-1.1"
# dataset_path = "data/LJSpeech_test"
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

text_config = TextConfig(
    language="english",
    cleaners=["base_cleaners"],
    use_g2p=True
)

audio_config = AudioConfig(
    sampling_rate=22050,
    trim_silence=True,
    trim_dbfs=-50, # anything below -50 is considered silent
    min_wav_duration=2,
    max_wav_duration=10,
    normalize=True,
    filter_length=1024,
    hop_length=256,
    n_mels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0,
    log_func="np.log",
    ref_level_db=20
)

dataset_config = DatasetConfig(
    text_config=text_config,
    audio_config=audio_config,
    delimiter="|",
    # transcript_path=os.path.join(dataset_path, "transcript.txt"),
    transcript_path=os.path.join(dataset_path, "metadata.csv"),
    wavs_path=os.path.join(dataset_path, "wavs"),
    remove_wav_dump=True,
    validation_split=500
)

# dataset_processor = DatasetProcessor(dataset_config)
# dataset_processor()

trainer_config = TrainerConfig(
    project_name="dev_run_ada",
    experiment_id="ljspeech_fullrun_3",
    wandb_logger=True,
    wandb_auth_key="56acc87c7b95662ff270b9556cdf68de699a210f",
    batch_size=32,
    num_loader_workers=0,
    run_validation=True,
    use_cuda=True,
    epochs=100,
    iters_for_checkpoint=500
)

tacotron2_config = Tacotron2Config()
optimizer_config = OptimizerConfig()

trainer = Trainer(text_config, audio_config, dataset_config, trainer_config, tacotron2_config, optimizer_config)
trainer.train()
