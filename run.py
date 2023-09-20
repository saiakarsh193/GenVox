# import os
from utils.formatters import BaseDataset
from configs import TextConfig, AudioConfig
from core.processors import DataPreprocessor

# from config import DownloadConfig, TextConfig, AudioConfig, DatasetConfig, TrainerConfig, Tacotron2Config, OptimizerConfig, MelGANConfig
# from processors import DownloadProcessor, DatasetProcessor
# from trainer import Trainer

dataset = BaseDataset(
    dataset_path="data/LJSpeech_test",
    formatter="ljspeech",
    dataset_name="u1"
)

dataset2 = BaseDataset(
    dataset_path="data/LJSpeech_small",
    formatter="ljspeech",
    dataset_name="u2"
)

text_config = TextConfig(
    language="english",
    cleaners=["base_cleaners"],
    use_g2p=False
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
    ref_level_db=1.0
)

data_preprocessor = DataPreprocessor(
    datasets=[dataset2],
    # datasets=[dataset, dataset2],
    text_config=text_config,
    audio_config=audio_config,
    validation_split=500,
    dump_dir="dump"
)
data_preprocessor.run()

exit()

trainer_config = TrainerConfig(
    project_name="dev_run_ada",
    experiment_id="run_10",
    notes="First Vocoder run",
    wandb_logger=True,
    batch_size=16,
    validation_batch_size=16,
    num_loader_workers=0,
    run_validation=True,
    use_cuda=True,
    epochs=200,
    max_best_models=5,
    iters_for_checkpoint=1000,
    dump_dir="dump",
    exp_dir="exp"
)

melgan_config = MelGANConfig()
# tacotron2_config = Tacotron2Config()

optimizer_config = OptimizerConfig(
    learning_rate=0.0001,
    beta1=0.5,
    beta2=0.9,
    weight_decay=0
)

trainer = Trainer(
    trainer_config=trainer_config,
    model_config=melgan_config,
    # model_config=tacotron2_config,
    optimizer_config=optimizer_config,
    audio_config=audio_config,
    # text_config=text_config,
    # dataset_config=dataset_config
)

trainer.train()
