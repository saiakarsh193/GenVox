from utils.formatters import BaseDataset
from configs import TextConfig, AudioConfig, TrainerConfig
from configs.models import Tacotron2Config
from models.tts.tacotron2 import Tacotron2
from core.processors import DataPreprocessor
from core.trainer import Trainer

dataset = BaseDataset(
    dataset_path="data/LJSpeech_test",
    formatter="ljspeech",
    dataset_name="ljspeech"
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
    datasets=[dataset],
    text_config=text_config,
    audio_config=audio_config,
    eval_split=500,
    dump_dir="dump"
)
data_preprocessor.run()

trainer_config = TrainerConfig(
    project_name="genvox_revamp",
    experiment_id="exp_1",
    notes="First Revamp run",
    use_cuda=True,
    epochs=20,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=0,
    iters_for_checkpoint=1000,
    max_best_models=5,
    run_eval=False,
    use_wandb=True,
    # debug_run=True
)

tacotron2 = Tacotron2(
    model_config=Tacotron2Config(),
    audio_config=audio_config,
    text_config=text_config,
    trainer_config=trainer_config
)

trainer = Trainer(
    model=tacotron2,
    trainer_config=trainer_config,
    text_config=text_config,
    audio_config=audio_config,
    dump_dir="dump",
    exp_dir="exp"
)

trainer.run()
