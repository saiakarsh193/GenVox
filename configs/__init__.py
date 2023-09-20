from __future__ import annotations
import os
from typing import Optional, Union, List, Literal, Dict
from utils import get_random_HEX_name, dump_json, dump_yaml, load_json, load_yaml
from utils.text import _CLEANER_TYPES

def check_argument(name: str, value: Union[float, int], min_val: Optional[Union[float, int]] = None, max_val: Optional[Union[float, int]] = None) -> None:
    if (min_val == None):
        assert (value <= max_val), f"The value \'{name}\' ({value}) is above max_val ({max_val})."
    elif (max_val == None):
        assert (value >= min_val), f"The value \'{name}\' ({value}) is below min_val ({min_val})."
    else:
        assert (value >= min_val and value <= max_val), f"The value \'{name}\' ({value}) is not in the required range ({min_val} -> {max_val})."


class BaseConfig:
    def __str__(self, level: int = 0) -> str:
        prefix_main = (("├── " + "    " * (level - 1)) if level > 0 else "")
        prefix = (("│   " + "    " * (level - 1)) if level > 0 else "")
        rstr = prefix_main + self.__class__.__name__ + "\n"
        for ind, (key, value) in enumerate(self.__dict__.items()):
            if (isinstance(value, BaseConfig)):
                svalue = "\n" + BaseConfig.__str__(value, level=level + 1)
            else:
                svalue = "(" + str(value) + ")"
            prefix_value = ("└── " if ind == len(self.__dict__) - 1 else "├── ")
            rstr += prefix + prefix_value + key.ljust(35) + svalue + "\n"
        return rstr.rstrip()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def _get_dict_from_config(config: BaseConfig) -> Dict:
        config_dict = {}
        for attr, value in vars(config).items():
            if not isinstance(value, BaseConfig):
                config_dict[attr] = value
        return config_dict

    @staticmethod
    def write_configs_to_file(path: str, configs: Dict[str, Optional[BaseConfig]]) -> None:
        config_ext = os.path.splitext(path)[1][1: ]
        assert config_ext in ["json", "yaml"], f"given config extension ({config_ext}) is invalid"
        config_dict = {}
        for config_name, config in configs.items():
            if config != None:
                config_dict[config_name] = BaseConfig._get_dict_from_config(config=config)
                config_dict[config_name]["CONFIG_TYPE"] = config.__class__.__name__
        if (config_ext == "json"):
            dump_json(path, config_dict)
        elif (config_ext == "yaml"):
            dump_yaml(path, config_dict)

    @staticmethod
    def load_configs_from_file(path: str) -> Dict[str, BaseConfig]:
        config_ext = os.path.splitext(path)[1][1: ]
        assert config_ext in ["json", "yaml"], f"given config extension ({config_ext}) is invalid"
        if (config_ext == "json"):
            config_dict = load_json(path)
        elif (config_ext == "yaml"):
            config_dict = load_yaml(path)
        CONFIG_MAP: Dict[str, BaseConfig] = {
            'TextConfig': TextConfig,
            'AudioConfig': AudioConfig
        }
        configs = {}
        for config_name in config_dict:
            config_type = config_dict[config_name]["CONFIG_TYPE"]
            del config_dict[config_name]["CONFIG_TYPE"]
            configs[config_name] = CONFIG_MAP[config_type](**config_dict[config_name])
        return configs


class TextConfig(BaseConfig):
    """
    Config for TextProcessor

    Args:
        language (str): The language of the text. 
                Default is ``"english"``.
        cleaners (Optional[List[_CLEANER_TYPES]]): A list of cleaner names or None. 
                Default is ``None``.
        use_g2p (bool): Whether to use grapheme-to-phoneme (g2p) conversion. 
                Default is ``False``.
    """
    def __init__(
        self,
        language: str = "english",
        cleaners: Optional[List[_CLEANER_TYPES]] = None,
        use_g2p: bool = False,
        token_map: Optional[Dict[str, int]] = None
    ):
        self.language = language.lower()
        self.cleaners = cleaners
        self.use_g2p = use_g2p
        self.token_map = token_map

_LOG_TYPE = Literal[
    "np.log",
    "np.log10"
]

class AudioConfig(BaseConfig):
    """
    Config for AudioProcessor

    Args:
        Args:
        sampling_rate (int): The sampling rate of the audio in Hz. (Min: 16000, Max: 44100)
                Default is ``22050``.
        trim_silence (bool): Whether to trim silence from the beginning and end of the audio.
                Default is ``True`.
        trim_dbfs (float): The threshold in dBFS below which audio is considered silence for trimming. (Min: -100, Max: 0)
                Default is ``-50.0``.
        min_wav_duration (float): The minimum duration of audio in seconds. Audio shorter than this duration will be excluded. (Min: 0.1)
                Default is ``0.5``.
        max_wav_duration (float): The maximum duration of audio in seconds. Audio longer than this duration will be excluded.
                Default is ``10``.
        normalize (bool): Whether to normalize the audio waveform. 
                Default is ``True``.
        filter_length (int): The length of the FIR filter for computing the STFT. (Min: 256, Max: 2048)
                Default is ``512``.
        hop_length (int): The number of samples between successive STFT columns. (Min: 128)
                Default is ``256``.
        n_mels (int): The number of mel bands to generate. (Min value: 12, Max value: 128)
                Default is ``80``.
        mel_fmin (float): The minimum frequency of the mel band filter bank. (Min: 0, Max: 8000)
                Default is ``0.0``.
        mel_fmax (float): The maximum frequency of the mel band filter bank. (Min: 8000, Max:22050)
                Default is ``8000.0``.
        log_func (str): The logarithm function to use for mel spectrogram computation. (Type: _LOG_TYPE)
                Default is ``"np.log10"``.
        ref_level_db (float): The reference level dB for normalizing the audio. (Min: 1)
                Default is ``1.0``.

    Raises:
        ValueError: If any of the arguments fails the specified constraints.
    """
    def __init__(
        self,
        sampling_rate: int = 22050,
        trim_silence: bool = True,
        trim_dbfs: float = -50.0,
        min_wav_duration: float = 0.5,
        max_wav_duration: float = 10,
        normalize: bool = True,
        filter_length: int = 512,
        hop_length: int = 256,
        n_mels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
        log_func: _LOG_TYPE = "np.log10",
        ref_level_db: float = 1.0
    ):
        self.sampling_rate = sampling_rate
        self.trim_silence = trim_silence
        self.trim_dbfs = trim_dbfs # decibels relative to full scale
        self.min_wav_duration = min_wav_duration
        self.max_wav_duration = max_wav_duration
        self.normalize = normalize
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.log_func = log_func
        self.ref_level_db = ref_level_db

        check_argument("sampling_rate", self.sampling_rate, min_val=16000, max_val=44100)
        check_argument("trim_dbfs", self.trim_dbfs, min_val=-100, max_val=0)
        check_argument("min_wav_duration", self.min_wav_duration, min_val=0.1)
        check_argument("max_wav_duration", self.max_wav_duration, min_val=self.min_wav_duration)
        check_argument("filter_length", self.filter_length, min_val=256, max_val=2048)
        check_argument("hop_length", self.hop_length, min_val=128, max_val=self.filter_length)
        check_argument("n_mels", self.n_mels, min_val=12, max_val=128)
        check_argument("mel_fmin", self.mel_fmin, min_val=0, max_val=8000)
        check_argument("mel_fmax", self.mel_fmax, min_val=8000, max_val=22050)
        check_argument("ref_level_db", self.ref_level_db, min_val=1)

class TrainerConfig(BaseConfig):
    """
    Config for Trainer class

    Args:
        project_name (str): The name of the project
                Default is ``""``.
        experiment_id (str): The ID of the experiment.
                Default is ``""``.
        notes (Optional[str]): Additional notes or description for the experiment. 
                Default is ``None``.
        batch_size (int): The batch size for training. 
                Default is ``64``.
        num_loader_workers (int): The number of worker processes for data loading. 
                Default is ``2``.
        run_validation (bool): Whether to run validation during training. 
                Default is ``True``.
        validation_batch_size (int): The batch size for validation. 
                Default is ``32``.
        epochs (int): The number of training epochs. 
                Default is ``100``.
        seed (int): The random seed for reproducibility. 
                Default is ``0``.
        use_cuda (bool): Whether to use CUDA for training if available. 
                Default is ``False``.
        max_best_models (int): The maximum number of best models to save during training. 
                Default is ``5``.
        iters_for_checkpoint (int): The number of iterations between saving checkpoints. 
                Default is ``1``.
        save_optimizer_dict (bool): Whether to save the optimizer state dictionary in checkpoints. 
                Default is ``False``.
        wandb_logger (bool): Whether to use wandb for logging. 
                Default is ``True``.
        resume_from_checkpoint (bool): Whether to resume training from a checkpoint. 
                Default is ``False``.
        checkpoint_path (str): The path to the checkpoint file for resuming training. 
                Default is an ``""``.
        epoch_start (int): The starting epoch count. 
                Default is ``1``.
        exp_dir (str): The directory for storing experiment results.
                Default is ``"exp"``.
        dump_dir (str): The directory for storing dumped files.
                Default is ``"dump"``.
    Raises:
        AssertionError: If `project_name` is not provided.
            If `experiment_id` is not provided, a random ID will be generated.
            If `epochs`, `max_best_models`, or `iters_for_checkpoint` are less than 1.
            If `resume_from_checkpoint` is True and `checkpoint_path` is not provided.
            If `epoch_start` is less than 1.
    """
    def __init__(
        self,
        project_name: str = "",
        experiment_id: str = "",
        notes: Optional[str] = None,
        batch_size: int = 64,
        num_loader_workers: int = 2,
        run_validation: bool = True,
        validation_batch_size: int = 32,
        epochs: int = 100,
        seed: int = 0,
        use_cuda: bool = False,
        max_best_models: int = 5,
        iters_for_checkpoint: int = 1,
        save_optimizer_dict: bool = False,
        wandb_logger: bool = True,
        resume_from_checkpoint: bool = False,
        checkpoint_path: str = "",
        epoch_start: int = 1,
        exp_dir: str = "exp",
        dump_dir: str = "dump",
    ):
        self.exp_dir = exp_dir
        self.dump_dir = dump_dir
        self.project_name = project_name
        self.experiment_id = experiment_id
        self.notes = notes
        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers # for DataLoader(num_workers=___) in torch.utils.data
        self.run_validation = run_validation
        self.validation_batch_size = validation_batch_size
        self.epochs = epochs
        self.seed = seed
        self.use_cuda = use_cuda
        self.max_best_models = max_best_models
        self.iters_for_checkpoint = iters_for_checkpoint
        self.save_optimizer_dict = save_optimizer_dict
        self.wandb_logger = wandb_logger
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_path = checkpoint_path
        self.epoch_start = epoch_start # 1-indexed epoch counter

        assert self.project_name, "project_name not provided"
        if self.experiment_id == "":
            self.experiment_id = get_random_HEX_name(40)
            print(f"experiment_id not provided, hence randomly generated ({self.experiment_id})")
        check_argument("epochs", self.epochs, min_val=1)
        check_argument("max_best_models", self.max_best_models, min_val=1, max_val=10)
        check_argument("iters_for_checkpoint", self.iters_for_checkpoint, min_val=1)
        if (self.resume_from_checkpoint):
            assert self.checkpoint_path, "checkpoint_path not provided (resume_from_checkpoint has been enabled) to start training"
        check_argument("epoch_start", self.epoch_start, min_val=1)
