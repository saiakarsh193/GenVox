import os
from typing import List, Union, Dict, Optional
from typing_extensions import Literal
from typeguard import typechecked

from utils import dump_json, load_json, dump_yaml, load_yaml, get_random_HEX_name

def check_argument(name, value, min_val=None, max_val=None):
    if (min_val == None):
        assert (value <= max_val), f"The value \'{name}\' ({value}) is above max_val ({max_val})."
    elif (max_val == None):
        assert (value >= min_val), f"The value \'{name}\' ({value}) is below min_val ({min_val})."
    else:
        assert (value >= min_val and value <= max_val), f"The value \'{name}\' ({value}) is not in the required range ({min_val} -> {max_val})."


class BaseConfig:
    def __str__(self, level = 0):
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
    
    def __repr__(self):
        return str(self)


class DownloadConfig(BaseConfig):
    """
    Config for DownloadProcessor

    Args:
        download_link (str): A string containing a direct download link for the audio file.
                Default is ``""``.
        is_youtube (bool): A boolean indicating whether the audio is from a YouTube video.
                Default is ``False``.
        youtube_link (str): A string containing the YouTube video link (is_youtube should be True).
                Default is ``""``.
        speaker_id (str): The ID of the speaker in the audio file.
                Default is ``""``.
        create_directory (bool): Whether to create a new directory for the downloaded file.
                Default is ``False``.
        directory_path (str): The path of the directory where the downloaded file will be saved.
                Default is ``""``.
        verbose (bool): Whether to print messages during the download process.
                Default is ``True``.
    Raises:
        AssertionError: If `youtube_link` or `speaker_id` are not given when `is_youtube` is True,
            If `download_link` is not given when `is_youtube` is False.
            If `create_directory` is True and `directory_path` already exists,
            If `create_directory` is False and `directory_path` does not exist.
    """
    @typechecked
    def __init__(
        self,
        download_link: str = "",
        is_youtube: bool = False,
        youtube_link: str = "",
        speaker_id: str = "",
        create_directory: bool = False,
        directory_path: str = "",
        verbose: bool = True
    ):
        self.download_link = download_link
        self.is_youtube = is_youtube
        self.youtube_link = youtube_link
        self.speaker_id = speaker_id
        self.create_directory = create_directory
        self.directory_path = directory_path
        self.verbose = verbose

        if (self.is_youtube):
            assert self.youtube_link, "youtube_link not given"
            assert self.speaker_id, "speaker_id not given"
        else:
            assert self.download_link, "download_link not given"
        
        if (self.create_directory):
            assert not os.path.isdir(self.directory_path), f"directory_path ({self.directory_path}) already exists"
            os.mkdir(self.directory_path)
        else:
            assert os.path.isdir(self.directory_path), f"directory_path ({self.directory_path}) does not exist"


class TextConfig(BaseConfig):
    """
    Config for TextProcessor

    Args:
        language (str): The language of the text. 
                Default is ``"english"``.
        cleaners (Union[List[str], None]): A list of cleaner names or None. 
                Default is ``None``.
        use_g2p (bool): Whether to use grapheme-to-phoneme (g2p) conversion. 
                Default is ``False``.
    """
    @typechecked
    def __init__(
        self,
        language: str = "english",
        cleaners: Union[List[str], None] = None,
        use_g2p: bool = False
    ):
        self.language = language.lower()
        self.cleaners = cleaners
        self.use_g2p = use_g2p

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
        log_func (str): The logarithm function to use for mel spectrogram computation. (Either "np.log10" or "np.log")
                Default is ``"np.log10"``.
        ref_level_db (float): The reference level dB for normalizing the audio. (Min: 1)
                Default is ``1``.

    Raises:
        ValueError: If any of the arguments fails the specified constraints.
    """
    @typechecked
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
        ref_level_db: float = 1
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


_DATASET_TYPE = Literal[
    "text",
    "json"
]

class DatasetConfig(BaseConfig):
    """
    Config for DatasetProcessor

    Args:
        text_config (TextConfig): Configuration settings for text processing.
        audio_config (AudioConfig): Configuration settings for audio processing.
        dataset_type (str): The type of the dataset. (Either "text" or "json")
                Default is ``"text"``. 
        delimiter (str): The delimiter used to separate fields in the dataset. 
                Default is ``" "``.
        uid_index (int): The index of the field containing the unique identifier in the dataset. (Min: 0)
                Default is ``0``.
        utt_index (int): The index of the field containing the utterance in the dataset. (Min: 0)
                Default is ``1``.
        uid_keyname (str): The key value containing the unique identifier in the dataset.
                Default is ``None``.
        utt_keyname (str): The key value containing the utterance in the dataset.
                Default is ``None``.
        transcript_path (str): The path to the transcript file for text-based datasets. 
                Default is ``""``.
        wavs_path (str): The path to the directory containing the audio files for audio-based datasets. 
                Default is ``""``.
        validation_split (Union[int, float]): The number of samples to use for validation. 
                Default is ``0``.
        dump_dir (str): The directory to store the prepared dataset. 
                Default is ``"dump"``.
        
    Raises:
        ValueError: If any of the arguments fails the specified constraints.
    """
    @typechecked
    def __init__(
        self,
        text_config: TextConfig,
        audio_config: AudioConfig,
        dataset_type: _DATASET_TYPE = "text",
        delimiter: str = " ",
        uid_index: int = 0,
        utt_index: int = 1,
        uid_keyname: Optional[str] = None,
        utt_keyname: Optional[str] = None,
        transcript_path: str = "",
        wavs_path: str = "",
        validation_split: Union[int, float] = 0,
        dump_dir: str = "dump",
    ):
        self.dump_dir = dump_dir
        self.text_config = text_config
        self.audio_config = audio_config
        self.dataset_type = dataset_type
        self.delimiter = delimiter
        self.uid_index = uid_index # used when type -> text
        self.utt_index = utt_index # used when type -> text
        self.uid_keyname = uid_keyname # used when type -> json
        self.utt_keyname = utt_keyname # used when type -> json
        self.transcript_path = transcript_path
        self.wavs_path = wavs_path
        self.validation_split = validation_split

        check_argument("uid_index", self.uid_index, min_val=0)
        check_argument("utt_index", self.utt_index, min_val=0)
        # assert os.path.isfile(transcript_path), f"transcript_path ({self.transcript_path}) file does not exist"
        if type(self.validation_split) == int: # count of validation samples
            check_argument("validation_split", self.validation_split, min_val=0)
        elif type(self.validation_split) ==  float: # fraction of validation samples
            check_argument("validation_split", self.validation_split, min_val=0, max_val=1)


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
        wandb_auth_key (str): The authentication key for wandb. 
                Default is an ``""``.
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
            If `wandb_logger` is True and `wandb_auth_key` is not provided.
            If `resume_from_checkpoint` is True and `checkpoint_path` is not provided.
            If `epoch_start` is less than 1.
    """
    @typechecked
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
        wandb_auth_key: str = "",
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
        self.wandb_auth_key = wandb_auth_key
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
        if (self.wandb_logger):
            assert self.wandb_auth_key, "wandb_auth_key not provided (wandb_logger is set as True). You can find your API key in your browser here: https://wandb.ai/authorize."
        if (self.resume_from_checkpoint):
            assert self.checkpoint_path, "checkpoint_path not provided (warm_start has been enabled) to start training"
        check_argument("epoch_start", self.epoch_start, min_val=1)


class ModelConfig(BaseConfig):
    """
    Config for Models -> TTS / VOC (Vocoder)
    """

    MODEL_DETAILS = {
        "TTS": ["Tacotron2"],
        "VOC": ["MelGAN"],
    }

    def __init__(self):
        config_name = self.__class__.__name__
        self.model_name = config_name[: config_name.find("Config")]
        if self.model_name in self.MODEL_DETAILS["TTS"]:
            self.task = "TTS"
        elif self.model_name in self.MODEL_DETAILS["VOC"]:
            self.task = "VOC"
        else:
            print(f"config {config_name} not implemented!")
            exit()

    @typechecked
    def load_symbols(self, dump_dir: str = "dump"):
        assert self.task == "TTS", f"invalid function for given task ({self.task})"
        self.symbols = load_json(os.path.join(dump_dir, "token_list.json"))
        self.n_symbols = len(self.symbols)


class Tacotron2Config(ModelConfig):
    """
    Config for Tacotron2 TTS architecture
    """
    @typechecked
    def __init__(
        self,
        symbols: Union[Dict, None] = None,
        n_symbols: Union[int, None] = None,
        symbols_embedding_dim: int = 512,
        encoder_kernel_size: int = 5,
        encoder_n_convolutions: int = 3,
        encoder_embedding_dim: int = 512,
        decoder_rnn_dim: int = 1024,
        prenet_dim: int = 256,
        max_decoder_steps: int = 1000,
        gate_threshold: float = 0.5,
        p_attention_dropout: float = 0.1,
        p_decoder_dropout: float = 0.1,
        attention_rnn_dim: int = 1024,
        attention_dim: int = 128,
        attention_location_n_filters: int = 32,
        attention_location_kernel_size: int = 31,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 5,
        mask_padding: bool = True
    ):
        # model details
        super().__init__()
        # input params
        self.symbols = symbols
        self.n_symbols = n_symbols
        self.symbols_embedding_dim = symbols_embedding_dim
        # encoder params
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_n_convolutions = encoder_n_convolutions
        self.encoder_embedding_dim = encoder_embedding_dim
        # decoder params
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        # attention
        self.attention_rnn_dim = attention_rnn_dim
        self.attention_dim = attention_dim
        self.attention_location_n_filters = attention_location_n_filters
        self.attention_location_kernel_size = attention_location_kernel_size
        # postnet
        self.postnet_embedding_dim = postnet_embedding_dim
        self.postnet_kernel_size = postnet_kernel_size
        self.postnet_n_convolutions = postnet_n_convolutions
        #
        self.mask_padding = mask_padding

        if (self.symbols != None):
            assert len(self.symbols) == self.n_symbols, f"symbols count ({len(self.symbols)}) does not match with n_symbols ({self.n_symbols})"
        check_argument("symbols_embedding_dim", self.symbols_embedding_dim, min_val=1)
        check_argument("encoder_kernel_size", self.encoder_kernel_size, min_val=1)
        check_argument("encoder_n_convolutions", self.encoder_n_convolutions, min_val=1)
        check_argument("encoder_embedding_dim", self.encoder_embedding_dim, min_val=1)
        check_argument("decoder_rnn_dim", self.decoder_rnn_dim, min_val=1)
        check_argument("prenet_dim", self.prenet_dim, min_val=1)
        check_argument("max_decoder_steps", self.max_decoder_steps, min_val=1, max_val=10000)
        check_argument("gate_threshold", self.gate_threshold, min_val=0, max_val=1)
        check_argument("p_attention_dropout", self.p_attention_dropout, min_val=0)
        check_argument("p_decoder_dropout", self.p_decoder_dropout, min_val=0)
        check_argument("attention_rnn_dim", self.attention_rnn_dim, min_val=1)
        check_argument("attention_dim", self.attention_dim, min_val=1)
        check_argument("attention_location_n_filters", self.attention_location_n_filters, min_val=1)
        check_argument("attention_location_kernel_size", self.attention_location_kernel_size, min_val=1)
        check_argument("postnet_embedding_dim", self.postnet_embedding_dim, min_val=1)
        check_argument("postnet_kernel_size", self.postnet_kernel_size, min_val=1)
        check_argument("postnet_n_convolutions", self.postnet_n_convolutions, min_val=1)


class MelGANConfig(ModelConfig):
    """
    Config for MelGAN Vocoder architecture
    """
    @typechecked
    def __init__(
        self,
        train_repeat_discriminator: int = 1,
        max_frames: int = 200,
        feat_match: float = 10.0
    ):
        # model details
        super().__init__()
        self.train_repeat_discriminator = train_repeat_discriminator
        self.max_frames = max_frames
        self.feat_match = feat_match

        check_argument("train_repeat_discriminator", self.train_repeat_discriminator, min_val=1)
        check_argument("max_frames", self.max_frames, min_val=100)
        check_argument("feat_match", self.feat_match, min_val=1)


class OptimizerConfig(BaseConfig):
    """
    Config for Optimizer (Adam)

    Args:
        learning_rate (float): Learning rate for the optimizer.
                Default is ``1e-3``.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.
                Default is ``1e-6``.
        grad_clip_thresh (float): The threshold value for gradient clipping.
                Default is ``1.0``.
        beta1 (float): The exponential decay rate for the first moment estimates in Adam optimizer.
                Default is ``0.9``.
        beta2 (float): The exponential decay rate for the second moment estimates in Adam optimizer.
                Default is ``0.999``.

    Raises:
        AssertionError: If any of the argument values are invalid.
    """
    @typechecked
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        grad_clip_thresh: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_thresh = grad_clip_thresh
        self.beta1 = beta1
        self.beta2 = beta2

        check_argument("learning_rate", self.learning_rate, min_val=1e-5)
        check_argument("weight_decay", self.weight_decay, min_val=0)
        check_argument("grad_clip_thresh", self.grad_clip_thresh, min_val=0)
        check_argument("beta1", self.beta1, min_val=0, max_val=1)
        check_argument("beta2", self.beta2, min_val=0, max_val=1)


def load_configs(path: str) -> Dict:
    config_type = os.path.splitext(path)[1][1: ]
    assert config_type in ["json", "yaml"], f"given config extension ({config_type}) is invalid"
    if (config_type == "json"):
        config_json = load_json(path)
    elif (config_type == "yaml"):
        config_json = load_yaml(path)
    configs = {}
    if 'text_config' in config_json:
        configs['text_config'] = TextConfig(**config_json['text_config'])
    if 'audio_config' in config_json:
        configs['audio_config'] = AudioConfig(**config_json['audio_config'])
    if 'dataset_config' in config_json:
        configs['dataset_config'] = DatasetConfig(
            text_config=(configs['text_config'] if 'text_config' in configs else TextConfig()),
            audio_config=(configs['audio_config'] if 'audio_config' in configs else AudioConfig()),
            **config_json['dataset_config'])
    if 'trainer_config' in config_json:
        configs['trainer_config'] = TrainerConfig(**config_json['trainer_config'])
    if 'model_config' in config_json:
        task, model_name = config_json['model_config']['task'], config_json['model_config']['model_name']
        del config_json['model_config']['task']
        del config_json['model_config']['model_name']
        if task == 'TTS':
            if model_name == 'Tacotron2':
                configs['model_config'] = Tacotron2Config(**config_json['model_config'])
    if 'optimizer_config' in config_json:
        configs['optimizer_config'] = OptimizerConfig(**config_json['optimizer_config'])
    return configs

def get_json_from_config(config):
    config_json = {}
    for attr, value in vars(config).items():
        if not isinstance(value, BaseConfig):
            config_json[attr] = value
    return config_json

@typechecked
def write_configs(
        path: str,
        text_config: Optional[TextConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        dataset_config: Optional[DatasetConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        model_config: Optional[ModelConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None
    ) -> None:
    config_type = os.path.splitext(path)[1][1: ]
    assert config_type in ["json", "yaml"], f"given config extension ({config_type}) is invalid"
    config_json = {}
    if text_config != None:
        config_json['text_config'] = get_json_from_config(text_config)
    if audio_config != None:
        config_json['audio_config'] = get_json_from_config(audio_config)
    if dataset_config != None:
        config_json['dataset_config'] = get_json_from_config(dataset_config)
    if trainer_config != None:
        config_json['trainer_config'] = get_json_from_config(trainer_config)
    if model_config != None:
        config_json['model_config'] = get_json_from_config(model_config)
    if optimizer_config != None:
        config_json['optimizer_config'] = get_json_from_config(optimizer_config)
    if (config_type == "json"):
        dump_json(path, config_json)
    elif (config_type == "yaml"):
        dump_yaml(path, config_json)
