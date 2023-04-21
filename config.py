import os
from typing import List, Union, Dict
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
    def __str__(self, level=0):
        rstr = ("    " * level) + self.__class__.__name__ + "\n"
        for key, value in iter(self.__dict__.items()):
            if (isinstance(value, BaseConfig)):
                svalue = "\n" + BaseConfig.__str__(value, level=level + 1)
            else:
                svalue = "(" + str(value) + ")"
            rstr += ("    " * level) + "└── " + key.ljust(35) + svalue + "\n"
        return rstr.rstrip()
    
    def __repr__(self):
        return str(self)


class DownloadConfig(BaseConfig):
    """
    Config for DownloadProcessor
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


class AudioConfig(BaseConfig):
    """
    Config for AudioProcessor
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
        log_func: str = "np.log10",
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
        assert self.log_func in ["np.log", "np.log10"], f"The value \'log_func\' ({self.log_func}) is an invalid function name."
        check_argument("ref_level_db", self.ref_level_db, min_val=1)


class DatasetConfig(BaseConfig):
    """
    Config for DatasetProcessor
    """
    @typechecked
    def __init__(
        self,
        text_config: TextConfig,
        audio_config: AudioConfig,
        dataset_type: str = "text",
        delimiter: str = " ",
        uid_index: int = 0,
        utt_index: int = 1,
        transcript_path: str = "",
        wavs_path: str = "",
        validation_split: Union[int, float] = 0,
        remove_wav_dump: bool = True
    ):
        self.text_config = text_config
        self.audio_config = audio_config
        self.dataset_type = dataset_type
        self.delimiter = delimiter
        self.uid_index = uid_index
        self.utt_index = utt_index
        self.transcript_path = transcript_path
        self.wavs_path = wavs_path
        self.validation_split = validation_split
        self.remove_wav_dump = remove_wav_dump

        assert self.dataset_type in ["text", "json"], f"dataset_type ({self.dataset_type}) is invalid"
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
    """
    @typechecked
    def __init__(
        self,
        project_name: str = "",
        experiment_id: str = "",
        batch_size: int = 64,
        num_loader_workers: int = 2,
        run_validation: bool = True,
        validation_batch_size: int = 32,
        epochs: int = 100,
        seed: int = 0,
        use_cuda: bool = False,
        max_best_models: int = 5,
        iters_for_checkpoint: int = 1,
        wandb_logger: bool = True,
        wandb_auth_key: str = ""
    ):
        self.project_name = project_name
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers # for DataLoader(num_workers=___) in torch.utils.data
        self.run_validation = run_validation
        self.validation_batch_size = validation_batch_size
        self.epochs = epochs
        self.seed = seed
        self.use_cuda = use_cuda
        self.max_best_models = max_best_models
        self.iters_for_checkpoint = iters_for_checkpoint
        self.wandb_logger = wandb_logger
        self.wandb_auth_key = wandb_auth_key

        assert self.project_name, "project_name not provided"
        if self.experiment_id == "":
            self.experiment_id = get_random_HEX_name(40)
            print(f"experiment_id not provided, hence randomly generated ({self.experiment_id})")
        check_argument("epochs", self.epochs, min_val=1)
        check_argument("max_best_models", self.max_best_models, min_val=1, max_val=10)
        check_argument("iters_for_checkpoint", self.iters_for_checkpoint, min_val=1)
        if (self.wandb_logger):
            assert self.wandb_auth_key, "wandb_auth_key not provided (wandb_logger is set as True). You can find your API key in your browser here: https://wandb.ai/authorize."


class Tacotron2Config(BaseConfig):
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
        self.model_architecture = "Tacotron2"
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
        else:
            self.symbols = load_json(os.path.join("dump", "token_list.json"))
            self.n_symbols = len(self.symbols)
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


class OptimizerConfig(BaseConfig):
    """
    Config for Tacotron2 Adam Optimizer
    """
    @typechecked
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        check_argument("learning_rate", self.learning_rate, min_val=1e-5)
        check_argument("weight_decay", self.weight_decay, min_val=0)


def load_config_from_file(path):
    config_type = os.path.splitext(path)[1][1: ]
    assert config_type in ["json", "yaml"], f"given config extension ({config_type}) is invalid"
    if (config_type == "json"):
        config_json = load_json(path)
    elif (config_type == "yaml"):
        config_json = load_yaml(path)
    text_config = TextConfig(**config_json['text_config'])
    audio_config = AudioConfig(**config_json['audio_config'])
    dataset_config = DatasetConfig(text_config=text_config, audio_config=audio_config, **config_json['dataset_config'])
    trainer_config = TrainerConfig(**config_json['trainer_config'])
    del config_json['tacotron2_config']['model_architecture']
    tacotron2_config = Tacotron2Config(**config_json['tacotron2_config'])
    optimizer_config = OptimizerConfig(**config_json['optimizer_config'])
    return text_config, audio_config, dataset_config, trainer_config, tacotron2_config, optimizer_config

def get_json_from_config(config):
    config_json = {}
    for attr, value in vars(config).items():
        if not isinstance(value, BaseConfig):
            config_json[attr] = value
    return config_json

@typechecked
def write_file_from_config(path, 
                           text_config: TextConfig, 
                           audio_config: AudioConfig, 
                           dataset_config: DatasetConfig, 
                           trainer_config: TrainerConfig, 
                           tacotron2_config: Tacotron2Config, 
                           optimizer_config: OptimizerConfig,
                           ):
    config_type = os.path.splitext(path)[1][1: ]
    assert config_type in ["json", "yaml"], f"given config extension ({config_type}) is invalid"
    config_json = {
        'text_config': get_json_from_config(text_config),
        'audio_config': get_json_from_config(audio_config),
        'dataset_config': get_json_from_config(dataset_config),
        'trainer_config': get_json_from_config(trainer_config),
        'tacotron2_config': get_json_from_config(tacotron2_config),
        'optimizer_config': get_json_from_config(optimizer_config)
    }
    if (config_type == "json"):
        dump_json(path, config_json)
    elif (config_type == "yaml"):
        dump_yaml(path, config_json)
