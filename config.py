import os
from typeguard import typechecked

from utils import dump_json, load_json

def check_argument(name, value, min_val=None, max_val=None):
    if (min_val == None):
        assert (value <= max_val), f"The value \'{name}\' ({value}) is above max_val ({max_val})."
    elif (max_val == None):
        assert (value >= min_val), f"The value \'{name}\' ({value}) is below min_val ({min_val})"
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
            rstr += ("    " * level) + "└── " + key.ljust(20) + svalue + "\n"
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
    def __init__(self):
        pass


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
        max_wav_duration: float = 10
    ):
        self.sampling_rate = sampling_rate
        self.trim_silence = trim_silence
        self.trim_dbfs = trim_dbfs # decibels relative to full scale
        self.min_wav_duration = min_wav_duration
        self.max_wav_duration = max_wav_duration

        check_argument("sampling_rate", self.sampling_rate, min_val=16000, max_val=44100)
        check_argument("trim_dbfs", self.trim_dbfs, min_val=-100, max_val=0)
        check_argument("min_wav_duration", self.min_wav_duration, min_val=0.1)
        check_argument("max_wav_duration", self.max_wav_duration, min_val=self.min_wav_duration)


class DatasetConfig(BaseConfig):
    """
    Config for DatasetProcessor
    """
    @typechecked
    def __init__(
        self,
        text_config: TextConfig,
        audio_config: AudioConfig,
        dataset_type = "text",
        delimiter = " ",
        uid_index: int = 0,
        utt_index: int = 1,
        transcript_path: str = "",
        wavs_path: str = "",
    ):
        self.text_config = text_config
        self.audio_config = audio_config
        self.dataset_type = dataset_type
        self.delimiter = delimiter
        self.uid_index = uid_index
        self.utt_index = utt_index
        self.transcript_path = transcript_path
        self.wavs_path = wavs_path

        assert self.dataset_type in ["text", "json"], f"dataset_type ({self.dataset_type}) is invalid"
        check_argument("uid_index", self.uid_index, min_val=0)
        check_argument("utt_index", self.utt_index, min_val=0)
        assert os.path.isfile(transcript_path), f"transcript_path ({self.transcript_path}) file does not exist"


def load_config_from_file(path):
    config_json = load_json(path)
    text_config = TextConfig(**config_json['text_config'])
    del config_json['text_config']
    audio_config = AudioConfig(**config_json['audio_config'])
    del config_json['audio_config']
    dataset_config = DatasetConfig(text_config=text_config, audio_config=audio_config, **config_json)
    return text_config, audio_config, dataset_config

def get_json_from_config(config):
    config_json = {}
    for attr, value in vars(config).items():
        if isinstance(value, BaseConfig):
            value = get_json_from_config(value)
        config_json[attr] = value
    return config_json

@typechecked
def write_file_from_config(config: DatasetConfig, path):
    config_json = get_json_from_config(config)
    dump_json(path, config_json)
