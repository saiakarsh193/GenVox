import os
from typeguard import typechecked

def check_argument(name, value, min_val=None, max_val=None):
    if (min_val == None):
        assert (value <= max_val), f"The value \'{name}\' ({value}) is above max_val ({max_val})."
    elif (max_val == None):
        assert (value >= min_val), f"The value \'{name}\' ({value}) is below min_val ({min_val})"
    else:
        assert (value >= min_val and value <= max_val), f"The value \'{name}\' ({value}) is not in the required range ({min_val} -> {max_val})."


class BaseConfig:
    def __repr__(self):
        rstr = self.__class__.__name__ + "\n"
        for key, value in iter(self.__dict__.items()):
            rstr += "└── " + key.ljust(20) + ":\t" + str(value) + "\n"
        return rstr.strip()


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


class DatasetConfig(BaseConfig):
    """
    Config for DatasetProcessor
    """
    @typechecked
    def __init__(
        self,
        dataset_type = "text",
        delimiter = " ",
        uid_index: int = 0,
        utt_index: int = 1,
        transcript_path: str = "",
        wavs_path: str = ""
    ):
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
        # decibels relative to full scale
        self.trim_dbfs = trim_dbfs
        self.min_wav_duration = min_wav_duration
        self.max_wav_duration = max_wav_duration

        check_argument("sampling_rate", self.sampling_rate, min_val=512, max_val=22050)
        check_argument("trim_dbfs", self.trim_dbfs, min_val=-100, max_val=0)
        check_argument("min_wav_duration", self.min_wav_duration, min_val=0)
        check_argument("max_wav_duration", self.max_wav_duration, min_val=self.min_wav_duration)
