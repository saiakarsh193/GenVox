import os
from typeguard import typechecked


def check_argument(name, value, min_val=None, max_val=None):
    if (min_val == None):
        assert (value <= max_val), f"The value \'{name}\' ({value}) is above max_val [{max_val}]."
    elif (max_val == None):
        assert (value >= min_val), f"The value \'{name}\' ({value}) is below min_val [{min_val}]"
    else:
        assert (value >= min_val and value <= max_val), f"The value \'{name}\' ({value}) is not in the required range [{min_val} -> {max_val}]."


class BaseConfig:
    def __repr__(self):
        rstr = self.__class__.__name__ + "\n"
        for key, value in iter(self.__dict__.items()):
            rstr += "└── " + key.ljust(20) + ":\t" + str(value) + "\n"
        return rstr.strip()


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
        transcript_path = None,
        wavs_path = None
    ):
        self.dataset_type = dataset_type
        assert self.dataset_type in ["text", "json"], f"dataset_type ({self.dataset_type}) is invalid"
        self.delimiter = delimiter
        self.uid_index = uid_index
        check_argument("uid_index", self.uid_index, min_val=0)
        self.utt_index = utt_index
        check_argument("utt_index", self.utt_index, min_val=0)
        self.transcript_path = transcript_path
        assert os.path.isfile(transcript_path), f"transcript_path ({self.transcript_path}) file does not exist"
        self.wavs_path = wavs_path


class AudioConfig(BaseConfig):
    """
    Config for AudioProcessor
    """
    @typechecked
    def __init__(
        self,
        sampling_rate: int = 22050,
        trim_silence: bool = True,
        trim_db: float = 50.0
    ):
        self.sampling_rate = sampling_rate
        check_argument("sampling_rate", self.sampling_rate, min_val=512, max_val=22050)
        self.trim_silence = trim_silence
        self.trim_db = trim_db
