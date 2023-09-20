from typing import Optional, Union, List, Literal
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
        return str(self)


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
