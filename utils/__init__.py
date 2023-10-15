import random
import yt_dlp
import scipy.io
import json
import yaml
import datetime
import numpy as np
from typing import Union, Tuple, Dict

def download_wav_from_youtube(link: str, target: str, verbose: bool = False) -> None:
    """
    download audio files from Youtube in wav format
    """
    ydl_opts = { # download options for youtube_dl
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': 'verbose' if verbose else 'quiet',
        'outtmpl': target,
        'writesubtitles' : target,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

def is_youtube_link(link: str) -> bool:
    """
    check if a given link is from youtube or not
    """
    if "youtube.com/watch?v=" in link:
        return True
    elif "youtu.be/" in link:
        return True
    return False

def get_random_HEX_name(size: int = 60) -> str:
    """
    get a random HEX string
    """
    return hex(random.getrandbits(size))[2: ]

def dBFS(signal: np.ndarray) -> Union[float, int]:
    """
    measure the dBFS (decibels relative to full scale) value for a given signal
    """
    max_amplitude = np.iinfo(signal.dtype).max
    norm_signal = signal / max_amplitude
    rms = np.sqrt(np.mean(np.square(norm_signal)))
    if (rms == 0):
        return -1000
    dbfs = 20 * np.log10(rms)
    return dbfs

def get_non_silent_boundary(signal: np.ndarray, fs: int, silence_threshold: Union[float, int]) -> Tuple[int, int]:
    """
    returns the boundary indices of a signal which are not silent based on the given threshold
    """
    chunk_size = 20 # ms
    chunk_length = int(chunk_size * 0.001 * fs) # frames

    for left_ind in range(0, signal.shape[0], chunk_length):
        ind_end = min(left_ind + chunk_length, signal.shape[0])
        sig = signal[left_ind: ind_end]
        if dBFS(sig) >= silence_threshold:
            break

    reverse_signal = np.flip(signal)
    for right_ind in range(0, reverse_signal.shape[0], chunk_length):
        ind_end = min(right_ind + chunk_length, reverse_signal.shape[0])
        sig = reverse_signal[right_ind: ind_end]
        if dBFS(sig) >= silence_threshold:
            break
    right_ind = signal.shape[0] - right_ind
    return left_ind, right_ind

def trim_audio_silence(input_path: str, output_path: str, silence_threshold: float = -50.0) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """
    returns both the boundary indices of a signal which are not silent based on the given threshold, and also the old and new length of the trimmed audio.
    it also saved the trimmed audio to the given path
    """
    fs, wav = scipy.io.wavfile.read(input_path)
    left_ind, right_ind = get_non_silent_boundary(wav, fs, silence_threshold)
    assert left_ind < right_ind, "empty audio signal given for trimming silence"
    trimmed_wav = wav[left_ind: right_ind]
    scipy.io.wavfile.write(output_path, fs, trimmed_wav)
    return (left_ind, right_ind), (wav.shape[0] / fs, trimmed_wav.shape[0] / fs)

def sec_to_formatted_time(seconds: int) -> str:
    """
    returns the standard time formatted string for a given amount of time (in seconds)
    """
    seconds = int(seconds)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    days = int(hours / 24)
    seconds -= minutes * 60
    minutes -= hours * 60
    hours -= days * 24
    if (days > 0):
        return f"{days}-{hours}:{minutes}:{seconds}"
    return f"{hours}:{minutes}:{seconds}"

def current_formatted_time(sec_add: int = 0) -> str:
    """
    returns the standard time formatted string for the current time
    """
    dt_now = datetime.datetime.now()
    dt_now = dt_now + datetime.timedelta(0, sec_add)
    return dt_now.strftime("%Y-%m-%d %H:%M:%S")

def center_print(sentence: str, space_factor: float = 0.5, fill_symbol: str = "=", fill_length: int = 100) -> None:
    """
    prints given string in the center
    """
    pad_left = (fill_length - len(sentence))
    space_pad = len(sentence) + int(pad_left * space_factor)
    sentence = sentence.center(space_pad, ' ').center(fill_length, fill_symbol)
    print(sentence)

def log_print(*args) -> None:
    print(current_formatted_time(), *args)

def dump_json(path: str, data: Dict) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def dump_yaml(path: str, data: Dict) -> None:
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

def load_yaml(path: str) -> Dict:
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data

def human_readable_int(val: int, precision: int = 2) -> str:
    """
    converts an integer into a human readable format
    """
    K, M, B = 1e3, 1e6, 1e9
    if abs(val) < K:
        return str(val)
    if abs(val) < M:
        if val % K == 0:
            return f"{val // K}K"
        return f"{val / K:.{precision}f}K"
    if abs(val) < B:
        if val % M == 0:
            return f"{val // M}M"
        return f"{val / M:.{precision}f}M"
    if val % B == 0:
        return f"{val // B}B"
    return f"{val / B:.{precision}f}B"

def count_parameters(model) -> Dict[str, int]:
    """
    count the number of parameters in a given torch model
    """
    total_c = 0
    trainable_c = 0
    for p in model.parameters():
        total_c += p.numel()
        if p.requires_grad:
            trainable_c += p.numel()
    return {"total_parameters": total_c, "trainable_parameters": trainable_c, "nontrainable_parameters": total_c - trainable_c}

def print_parameter_count(model) -> None:
    """
    print the number of parameters in a given torch model
    """
    counts = count_parameters(model)
    just_count = 10
    center_print(f" {model.__class__.__module__}.{model.__class__.__name__} (Parameters Count Summary) ", space_factor=0, fill_symbol="-")
    print("Trainable Parameters     : " + str(counts["trainable_parameters"]).rjust(just_count) + " (" + human_readable_int(counts["trainable_parameters"]) + ")")
    print("Non-Trainable Parameters : " + str(counts["nontrainable_parameters"]).rjust(just_count) + " (" + human_readable_int(counts["nontrainable_parameters"]) + ")")
    print("Total Parameters         : " + str(counts["total_parameters"]).rjust(just_count) + " (" + human_readable_int(counts["total_parameters"]) + ")")
    center_print(f"", space_factor=0, fill_symbol="-")
