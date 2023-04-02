import random
import yt_dlp
import scipy.io
import json
import numpy as np

def download_YT_mp3(link, target, verbose=False):
    # download options for youtube_dl
    ydl_opts = {
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

def is_youtube_link(link):
    if "youtube.com/watch?v=" in link:
        return True
    elif "youtu.be/" in link:
        return True
    return False

def get_random_HEX_name(size = 60):
    return hex(random.getrandbits(size))[2: ]

def dBFS(signal):
    max_amplitude = np.iinfo(signal.dtype).max
    norm_signal = signal / max_amplitude
    rms = np.sqrt(np.mean(np.square(norm_signal)))
    if (rms == 0):
        return -1000
    dbfs = 20 * np.log10(rms)
    return dbfs

def get_silent_signal_ind(signal, fs, silence_threshold):
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

def trim_audio_silence(input_path, output_path, silence_threshold=-50.0):
    fs, wav = scipy.io.wavfile.read(input_path)
    left_ind, right_ind = get_silent_signal_ind(wav, fs, silence_threshold)
    assert left_ind < right_ind, "empty audio signal given for trimming silence"
    trimmed_wav = wav[left_ind: right_ind]
    scipy.io.wavfile.write(output_path, fs, trimmed_wav)

def sec_to_formatted_time(seconds):
    seconds = int(seconds)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    days = int(hours / 24)
    seconds -= minutes * 60
    minutes -= hours * 60
    hours -= days * 24
    if (hours > 0):
        return f"{days}-{hours}:{minutes}:{seconds}"
    return f"{hours}:{minutes}:{seconds}"

def dump_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data