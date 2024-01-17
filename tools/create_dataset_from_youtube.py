from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent)) # file: GenVox/tools/  -> parent.parent: GenVox/
import re
import os
import scipy.io.wavfile
import argparse
import webvtt
from hashlib import sha256
import numpy as np
from typing import Tuple, List

from utils import get_random_HEX_name, download_wav_from_youtube, is_youtube_link, get_non_silent_boundary

def timestamp_to_seconds(timestamp: str) -> float:
    hour, minute, seconds = timestamp.split(":")
    total_seconds = float(seconds)
    total_seconds += int(minute) * 60
    total_seconds += int(hour) * 60 * 60
    return total_seconds

def process_vtt(path: str) -> List[Tuple[float, float, str]]:
    data = []
    for caption in webvtt.read(path):
        start = timestamp_to_seconds(caption.start)
        end = timestamp_to_seconds(caption.end)
        text = str(caption.text).strip().split("\n")[-1] # last sentence of segment only
        if text == "":
            continue
        if len(data) > 0 and data[-1][2] == text:
            data[-1] = (data[-1][0], end, text)
        else:
            data.append((start, end, text))
    return data

def clean_transcript(text: str) -> str:
    text = re.sub(r"^ *- +", '', text)
    text = re.sub(r"\.{2,}", '', text)
    text = re.sub(r"\[.*\]", '', text)
    text = re.sub(r"\(.*\)", '', text)
    return text

def process_transcript(transcript: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    MAX_AUDIO_LENGTH = 12
    new_transcript = []
    for st, en, text in transcript:
        text = clean_transcript(text)
        if text == "":
            continue
        if len(new_transcript) > 0 and (en - st) + (new_transcript[-1][1] - new_transcript[-1][0]) < MAX_AUDIO_LENGTH:
            new_transcript[-1] = (new_transcript[-1][0], en, new_transcript[-1][2] + " " + text)
        else:
            new_transcript.append((st, en, text))
    return new_transcript

def trim_silence_and_update_transcript(wav: np.ndarray, fs: int, transcript: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    MIN_AUDIO_LENGTH = 3
    new_transcript = []
    for st, en, text in transcript:
        segment = wav[int(st) * fs: int(en) * fs]
        left_ind, right_ind = get_non_silent_boundary(segment, fs, silence_threshold=-50)
        if (right_ind - left_ind) / fs > MIN_AUDIO_LENGTH:
            new_transcript.append((st + (left_ind / fs), st + (right_ind / fs), text))
    return new_transcript

def chunk_audio(wav: np.ndarray, fs: int, transcript: List[Tuple[float, float, str]], output_path: str, speaker_id: str, start_index: int, index_pad: str) -> List[str]:
    data = []
    ind = start_index
    for st, en, text in transcript:
        segment = wav[int(st) * fs: int(en) * fs]
        utt_id = f"{speaker_id}_{str(ind).rjust(index_pad, '0')}"
        segment_path = os.path.join(output_path, utt_id + ".wav")
        scipy.io.wavfile.write(segment_path, fs, segment)
        data.append(f"{utt_id}|{text}")
        ind += 1
    return data

def createDatasetFromYoutube(links: List[str], output_path: str, speaker_id: str, remove_cache: bool = True, verbose: bool = True) -> None:
    assert not os.path.isdir(output_path), f"output_path ({output_path}) already exists"
    os.mkdir(output_path)
    cache_path = os.path.join(output_path, get_random_HEX_name())
    os.mkdir(cache_path)
    link_path = {}
    for link in links:
        assert is_youtube_link(link=link), f"given link ({link}) is not a youtube link"
        link_path[link] = sha256(link.encode('utf-8')).hexdigest()[:15]
    print(f"downloading youtube data to {cache_path} directory")
    link_data = {}
    total_samples = 0
    for link in links:
        download_wav_from_youtube(
            link=link,
            target=os.path.join(cache_path, link_path[link]),
            verbose=verbose
        )
        wav_path = os.path.join(cache_path, link_path[link] + ".wav")
        vtt_path = os.path.join(cache_path, link_path[link] + ".en.vtt")
        assert os.path.isfile(vtt_path), f"transcript file not found at ({vtt_path})"
        transcript = process_vtt(path=vtt_path)
        transcript = process_transcript(transcript)
        fs, wav = scipy.io.wavfile.read(wav_path)
        transcript = trim_silence_and_update_transcript(wav, fs, transcript)
        link_data[link] = (wav, fs, transcript)
        total_samples += len(transcript)
        print(len(transcript))
    wav_output_path = os.path.join(output_path, "wavs")
    os.mkdir(wav_output_path)
    index_pad = len(str(total_samples))
    transcript_data = []
    print(f"chunking audio to {wav_output_path} directory")
    for link in links:
        transcript_data += chunk_audio(
            wav=link_data[link][0],
            fs=link_data[link][1],
            transcript=link_data[link][2],
            output_path=wav_output_path,
            speaker_id=speaker_id,
            start_index=len(transcript_data) + 1,
            index_pad=index_pad,
        )
    transcript_path = os.path.join(output_path, "metadata.csv")
    with open(transcript_path, 'w') as f:
        f.write("\n".join(transcript_data))
    if remove_cache:
        print(f"removing {cache_path} directory")
        os.system(f"rm -r {cache_path}")
    print(f"youtube dataset successfully created in {output_path}")
    print("NOTE: the quality of the dataset depends on the transcript downloaded from youtube, so check the dataset before using")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download and create annotated dataset from youtube videos")
    parser.add_argument("link", type=str, help="link to youtube video")
    parser.add_argument("output_dir", type=str, help="path of directory to store the dataset")
    parser.add_argument("--speaker_id", "-sid", type=str, help="speaker id prefix for storing dataset", default="SPK")
    parser.add_argument("--remove_cache", help="flag for removing cache dir", action="store_true")
    parser.add_argument("--verbose", help="print will be more verbose", action="store_true")
    args = parser.parse_args()
    createDatasetFromYoutube(
        links=[args.link],
        output_path=args.output_dir,
        speaker_id=args.speaker_id,
        remove_cache=args.remove_cache,
        verbose=args.verbose
    )
