from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent)) # file: GenVox/tools/  -> parent.parent: GenVox/
import os
import scipy.io.wavfile
import argparse
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
    with open(path, 'r') as f:
        lines = f.readlines()
    for ind, line in enumerate(lines):
        line = line.strip()
        if "-->" in line: # timestamp line
            st_time, en_time = line.split("-->")
            st_time, en_time = timestamp_to_seconds(st_time.strip()), timestamp_to_seconds(en_time.strip())
            text = lines[ind + 1].strip()
            if not (text[0] == "[" and text[-1] == "]"):
                data.append((st_time, en_time, text))
    return data

def chunk_audio(
        wav_path: str,
        transcript_path: str,
        wavs_path: str,
        transcript_data: List[Tuple[float, float, str]],
        speaker_id: str
    ):
    fs, wav = scipy.io.wavfile.read(wav_path)
    os.makedirs(wavs_path)
    count_pad = len(str(len(transcript_data)))
    transcript = []
    for ind, (st_time, en_time, text) in enumerate(transcript_data):
        st_sample, en_sample = int(st_time) * fs, int(en_time) * fs
        segment = wav[st_sample: en_sample]
        st_sample, en_sample = get_non_silent_boundary(
            signal=wav, 
            fs=fs, 
            silence_threshold=-50.0
        )
        segment = segment[st_sample: en_sample]
        ind_val = str(ind + 1).rjust(count_pad, "0")
        utt_id = f"{speaker_id}_{ind_val}"
        segment_path = os.path.join(wavs_path, utt_id + ".wav")
        transcript.append(f"{utt_id}|{text}")
        scipy.io.wavfile.write(segment_path, fs, segment)
    with open(transcript_path, 'w') as f:
        f.write("\n".join(transcript))

def createDatasetFromYoutube(link: str, output_path: str, cache_path: str, remove_cache: bool, speaker_id: str, verbose: bool) -> None:
    assert is_youtube_link(link=link), f"given link ({link}) is not a youtube link"
    assert not os.path.isdir(output_path), f"output_path ({output_path}) already exists"
    assert not os.path.isdir(cache_path), f"cache_path ({cache_path}) already exists"
    print(f"downloading youtube data to {cache_path} directory")
    output_name = os.path.join(cache_path, "output")
    download_wav_from_youtube(
        link=link,
        target=output_name,
        verbose=verbose
    )
    wav_path = output_name + ".wav"
    vtt_path = output_name + ".en.vtt"
    assert os.path.isfile(vtt_path), f"transcript file not found at ({vtt_path})"
    transcript_data = process_vtt(path=vtt_path)
    chunk_audio(
        wav_path=wav_path,
        transcript_path=os.path.join(output_path, "metadata.csv"),
        wavs_path=os.path.join(output_path, "wavs"),
        transcript_data=transcript_data,
        speaker_id=speaker_id
    )
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
    parser.add_argument("--cache_dir", type=str, help="path of directory to download the data into", default=get_random_HEX_name())
    parser.add_argument("--remove_cache", help="flag for removing cache dir", action="store_true")
    parser.add_argument("--verbose", help="print will be more verbose", action="store_true")
    args = parser.parse_args()
    createDatasetFromYoutube(
        link=args.link,
        output_path=args.output_dir,
        cache_path=args.cache_dir,
        remove_cache=args.remove_cache,
        speaker_id=args.speaker_id,
        verbose=args.verbose
    )
