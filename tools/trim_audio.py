import argparse
from pathlib import Path
from multiprocessing import Pool
import os
import sys
# file: GenVox/tools/resample.py  -> parent.parent: GenVox/
sys.path.append(str(Path(__file__).parent.parent))
import tqdm

from utils import function_timer, trim_audio_silence, sec_to_formatted_time

@function_timer
def trim_audio(args):
    assert not os.path.isdir(args.output_dir), f"output directory {args.output_dir} already exists"
    os.mkdir(args.output_dir)
    tasks = []
    for wav_path in sorted(Path(args.input_dir).rglob("*.wav")):
        # wav_path, wav_path.parent, wav_path.name, wav_path.stem, wav_path.suffix
        outpath = os.path.join(args.output_dir, wav_path.name)
        tasks.append((wav_path, outpath))
    total_length = 0
    final_length = 0
    for inpath, outpath in tqdm.tqdm(tasks):
        _, (old_len, new_len) = trim_audio_silence(inpath, outpath, args.trim_dbfs)
        total_length += old_len
        final_length += new_len
    print(f"inital total length: {sec_to_formatted_time(total_length)}, final total length: {sec_to_formatted_time(final_length)}, trimmed total length: {sec_to_formatted_time(total_length - final_length)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trim_audio.py python script to manually preprocess (trim) directory containing wav")
    parser.add_argument("--input_dir", "-i", required=True, type=str, help="directory containing wav files. will be searched recursively")
    parser.add_argument("--output_dir", "-o", required=True, type=str, help="directory where to store trimmed files. relative paths of input files is not preserved")
    parser.add_argument("--trim_dbfs", type=float, help="decibel level threshold for trimming audio", default=-50)
    args = parser.parse_args()
    trim_audio(args)
