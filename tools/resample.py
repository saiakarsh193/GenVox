import argparse
from pathlib import Path
from multiprocessing import Pool
import os
import sys
# file: GenVox/tools/resample.py  -> parent.parent: GenVox/
sys.path.append(str(Path(__file__).parent.parent))
import tqdm

from utils import function_timer
from processors import AudioProcessor
from config import AudioConfig

@function_timer
def resample(args):
    assert not os.path.isdir(args.output_dir), f"output directory {args.output_dir} already exists"
    os.mkdir(args.output_dir)
    tasks = []
    for wav_path in sorted(Path(args.input_dir).rglob("*.wav")):
        # wav_path, wav_path.parent, wav_path.name, wav_path.stem, wav_path.suffix
        outpath = os.path.join(args.output_dir, wav_path.name)
        tasks.append((wav_path, outpath))
    audio_config = AudioConfig(sampling_rate=args.sampling_rate)
    audio_processor = AudioProcessor(audio_config)
    # with Pool(processes=5) as p:
    for inpath, outpath in tqdm.tqdm(tasks):
        audio_processor.format(inpath, outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="resample.py python script to manually preprocess (resample) directory containing wav")
    parser.add_argument("--input_dir", "-i", required=True, type=str, help="directory containing wav files. will be searched recursively")
    parser.add_argument("--output_dir", "-o", required=True, type=str, help="directory where to store resampled files. relative paths of input files is not preserved")
    parser.add_argument("--sampling_rate", "-fs", type=int, help="required sampling rate of wav files", default=22050)
    args = parser.parse_args()
    resample(args)
