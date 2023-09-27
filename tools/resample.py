import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # file: GenVox/tools/resample.py  -> parent.parent: GenVox/
import os
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.experimental import function_timer
from core.processors import AudioProcessor
from configs import AudioConfig

@function_timer
def resample(input_dir: str, output_dir: str, sampling_rate: int, n_jobs: int):
    assert not os.path.isdir(output_dir), f"output directory {output_dir} already exists"
    os.mkdir(output_dir)
    tasks = []
    for wav_path in sorted(Path(input_dir).rglob("*.wav")): # wav_path, wav_path.parent, wav_path.name, wav_path.stem, wav_path.suffix
        out_wav_path = os.path.join(output_dir, os.path.relpath(wav_path, input_dir))
        par_dir = os.path.dirname(out_wav_path)
        if not os.path.isdir(par_dir):
            os.makedirs(par_dir)
        tasks.append((str(wav_path), out_wav_path))
    audio_processor = AudioProcessor(config=AudioConfig(sampling_rate=sampling_rate))
    def resample_func(input_path, output_path):
        audio_processor.format_audio2wav(input_path=input_path, output_path=output_path)
    Parallel(n_jobs=n_jobs)(delayed(resample_func)(input_path, output_path) for (input_path, output_path) in tqdm(tasks))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="resample.py can resample all the wavs in a given directory")
    parser.add_argument("--input_dir", "-i", required=True, type=str, help="directory containing wav files. will be searched recursively")
    parser.add_argument("--output_dir", "-o", required=True, type=str, help="directory where to store resampled files. file structure is maintained")
    parser.add_argument("--sampling_rate", "-fs", type=int, help="required sampling rate of wav files", default=22050)
    parser.add_argument("--n_jobs", "-nj", type=int, help="number of jobs for parallel processing", default=10)
    args = parser.parse_args()
    resample(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        n_jobs=args.n_jobs
    )
