import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # file: GenVox/tools/resample.py  -> parent.parent: GenVox/
import scipy.io
import argparse
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from utils import sec_to_formatted_time

def get_audio_length(path):
    fs, wav = scipy.io.wavfile.read(path)
    return wav.shape[0] / fs

def main(path, n_bins, max_threshold, n_jobs):
    w_lens = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(get_audio_length)(wav_path) for wav_path in tqdm(list(Path(path).rglob("*.wav")))
    )
    max_len = max(w_lens)
    total_len = sum(w_lens)
    bin_size = max_len / n_bins
    bins = [0] * n_bins
    binsum = [0] * n_bins
    len_retained = 0

    for w_len in w_lens:
        ind = min(int(w_len / bin_size), n_bins - 1)
        bins[ind] += 1
        binsum[ind] += w_len
        if max_threshold != None and w_len <= max_threshold:
            len_retained += w_len

    print(f"total files: {len(w_lens)}")
    print(f"total len: {sec_to_formatted_time(total_len)} ({total_len:.2f}s)")
    print(f"max len: {max_len:.2f}s")
    print(f"number of bins: {n_bins}")

    for i, b in enumerate(bins):
        print(f"{i + 1:3}: ({i * bin_size:6.2f} -{(i + 1) * bin_size:6.2f}) -> {b:4} | {sec_to_formatted_time(binsum[i])} -> {(binsum[i] * 100 / total_len):5.2f}%")

    if max_threshold != None:
        print(f"max threshold: {max_threshold}s")
        print(f"total len: {sec_to_formatted_time(total_len)} ({total_len:.2f}s)")
        print(f"retained len: {sec_to_formatted_time(len_retained)} ({len_retained:.2f}s) -> {len_retained * 100 / total_len:5.2f}%")
        print(f"lost len: {sec_to_formatted_time(total_len - len_retained)} ({total_len - len_retained:.2f}s) -> {(total_len - len_retained) * 100 / total_len:5.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get wav file length distribution and analysis for given threshold")
    parser.add_argument("path", help="path to directory containing wav files", type=str)
    parser.add_argument("--n_bins", help="number of bins for distribution", type=int, default=20)
    parser.add_argument("--threshold", help="max threshold of length allowed (for analysis)", type=float, default=None)
    parser.add_argument("--n_jobs", help="number of threads for parallel processing", type=int, default=None)
    args = parser.parse_args()
    main(
        path = args.path,
        n_bins = args.n_bins,
        max_threshold = args.threshold,
        n_jobs = args.n_jobs
    )
