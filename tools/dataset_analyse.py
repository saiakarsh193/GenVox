import argparse
import scipy.io
from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def analyze_data(args):
    g_fs = -1
    fs_same_flag = True
    g_wav_lens = []
    for wav_path in tqdm.tqdm(list(Path(args.input_dir).rglob("*.wav"))):
        fs, wav = scipy.io.wavfile.read(wav_path)
        wav_len = wav.shape[0] / fs
        if (g_fs == -1):
            g_fs = fs
        elif (g_fs != fs):
            fs_same_flag = False
        g_wav_lens.append(int(wav_len))
    
    if not fs_same_flag:
        print("WARNING: fs (sampling rate) was not same for every wav file found in the directory")
    
    g_wav_lens = np.array(g_wav_lens)
    max_len = np.max(g_wav_lens)
    bin_intervals = np.linspace(0, max_len, args.len_nbins + 1)
    bin_label = ((bin_intervals[: -1] + bin_intervals[1: ]) / 2)
    bins = np.zeros(args.len_nbins, dtype=int)

    for wav_len in g_wav_lens:
        bin_ind = int((wav_len / max_len) * (args.len_nbins - 1))
        bins[bin_ind] += 1

    plt.figure(figsize=[10, 10])
    plt.suptitle("Audio Length Distribution", fontsize=15)
    plt.subplot(221)
    plt.bar(bin_label, bins)
    plt.ylabel("count")
    plt.xlabel("length (s)")

    plt.subplot(222)
    bins_tlen = bins * bin_label
    plt.bar(bin_label, bins_tlen)
    plt.ylabel("total length (s)")
    plt.xlabel("length (s)")

    plt.subplot(223)
    bins_percentage = (bins_tlen / np.sum(g_wav_lens)) * 100
    plt.bar(bin_label, bins_percentage)
    plt.ylabel("percentage (total length)")
    plt.xlabel("length (s)")

    plt.subplot(224)
    plt.pie(bins_percentage, labels = (bin_label).astype(int), startangle = 90)
    plt.savefig(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataset_analyze.py python script to perform data analysis on directory containing wav")
    parser.add_argument("input_dir", type=str, help="directory containing wav files. will be searched recursively")
    parser.add_argument("--output_path", "-o", type=str, help="output path to save the image", default="dataset_analysis.png")
    parser.add_argument("--len_nbins", type=int, help="number of bins while plotting bar graph for lengths", default=30)
    args = parser.parse_args()
    analyze_data(args)
