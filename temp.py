import scipy.io
import numpy as np
import matplotlib.pyplot as plt

path = "dump/wavs/LJ001-0009.wav"
fs, sig = scipy.io.wavfile.read(path)
print(sig.shape[0], fs, sig.shape[0] / fs)
norm_fac = max(abs(np.min(sig)), abs(np.max(sig)))
print(norm_fac)
sig = (sig / norm_fac).astype(np.float32)
print(np.max(sig), np.min(sig)) # assert
# plt.plot(sig)
# plt.show()
