import scipy.io
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def signal_to_frames(y, window_length, hop_length, pad = False):
    # total_length = window_length + (n - 1) * hop_length
    count = (y.shape[0] - window_length) // hop_length + 1
    if (pad):
        pad_len = hop_length - (y.shape[0] - (window_length + (count - 1) * hop_length))
        if pad_len > 0:
            y = np.hstack([y, np.zeros(pad_len)])
            count += 1
    frames = []
    for i in range(count):
        frames.append(y[i * hop_length: i * hop_length + window_length])
    return np.vstack(frames)

def normalize_signal(y):
    norm_fac = max(abs(np.min(y)), abs(np.max(y)))
    return (y / norm_fac).astype(np.float32)

def amplitude_to_db(spectrogram, amin = 1e-5, ref = 1, log_func = "np.log10", power = True, scale = 20):
    if (log_func == "np.log"):
        log_func = np.log
    elif (log_func == "np.log10"):
        log_func = np.log10
    magnitude_spectrogram = np.abs(spectrogram)
    if power:
        power_spectrogram = magnitude_spectrogram**2
        db = log_func(np.maximum(amin, power_spectrogram))
    else:
        db = log_func(np.maximum(amin, magnitude_spectrogram))
    db -= log_func(np.maximum(amin, ref))
    return db * scale

def db_to_amplitude(db, amin = 1e-5, ref = 1, log_func = "np.log10", power = True, scale = 20):
    if (log_func == "np.log"):
        log_func = np.log
        inv_log_func = np.exp
    elif (log_func == "np.log10"):
        log_func = np.log10
        inv_log_func = lambda value: np.power(10, value)
    db /= scale
    db += log_func(np.maximum(amin, ref))
    if power:
        power_spectrogram = inv_log_func(db)
        magnitude_spectrogram = np.sqrt(power_spectrogram)
    else:
        magnitude_spectrogram = inv_log_func(db)
    return magnitude_spectrogram

def combine_magnitude_phase(magnitude, phase):
    assert magnitude.shape == phase.shape
    return magnitude * (np.cos(phase) + 1j * np.sin(phase))

def stft(y, n_fft, hop_length):
    # float32 and real array and 1-D signal
    assert y.dtype == np.float32 and y.dtype.kind == 'f' and y.ndim == 1
    # [n x window_length]
    frames = signal_to_frames(y, window_length=n_fft, hop_length=hop_length)
    # window_length
    window = scipy.signal.get_window("hann", n_fft, fftbins=True).astype(np.float32)
    # (1 + n_fft // 2, n)
    stft_matrix = np.zeros((1 + n_fft // 2, frames.shape[0]), dtype=np.complex64)
    for ind, frame in enumerate(frames):
        stft_matrix[: , ind] = np.fft.rfft(window * frame, n_fft)
    return stft_matrix

def istft(stft_matrix, n_fft, hop_length):
    # n_fft x n -> window_length + (n - 1) * hop_length = signal_length
    y = np.zeros(n_fft + (stft_matrix.shape[1] - 1) * hop_length, dtype=np.float32)
    window = scipy.signal.get_window("hann", n_fft, fftbins=True).astype(np.float32)
    for frame_ind in range(stft_matrix.shape[1]):
        y[frame_ind * hop_length: frame_ind * hop_length + n_fft] += window * np.fft.irfft(stft_matrix[:, frame_ind], n_fft)
        # here y is getting window * irfft(spec) => window * (window * signal)
        # because spec = rfft(window * signal)
        # therefore y is getting window^2 * signal, hence we divide by window^2 to get back just signal
    # dividing by window sum square
    window_sq = window**2
    ifft_window_sum = np.zeros(y.shape, dtype=np.float32)
    for frame_ind in range(stft_matrix.shape[1]):
        ifft_window_sum[frame_ind * hop_length: frame_ind * hop_length + n_fft] += window_sq
    # to avoid divide by zero
    non_zero_indices = ifft_window_sum > np.finfo(np.float32).tiny
    y[non_zero_indices] /= ifft_window_sum[non_zero_indices]
    return y

def hz_to_mel(hz):
    # htk method => mel = 2595.0 * np.log10(1.0 + f / 700.0)
    # we are using Slaney instead of htk
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region
    if hz >= min_log_hz:
        mel = min_log_mel + np.log(hz / min_log_hz) / logstep
    else:
        mel = (hz - f_min) / f_sp
    return mel

def mel_to_hz(mel):
    # we are using Slaney instead of htk
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region
    if mel >= min_log_mel:
        hz = min_log_hz * np.exp(logstep * (mel - min_log_mel))
    else:
        hz = f_min + f_sp * mel
    return hz

def get_mel_filter(fs, n_fft, n_mels, fmin, fmax):
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    # 0 to fs / 2 (nyquist theorem) map to (1 + n_fft / 2) bins (due to np.fft.rfft)
    fftfreqs = np.linspace(0, fs / 2, 1 + n_fft // 2)
    # mel mapping
    min_mel, max_mel = hz_to_mel(fmin), hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = np.vectorize(mel_to_hz)(mels)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights

def get_inverse_mel_filter(mel_basis):
    return np.linalg.pinv(mel_basis)

def fft2mel(stft_matrix, mel_basis):
    assert stft_matrix.dtype.kind == 'f' # checking data type is real (f: real, c: complex)
    return np.matmul(mel_basis, stft_matrix) # mel_basis should be output of get_mel_filter

def mel2fft(mel_matrix, inverse_basis):
    assert mel_matrix.dtype.kind == 'f'
    return np.matmul(inverse_basis, mel_matrix) # inverse_basis should be output of get_inverse_mel_filter

def griffin_lim(magnitude_spectrogram, n_fft, hop_length, momentum=0.99, n_iter=32):
    """
    magnitude spectrogram (1 + n_fft / 2, n_frames) -> angles spectrogram (1 + n_fft / 2, n_frames)
    """
    angles = np.ones(magnitude_spectrogram.shape, dtype=np.complex64)
    eps = np.finfo(np.complex64).tiny
    previous_angles = np.zeros(angles.shape)
    angles *= magnitude_spectrogram
    for _ in range(n_iter):
        inverse = istft(angles, n_fft, hop_length)
        rebuilt = stft(inverse, n_fft, hop_length)
        angles = rebuilt - (momentum / (1 + momentum)) * previous_angles
        angles /= np.abs(angles) + eps
        angles *= magnitude_spectrogram
        previous_angles = rebuilt
    return np.angle(angles)


if __name__ == "__main__":
    path = "data/LJSpeech_test/wavs/LJ001-0013.wav"
    fs, sig = scipy.io.wavfile.read(path)
    sig = normalize_signal(sig)

    filter_length = 1024 # same as window_length
    hop_length = 256
    n_mels, fmin, fmax = 80, 0.0, 8000

    tlen = (sig.shape[0] - filter_length) // hop_length + 1
    tsiz = filter_length + (tlen - 1) * hop_length
    sig = sig[: tsiz]

    # print(sig.shape, fs, filter_length, hop_length)
    st = stft(sig, n_fft=filter_length, hop_length=hop_length)
    # print(st.shape)
    mag, ang = np.abs(st), np.angle(st)

    mel_basis = get_mel_filter(fs=fs, n_fft=filter_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    inverse_basis = get_inverse_mel_filter(mel_basis)
    mel_mag = fft2mel(mag, mel_basis)
    mel_db = amplitude_to_db(mel_mag, power=False, scale=1)
    mel_mag = db_to_amplitude(mel_db, power=False, scale=1)
    mag = mel2fft(mel_mag, inverse_basis)

    ang = griffin_lim(mag, n_fft=filter_length, hop_length=hop_length)
    # ang = np.random.random(mag.shape).astype(np.float32)
    st_comb = combine_magnitude_phase(mag, ang)
    ist = istft(st_comb, n_fft=filter_length, hop_length=hop_length)
    ist[(ist > 1) | (ist < -1)] = 0
    # print(ist.shape)

    print(np.mean(np.abs(st) - mag)**2)
    print(np.sum(ist - sig))
    print(np.mean(ist - sig)**2)

    plt.figure()
    plt.subplot(411)
    plt.plot(sig)
    plt.subplot(412)
    plt.imshow(amplitude_to_db(st), aspect='auto', origin='lower')
    plt.subplot(413)
    # plt.plot(ist)
    plt.plot(normalize_signal(ist[500: -500]))
    plt.subplot(414)
    # plt.imshow(np.abs(mel_mag)**2, aspect='auto', origin='lower')
    plt.imshow(mel_db, aspect='auto', origin='lower')
    # plt.imshow(amplitude_to_db(stft(normalize_signal(ist[500: -500]), n_fft=filter_length, hop_length=hop_length)), aspect='auto', origin='lower')
    plt.show()
