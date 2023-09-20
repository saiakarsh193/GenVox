import matplotlib.pyplot as plt
import numpy as np

def saveplot_mel(mel: np.ndarray, path: str, title: bool = False) -> None:
    plt.close()
    plt.figure()
    plt.imshow(mel, aspect='auto', origin='lower')
    if title:
        plt.title("mel spectrogram")
    plt.tight_layout()
    plt.savefig(path)

def saveplot_signal(signal: np.ndarray, path: str, title: bool = False) -> None:
    plt.close()
    plt.figure()
    plt.plot(signal)
    if title:
        plt.title("wav signal")
    plt.tight_layout()
    plt.savefig(path)

def saveplot_alignment(alignment: np.ndarray, path: str, title: bool = False) -> None:
    plt.close()
    plt.figure()
    plt.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    if title:
        plt.title("alignment")
    plt.tight_layout()
    plt.savefig(path)

def saveplot_gate(gate_target: np.ndarray, gate_pred: np.ndarray, path: str, title: bool = False, plot_both: bool = False) -> None:
    plt.close()
    plt.figure()
    if plot_both and type(gate_target) == np.ndarray:
        plt.plot(gate_target, color='blue', alpha=0.5, label='gate target')
    plt.plot(gate_pred, color='red', alpha=0.8, label='gate prediction')
    if title:
        plt.title("gate")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(path)
