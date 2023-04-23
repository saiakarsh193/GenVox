import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io

from audio import db_to_amplitude, mel2fft, combine_magnitude_phase, istft, normalize_signal
from config import load_config_from_file
from processors import TextProcessor
from utils import saveplot_mel, saveplot_signal, saveplot_gate, saveplot_alignment
import tacotron2


class TTSModel:
    """
    TTS class for inference
    """
    def __init__(self, config_path, model_path, use_cuda=False):
        self.config_path = config_path
        assert os.path.isfile(self.config_path), f"config_path ({self.config_path}) does not exist"
        self.model_path = model_path
        # assert os.path.isfile(self.model_path), f"model_path ({self.model_path}) does not exist"
        self.use_cuda = use_cuda
        if (self.use_cuda):
            assert torch.cuda.is_available(), "torch CUDA is not available"
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        print(f"running inference on device: {self.device}")
        # loading configs
        text_config, audio_config, dataset_config, trainer_config, tacotron2_config, optimizer_config = load_config_from_file(self.config_path)
        self.text_config = text_config
        self.audio_config = audio_config
        self.tacotron2_config = tacotron2_config
        print(self.text_config)
        print(self.audio_config)
        print(self.tacotron2_config)
        # loading processor
        self.text_processor = TextProcessor(self.text_config)
        self.token_map = self.tacotron2_config.symbols
        # creating model instance
        self.model = tacotron2.Tacotron2(self.tacotron2_config, self.audio_config, self.use_cuda)
        self.model.to(self.device)
        # loading model params
        model_dict = torch.load(self.model_path, map_location=self.device)['model_state_dict']
        self.model.load_state_dict(model_dict)
        self.model.eval()
    
    def __call__(self, text: str):
        with torch.no_grad():
            print(text)
            tokens = self.text_processor.tokenize(text)
            tokens = [self.token_map[tk] for tk in tokens]
            tokens = torch.IntTensor(tokens).unsqueeze(0).to(self.device)
            y_pred = self.model.inference(tokens)
            mel, mel_postnet, gate, alignments = y_pred
            mel_postnet = mel_postnet.squeeze(0)
            gate = gate.squeeze(0)
            alignments = alignments.squeeze(0)
            mel_postnet = mel_postnet.cpu().numpy()
            gate_pred = torch.sigmoid(gate).cpu().numpy().reshape(-1)
            alignments = alignments.cpu().numpy().T
        saveplot_gate(None, gate_pred, 'inf_gate.png', title=True)
        saveplot_alignment(alignments, 'inf_align.png', title=True)
        return mel_postnet

    def mel2audio(self, mel):
        mel_mag = db_to_amplitude(mel, log_func=self.audio_config.log_func, ref=self.audio_config.ref_level_db)
        mag = mel2fft(mel_mag, fs=self.audio_config.sampling_rate, n_fft=self.audio_config.filter_length, n_mels=self.audio_config.n_mels, fmin=self.audio_config.mel_fmin, fmax=self.audio_config.mel_fmax)
        ang = np.random.random(mag.shape).astype(np.float32)
        st_comb = combine_magnitude_phase(mag, ang)
        ist = istft(st_comb, n_fft=self.audio_config.filter_length, hop_length=self.audio_config.hop_length)
        ist[(ist > 1) | (ist < -1)] = 0
        sig = normalize_signal(ist[500: -500])
        return self.audio_config.sampling_rate, sig


if __name__ == "__main__":
    tts = TTSModel('config.json', 'exp/checkpoint_9500.pt', False)
    mel = tts('hello world this is a sample sentence')
    saveplot_mel(mel, 'inf_mel.png')
    fs, wav = tts.mel2audio(mel)
    saveplot_signal(wav, 'inf_sig.png')
    scipy.io.wavfile.write('inf_out.wav', fs, wav)
