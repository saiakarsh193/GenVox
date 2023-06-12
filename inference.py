import os
import torch
import scipy.io
from typing import Optional
from typeguard import typechecked

from audio import db_to_amplitude, get_mel_filter, get_inverse_mel_filter, mel2fft, combine_magnitude_phase, istft, normalize_signal, griffin_lim, reduce_noise
from config import load_configs, TextConfig, AudioConfig, ModelConfig
from processors import TextProcessor
from utils import saveplot_mel, saveplot_signal, saveplot_gate, saveplot_alignment
import tts
import vocoder


class TTSModel:
    """
    TTS class for inference
    """
    @typechecked
    def __init__(self, config_path: str, model_path: str, vocoder_config_path: Optional[str] = None, vocoder_model_path: Optional[str] = None, use_cuda: bool = False):
        self.config_path = config_path
        assert os.path.isfile(self.config_path), f"config_path ({self.config_path}) does not exist"
        self.model_path = model_path
        assert os.path.isfile(self.model_path), f"model_path ({self.model_path}) does not exist"
        self.vocoder_config_path = vocoder_config_path
        self.vocoder_model_path = vocoder_model_path
        self.use_vocoder = False
        if (self.vocoder_config_path or self.vocoder_model_path):
            assert os.path.isfile(self.vocoder_config_path), f"vocoder_config_path ({self.vocoder_config_path}) does not exist"
            assert os.path.isfile(self.vocoder_model_path), f"vocoder_model_path ({self.vocoder_model_path}) does not exist"
            self.use_vocoder = True
            print("using vocoder for inference")
        self.use_cuda = use_cuda
        if (self.use_cuda):
            assert torch.cuda.is_available(), "torch CUDA is not available"
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        print(f"running inference on device: {self.device}")

        # loading TTS configs
        configs = load_configs(self.config_path)
        self.text_config: TextConfig = configs['text_config']
        self.audio_config: AudioConfig = configs['audio_config']
        self.model_config: ModelConfig = configs['model_config']
        assert self.model_config.task == "TTS", f"given config_path ({self.config_path}) is not TTS config"
        # loading processor
        self.text_processor = TextProcessor(self.text_config)
        self.token_map = self.model_config.symbols
        # creating model instance
        if self.model_config.model_name == "Tacotron2":
            self.model = tts.tacotron2.Tacotron2(self.model_config, self.audio_config, self.use_cuda)
        self.model.to(self.device)
        # loading model params
        model_dict = torch.load(self.model_path, map_location=self.device)['model_state_dict']
        self.model.load_state_dict(model_dict)
        self.model.eval()

        # loading VOC
        if self.use_vocoder:
            configs = load_configs(self.vocoder_config_path)
            self.vocoder_audio_config: AudioConfig = configs['audio_config']
            self.vocoder_model_config: ModelConfig = configs['model_config']
            assert self.audio_config.sampling_rate == self.vocoder_audio_config.sampling_rate, "sampling_rate in TTS and VOC Audio config dont match"
            assert self.audio_config.n_mels == self.vocoder_audio_config.n_mels, "n_mels in TTS and VOC Audio config dont match"
            assert self.audio_config.filter_length == self.vocoder_audio_config.filter_length, "filter_length in TTS and VOC Audio config dont match"
            assert self.audio_config.hop_length == self.vocoder_audio_config.hop_length, "hop_length in TTS and VOC Audio config dont match"
            assert self.vocoder_model_config.task == "VOC", f"given vocoder_config_path ({self.vocoder_config_path}) is not VOC config"
            if self.vocoder_model_config.model_name == "MelGAN":
                self.vocoder_model = vocoder.melgan.Generator(self.vocoder_audio_config.n_mels)
            self.vocoder_model.to(self.device)
            model_dict = torch.load(self.vocoder_model_path, map_location=self.device)['model_state_dict'][0]
            self.vocoder_model.load_state_dict(model_dict)
            self.vocoder_model.eval()
    
    @typechecked
    def __call__(self, text: str):
        with torch.no_grad():
            print(text)
            tokens = self.text_processor.tokenize(text)
            tokens = [self.token_map[tk] for tk in tokens if tk in self.token_map]
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
        if not self.use_vocoder:
            mel_mag = db_to_amplitude(mel, log_func=self.audio_config.log_func, ref=self.audio_config.ref_level_db, power=False, scale=1)
            if not hasattr(self, 'mel_basis'):
                self.mel_basis = get_mel_filter(fs=self.audio_config.sampling_rate, n_fft=self.audio_config.filter_length, n_mels=self.audio_config.n_mels, fmin=self.audio_config.mel_fmin, fmax=self.audio_config.mel_fmax)
                self.inverse_basis = get_inverse_mel_filter(self.mel_basis)
            mag = mel2fft(mel_mag, self.inverse_basis)
            ang = griffin_lim(mag, n_fft=self.audio_config.filter_length, hop_length=self.audio_config.hop_length)
            st_comb = combine_magnitude_phase(mag, ang)
            ist = istft(st_comb, n_fft=self.audio_config.filter_length, hop_length=self.audio_config.hop_length)
            ist[(ist > 1) | (ist < -1)] = 0
            sig = normalize_signal(ist[500: -500])
            sig = reduce_noise(sig, self.audio_config.sampling_rate)
            return self.audio_config.sampling_rate, sig
        else:
            mel = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            audio = self.vocoder_model.inference(mel, self.vocoder_audio_config.hop_length)
            return self.vocoder_audio_config.sampling_rate, audio.detach().cpu().numpy()


if __name__ == "__main__":
    # tts = TTSModel(
    #     'old_exp/tac2_lj_74k.yaml',
    #     'old_exp/tac2_lj_74k.pt',
    #     'old_exp/melg_lj.yaml',
    #     'old_exp/melg_lj.pt',
    #     False)
    # mel = tts('with the active cooperation of the responsible agencies and with the understanding of the people of the United States in their demands upon their President')
    tts = TTSModel(
        'old_exp/tac2_hin_m.yaml',
        'old_exp/tac2_hin_m.pt',
        'old_exp/melg_lj.yaml',
        'old_exp/melg_lj.pt',
        use_cuda=False)
    mel = tts('से अधिक अन्य भाषाओं के बीच शब्दों, वाक्यांशों और वेब पृष्ठों का तुरंत अनुवाद करती है।')
    saveplot_mel(mel, 'inf_mel.png')
    fs, wav = tts.mel2audio(mel)
    saveplot_signal(wav, 'inf_sig.png')
    scipy.io.wavfile.write('inf_out.wav', fs, wav)
