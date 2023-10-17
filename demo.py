import scipy.io
from core.synthesizer import Synthesizer
from models.tts.tacotron2 import Tacotron2
from models.vocoder.melgan import MelGAN
from utils.plotting import saveplot_mel

syn = Synthesizer(
    tts_model_class=Tacotron2,
    tts_config_path="exp/config.yaml",
    tts_checkpoint_path="exp/checkpoint_8600.pt",
    vocoder_model_class=MelGAN,
    vocoder_config_path="exp_voc/config.yaml",
    vocoder_checkpoint_path="exp_voc/checkpoint_8600.yaml"
)

outputs = syn.tts(
    text="token_map not yet generated, use TextProcessor.generate_token_map() to generate it"
)
print(outputs.keys())

print(outputs["mel"].shape)
saveplot_mel(outputs["mel"], path="pred_mel_spec.png")

print(outputs["sampling_rate"], outputs["waveform"].shape)
scipy.io.wavfile.write('pred_sig.wav', outputs["sampling_rate"], outputs["waveform"])
