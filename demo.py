import scipy.io
from core.synthesizer import Synthesizer
from models.tts.tacotron2 import Tacotron2
from utils.plotting import saveplot_mel

syn = Synthesizer(
    tts_model_class=Tacotron2,
    tts_config_path="exp_temp/config.yaml",
    tts_checkpoint_path="exp_temp/checkpoint_8600.pt"
)

outputs = syn.tts(
    text="token_map not yet generated, use TextProcessor.generate_token_map() to generate it"
)
print(outputs.keys())
print(outputs["mel_outputs_postnet"].shape)
saveplot_mel(outputs["mel_outputs_postnet"], path="pred_mel_spec.png")
print(outputs["sampling_rate"], outputs["waveform"].shape)
scipy.io.wavfile.write('pred_sig.wav', outputs["sampling_rate"], outputs["waveform"])
