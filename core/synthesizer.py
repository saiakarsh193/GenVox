import torch
import numpy as np
from typing import Dict

from models.tts import TTSModel
from core.processors import TextProcessor, AudioProcessor

class Synthesizer:
    def __init__(
            self,
            tts_model_class: TTSModel,
            tts_config_path: str,
            tts_checkpoint_path: str,
            use_cuda: bool = False
        ) -> None:
        self.device = "cuda:0" if (use_cuda & torch.cuda.is_available()) else "cpu"
        self.tts_model = tts_model_class.load_from_config(config_path=tts_config_path)
        self.tts_model.to(self.device)
        self.tts_model.eval()
        print(f"using TTS model: {self.tts_model.model_name}, device: {self.device}")
        tts_model_dict = torch.load(tts_checkpoint_path, map_location=self.device)
        print("loading tts_model_dict (iteration: {itr}) from checkpoint_path {chk_path}".format(itr=tts_model_dict["iteration"], chk_path=tts_checkpoint_path))
        self.tts_model.load_checkpoint_statedicts(statedicts=tts_model_dict, save_optimizer_dict=False, optimizer=None)
        self.text_processor = TextProcessor(config=self.tts_model.text_config)
        self.audio_processor = AudioProcessor(config=self.tts_model.audio_config)

    def tts(
            self,
            text: str
        ) -> Dict[str, np.ndarray]:
        tokens = self.text_processor.tokenize(text)
        tokens = self.text_processor.tokens_to_indices(tokens)
        tokens = torch.IntTensor(tokens).unsqueeze(0).to(self.device)
        outputs = self.tts_model.inference(
            inputs={
                "tokens": tokens
            }
        )
        outputs = {key: val.squeeze(0).cpu().numpy() for key, val in outputs.items()}
        if "mel_outputs_postnet" in outputs:
            mel_spec_pred = outputs["mel_outputs_postnet"]
        fs, wav = self.audio_processor.convert_mel2wav(mel=mel_spec_pred)
        outputs["waveform"] = wav
        outputs["sampling_rate"] = fs
        return outputs
