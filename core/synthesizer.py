import torch
import numpy as np
from typing import Dict, Optional

from models.tts import TTSModel
from models.vocoder import VocoderModel
from core.processors import TextProcessor, AudioProcessor

class Synthesizer:
    def __init__(
            self,
            tts_model_class: TTSModel,
            tts_config_path: str,
            tts_checkpoint_path: str,
            vocoder_model_class: Optional[VocoderModel] = None,
            vocoder_config_path: Optional[str] = None,
            vocoder_checkpoint_path: Optional[str] = None,
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
        if vocoder_model_class != None:
            self.vocoder_model = vocoder_model_class.load_from_config(config_path=vocoder_config_path)
            self.vocoder_model.to(self.device)
            self.vocoder_model.eval()
            print(f"using Vocoder model: {self.vocoder_model.model_name}, device: {self.device}")
            vocoder_model_dict = torch.load(vocoder_checkpoint_path, map_location=self.device)
            print("loading vocoder_model_dict (iteration: {itr}) from checkpoint_path {chk_path}".format(itr=vocoder_model_dict["iteration"], chk_path=vocoder_checkpoint_path))
            self.vocoder_model.load_checkpoint_statedicts(statedicts=vocoder_model_dict, save_optimizer_dict=False, optimizer=None)
        else:
            self.vocoder_model = None

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
        if self.vocoder_model != None:
            voc_outputs = self.vocoder_model.inference(
                inputs={
                    "features": outputs["mel"]
                }
            )
            outputs["waveform"] = voc_outputs["wav"]
            outputs["sampling_rate"] = self.vocoder_model.audio_config.sampling_rate
        outputs = {key: val.squeeze().cpu().numpy() for key, val in outputs.items()}
        if self.vocoder_model == None:
            fs, wav = self.audio_processor.convert_mel2wav(mel=outputs["mel"])
            outputs["waveform"] = wav
            outputs["sampling_rate"] = fs
        return outputs
