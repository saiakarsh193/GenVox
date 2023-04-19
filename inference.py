import os
import torch
import matplotlib.pyplot as plt

from config import load_config_from_file
from processors import TextProcessor
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
        model_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(model_dict)
        self.model.eval()
    
    def __call__(self, text: str):
        with torch.no_grad():
            print(text)
            tokens = self.text_processor.tokenize(text)
            tokens = [self.token_map[tk] for tk in tokens]
            tokens = torch.IntTensor(tokens).unsqueeze(0).to(self.device)
            y_pred = self.model.inference(tokens)
            mel, mel_postnet, *_ = y_pred
            mel_postnet = mel_postnet.squeeze(0)
            mel_postnet = mel_postnet.cpu().numpy()
        return mel_postnet


if __name__ == "__main__":
    tts = TTSModel('config.json', 'exp/checkpoint_2740.pt', True)
    mel = tts('hello world this is a sample sentence')
    plt.figure()
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.savefig('temp.png')
