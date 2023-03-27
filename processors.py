import os
import tqdm
import subprocess
from pathlib import Path
from typeguard import typechecked

from config import DatasetConfig, AudioConfig


class DatasetProcessor:
    """
    For extracting text from dataset transcript and creating i2t and i2w files
    """
    @typechecked
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def __call__(self):
        print(self.config)
        if (not os.path.isdir("dump")):
            os.mkdir("dump")
        if (self.config.dataset_type == "text"):
            f_text = open("dump/i2t", 'w')
            with open(self.config.transcript_path, 'r') as f:
                raw_text = f.readlines()
            for rt in sorted(raw_text):
                rt = rt.strip().split(self.config.delimiter)
                uid = rt[self.config.uid_index]
                utt = rt[self.config.utt_index]
                f_text.write(f"{uid}\t{utt}\n")
            f_text.close()

            f_wav = open("dump/i2w", 'w')
            for wav_path in sorted(Path(self.config.wavs_path).rglob("*.wav")):
                f_wav.write(f"{wav_path.stem}\t{wav_path}\n")
            f_wav.close()


class AudioProcessor:
    """
    for processing all the audio files
    """
    @typechecked
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def __call__(self):
        print(self.config)
        assert os.path.isdir("dump"), "dump directory does not exist"
        # ffmpeg -i data/LJSpeech_sample/wavs/LJ001-0154.wav -acodec pcm_u8 -ac 1 -ar 16000 temp.wav

        with open("dump/i2w", 'r') as f:
            i2w = f.readlines()
        for wav_path in i2w:
            print(wav_path)