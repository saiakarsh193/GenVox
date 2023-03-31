import os
import tqdm
import subprocess
import shutil
from pathlib import Path
from typeguard import typechecked

from config import DownloadConfig, DatasetConfig, AudioConfig
from utils import getRandomHexName
from create_dataset import createDatasetFromYoutube


class DownloadProcessor:
    """
    For downloading or preparing dataset
    """
    @typechecked
    def __init__(self, config: DownloadConfig):
        self.config = config
    
    def __call__(self):
        print(self.config)
        if (self.config.is_youtube):
            temp_directory_path = os.path.join(self.config.directory_path, f"temp_{getRandomHexName()}")
            os.mkdir(temp_directory_path)
            createDatasetFromYoutube(self.config.youtube_link, self.config.directory_path, temp_directory_path, self.config.speaker_id)
            shutil.rmtree(temp_directory_path)
        else:
            command = ["wget", self.config.download_link, "-P", self.config.directory_path]
            print("running subprocess command: {comm}".format(comm=" ".join(command)))
            subprocess.run(command, capture_output=(not self.config.verbose))
            file_path = os.path.join(self.config.directory_path, os.path.basename(self.config.download_link))
            file_ext = os.path.splitext(file_path)[1]
            if (file_ext == ".zip"):
                command = ["unzip", ]
            elif (file_ext == ".tar"):
                command = ["tar"]
            else:
                raise ValueError(f"file_extension ({file_ext}) is not supported for extraction")
            # print("running subprocess command: {comm}".format(comm=" ".join(command)))
            # subprocess.run(command, capture_output=(not self.config.verbose))


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
