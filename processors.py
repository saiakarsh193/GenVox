import os
import tqdm
import subprocess
import shutil
import scipy.io
from pathlib import Path
from typeguard import typechecked

from config import DownloadConfig, DatasetConfig, AudioConfig
from utils import get_random_HEX_name, sec_to_formatted_time
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
            temp_directory_path = os.path.join(self.config.directory_path, f"temp_{get_random_HEX_name()}")
            os.mkdir(temp_directory_path)
            createDatasetFromYoutube(self.config.youtube_link, self.config.directory_path, temp_directory_path, self.config.speaker_id, self.config.verbose)
            shutil.rmtree(temp_directory_path)
        else:
            command = ["wget", self.config.download_link, "-P", self.config.directory_path]
            print("running subprocess command: {comm}".format(comm=" ".join(command)))
            subprocess.run(command, capture_output=(not self.config.verbose))
            file_path = os.path.join(self.config.directory_path, os.path.basename(self.config.download_link))
            file_ext = os.path.splitext(file_path)[1]
            if (file_ext == ".zip"):
                command = ["unzip", file_path, "-d", self.config.directory_path]
            elif (file_ext == ".tar"):
                command = ["tar", "-xzf", file_path, "-C", self.config.directory_path]
            else:
                raise ValueError(f"file_extension ({file_ext}) is not supported for extraction")
            print("running subprocess command: {comm}".format(comm=" ".join(command)))
            subprocess.run(command, capture_output=(not self.config.verbose))


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
            f_text = open("dump/i2at", 'w')
            with open(self.config.transcript_path, 'r') as f:
                raw_text = f.readlines()
            for rt in sorted(raw_text):
                rt = rt.strip().split(self.config.delimiter)
                utt_id = rt[self.config.uid_index]
                utt = rt[self.config.utt_index]
                f_text.write(f"{utt_id}\t{utt}\n")
            f_text.close()

            f_wav = open("dump/i2aw", 'w')
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

        wav_dump = os.path.join("dump", "wavs")
        assert not os.path.isdir(wav_dump), f"wav_dump ({wav_dump}) directory already exists"
        os.mkdir(wav_dump)

        with open("dump/i2at", 'r') as f_text, open("dump/i2aw", 'r') as f_wav:
            i2at = f_text.readlines()
            i2aw = f_wav.readlines()

        total_count = 0
        total_length = 0
        valid_count = 0
        valid_length = 0
        valid_ids = {}
        for line in i2aw:
            utt_id, wav_path = line.strip().split("\t")
            fs, wav = scipy.io.wavfile.read(wav_path)
            wav_len = wav.shape[0] / fs
            total_count += 1
            total_length += wav_len
            if (self.config.min_wav_duration <= wav_len and wav_len <= self.config.max_wav_duration):
                valid_ids[utt_id] = {"wav_path": wav_path}
                valid_count += 1
                valid_length += wav_len

        for line in i2at:
            utt_id, text = line.strip().split("\t")
            if utt_id in valid_ids:
                valid_ids[utt_id]["text"] = text

        for utt_id, values in tqdm.tqdm(valid_ids.items()):
            wav_name = os.path.basename(values["wav_path"])
            new_wav_path = os.path.join(wav_dump, wav_name)
            command = ["ffmpeg", "-i", values["wav_path"], "-acodec", "pcm_u8", "-ac", "1", "-ar", str(self.config.sampling_rate), new_wav_path]
            subprocess.run(command, capture_output=True)
            fs, wav = scipy.io.wavfile.read(new_wav_path)
            valid_ids[utt_id]["wav_path"] = new_wav_path
            valid_ids[utt_id]["wav_shape"] = wav.shape[0]
            valid_ids[utt_id]["wav_length"] = wav.shape[0] / fs

        print(f"removed long and short wav files -> before: {total_count} ({sec_to_formatted_time(total_length)}) after: {valid_count} ({sec_to_formatted_time(valid_length)}) removed: {total_count - valid_count} ({sec_to_formatted_time(total_length - valid_length)})")

        with open("dump/i2t", 'w') as f_text, open("dump/i2w", 'w') as f_wav:
            for utt_id, values in valid_ids.items():
                f_text.write("{utt_id}\t{text}\n".format(utt_id=utt_id, text=values["text"]))
                f_wav.write("{utt_id}\t{wav_path}\t{wav_shape}\n".format(utt_id=utt_id, wav_path=values["wav_path"], wav_shape=values["wav_shape"]))
