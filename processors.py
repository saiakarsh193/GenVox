import os
import tqdm
import subprocess
import shutil
import scipy.io
from pathlib import Path
from typeguard import typechecked

from config import DownloadConfig, TextConfig, AudioConfig, DatasetConfig
from utils import get_random_HEX_name, sec_to_formatted_time, get_silent_signal_ind, dump_json
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


class TextProcessor:
    """
    For preprocessing text
    """
    @typechecked
    def __init__(self, config: TextConfig) -> None:
        self.config = config
    
    def tokenize(self, text):
        tokens = list(text.lower())
        return tokens
    
    def get_symbols_extra(self):
        pass


class AudioProcessor:
    """
    For formatting and preprocessing audio
    """
    @typechecked
    def __init__(self, config: AudioConfig):
        self.config = config

    def format(self, input_path, output_path):
        command = ["ffmpeg", "-y", "-i", input_path, "-acodec", "pcm_u8", "-ac", "1", "-ar", str(self.config.sampling_rate), output_path]
        subprocess.run(command, capture_output=True)
    
    def preprocess(self):
        pass


class DatasetProcessor:
    """
    For extracting text from dataset transcript, tokenizing text, formatting audio, filtering data and creating data.csv file and dump wav files
    """
    @typechecked
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.text_config = self.config.text_config
        self.audio_config = self.config.audio_config
        self.text_processor = TextProcessor(self.text_config)
        self.audio_processor = AudioProcessor(self.audio_config)
    
    def __call__(self):
        print(self.config)
        dump_dir = "dump"
        wav_dump_dir = os.path.join("dump", "wavs")
        assert not os.path.isdir(dump_dir), f"dump ({dump_dir}) directory already exists"
        os.mkdir("dump")
        assert not os.path.isdir(wav_dump_dir), f"wav_dump ({wav_dump_dir}) directory already exists"
        os.mkdir(wav_dump_dir)

        # extracting text data
        text_data = []
        if (self.config.dataset_type == "text"):
            with open(self.config.transcript_path, 'r') as f:
                raw_text = f.readlines()
            for text in sorted(raw_text):
                text = text.strip().split(self.config.delimiter)
                utt_id = text[self.config.uid_index]
                utt = text[self.config.utt_index]
                text_data.append((utt_id, utt))
        # extracting wav data
        wav_data = []
        for wav_path in sorted(Path(self.config.wavs_path).rglob("*.wav")):
            utt_id = wav_path.stem
            wav_data.append((utt_id, str(wav_path)))

        # simple checks for data validation
        assert len(text_data) == len(wav_data), f"number of utterances ({len(text_data)}) and number of wavs ({len(wav_data)}) found do not match"
        # since already sorted we can linearly check if everything is matching
        for i in range(len(text_data)):
            assert text_data[i][0] == wav_data[i][0], f"utterance id ({text_data[i][0]}) and the wav name ({wav_data[i][0]}) do not match"
        
        # filtering data by trimming silence and checking length
        valid_data = []
        total_length = 0
        valid_length = 0
        for i in tqdm.tqdm(range(len(text_data))):
            fs, wav = scipy.io.wavfile.read(wav_data[i][1])
            total_length += wav.shape[0] / fs
            if (self.audio_config.trim_silence):
                left_ind, right_ind = get_silent_signal_ind(wav, fs, silence_threshold=self.audio_config.trim_dbfs)
                wav_len = (right_ind - left_ind) / fs
            else:
                wav_len = wav.shape[0] / fs
            if (self.audio_config.min_wav_duration <= wav_len and wav_len <= self.audio_config.max_wav_duration):
                valid_length += wav_len
                wav_name = os.path.basename(wav_data[i][1])
                new_wav_path = os.path.join(wav_dump_dir, wav_name)
                if (self.audio_config.trim_silence):
                    scipy.io.wavfile.write(new_wav_path, fs, wav[left_ind: right_ind])
                    self.audio_processor.format(new_wav_path, new_wav_path)
                else:
                    self.audio_processor.format(wav_data[i][1], new_wav_path)
                valid_data.append((text_data[i][0], text_data[i][1], new_wav_path))
        
        print("trimmed and removed long and short wav files -> before: {bf_cnt} ({bf_len}), after: {af_cnt} ({af_len}), removed: {rm_cnt} ({rm_len})".format(
            bf_cnt=len(text_data),
            bf_len=sec_to_formatted_time(total_length),
            af_cnt=len(valid_data),
            af_len=sec_to_formatted_time(valid_length),
            rm_cnt=len(text_data) - len(valid_data),
            rm_len=sec_to_formatted_time(total_length - valid_length)
        ))

        # tokenizing text
        token_set = set()
        total_tokens = []
        for i in range(len(valid_data)):
            tokens = self.text_processor.tokenize(valid_data[i][1])
            token_set.update(tokens)
            total_tokens.append(tokens)
        token_map = {val: ind for ind, val in enumerate(token_set)}

        # updating token with index and creating final data
        final_data = []
        for i in range(len(valid_data)):
            tokens = total_tokens[i]
            tokens_ind = [str(token_map[tk]) for tk in tokens]
            final_data.append((valid_data[i][2], " ".join(tokens_ind)))

        print(f"tokenized text and updated with token index, token vocabulary size: {len(token_set)}")

        dump_json(os.path.join(dump_dir, "token_list.json"), token_map)
        with open(os.path.join(dump_dir, "data.csv"), 'w') as f:
            for text, wav_path in final_data:
                f.write(f"{text}|{wav_path}\n")
