import os
import tqdm
import subprocess
import shutil
import random
import scipy.io
import numpy as np
from pathlib import Path
from typeguard import typechecked
import g2p_en

from config import DownloadConfig, TextConfig, AudioConfig, DatasetConfig
from utils import get_random_HEX_name, sec_to_formatted_time, get_silent_signal_ind, dump_json, load_json
from create_dataset import createDatasetFromYoutube
from text import base_cleaners, symbols
from audio import normalize_signal, stft, get_mel_filter, get_inverse_mel_filter, fft2mel, amplitude_to_db

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
        self.cleaner_map = {
            "base_cleaners": lambda text: base_cleaners(text, self.config.language)
            }
        self.g2p_map = {
            "english": g2p_en.G2p()
        }
    
    def tokenize(self, text):
        for cleaner in self.config.cleaners:
            text = self.cleaner_map[cleaner](text)
        if (self.config.use_g2p):
            tokens = self.g2p_map[self.config.language](text)
        else:
            tokens = list(text)
        return tokens
    
    def get_token_map(self, token_set):
        # symbol: index
        token_map = {sym: ind for ind, sym in enumerate(symbols)}
        count = len(token_map)
        for sym in token_set:
            if (sym not in token_map):
                token_map[sym] = count
                count += 1
        return token_map


class AudioProcessor:
    """
    For formatting and preprocessing audio
    """
    @typechecked
    def __init__(self, config: AudioConfig):
        self.config = config
        self.mel_basis = get_mel_filter(fs=self.config.sampling_rate, n_fft=self.config.filter_length, n_mels=self.config.n_mels, fmin=self.config.mel_fmin, fmax=self.config.mel_fmax)

    def format(self, input_path, output_path):
        command = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(self.config.sampling_rate), output_path]
        subprocess.run(command, capture_output=True)

    def convert2mel(self, input_path, output_path):
        fs, signal = scipy.io.wavfile.read(input_path)
        assert fs == self.config.sampling_rate, f"wav file ({input_path}) sampling rate ({fs}) does not match with config ({self.config.sampling_rate})"
        if (self.config.normalize):
            signal = normalize_signal(signal)
        spectrogram = stft(signal, n_fft=self.config.filter_length, hop_length=self.config.hop_length)
        mel_spectrogram = fft2mel(np.abs(spectrogram), self.mel_basis)
        mel_db = amplitude_to_db(mel_spectrogram, log_func=self.config.log_func, ref=self.config.ref_level_db, power=False, scale=1)
        np.save(output_path, mel_db)

class DatasetProcessor:
    """
    For extracting and tokenizing text from transcript and
    formatting and extracting features from audio to create dump wav and feature files and data*.csv
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
        dump_dir = self.config.dump_dir
        wav_dump_dir = os.path.join(dump_dir, "wavs")
        feature_dump_dir = os.path.join(dump_dir, "feats")
        assert not os.path.isdir(dump_dir), f"dump ({dump_dir}) directory already exists"
        os.mkdir(dump_dir)
        os.mkdir(wav_dump_dir)
        os.mkdir(feature_dump_dir)

        # extracting text data
        text_data = []
        if (self.config.dataset_type == "text"):
            with open(self.config.transcript_path, 'r') as f:
                raw_text = f.readlines()
            for text in sorted(raw_text):
                text = text.strip().split(self.config.delimiter)
                utt_id = text[self.config.uid_index]
                if (utt_id.find(".") >= 0): # if it ends with .wav then we remove that
                    utt_id = utt_id[: utt_id.find(".")]
                utt = text[self.config.utt_index]
                text_data.append((utt_id, utt))
        elif (self.config.dataset_type == "json"):
            raw_json = load_json(self.config.transcript_path)
            for sample in raw_json:
                utt_id = sample[self.config.uid_keyname]
                if (utt_id.find(".") >= 0): # if it ends with .wav then we remove that
                    utt_id = utt_id[: utt_id.find(".")]
                utt = sample[self.config.utt_keyname]
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
                    scipy.io.wavfile.write(new_wav_path + "_", fs, wav[left_ind: right_ind])
                    self.audio_processor.format(new_wav_path + "_", new_wav_path)
                    os.remove(new_wav_path + "_")
                else:
                    self.audio_processor.format(wav_data[i][1], new_wav_path)
                feature_name = os.path.splitext(wav_name)[0] + ".npy"
                feature_path = os.path.join(feature_dump_dir, feature_name)
                self.audio_processor.convert2mel(new_wav_path, feature_path)
                # utt_id, text, wav_path, feat_path
                valid_data.append((text_data[i][0], text_data[i][1], new_wav_path, feature_path))

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
        token_map = self.text_processor.get_token_map(token_set)

        # updating token with index and creating final data
        final_data = []
        for i in range(len(valid_data)):
            tokens = total_tokens[i]
            tokens_ind = " ".join([str(token_map[token]) for token in tokens])
            final_data.append(f"{valid_data[i][0]}|{valid_data[i][2]}|{valid_data[i][3]}|{tokens_ind}")

        print(f"tokenized text and updated with token index, token vocabulary size: {len(token_set)}")

        dump_json(os.path.join(dump_dir, "token_list.json"), token_map)
        
        all_index = set(range(len(final_data)))
        if type(self.config.validation_split) == int:
            validation_count = self.config.validation_split
            assert validation_count < len(all_index), f"validation sample count ({validation_count}) is more than total data count ({len(all_index)})"
        else:
            validation_count = int(self.config.validation_split * len(all_index))
        validation_index = set(random.sample(all_index, k=validation_count))
        train_index = all_index - validation_index

        with open(os.path.join(dump_dir, "data.csv"), 'w') as f:
            for index in all_index:
                f.write(f"{final_data[index]}\n")

        with open(os.path.join(dump_dir, "data_train.csv"), 'w') as f:
            for index in train_index:
                f.write(f"{final_data[index]}\n")

        with open(os.path.join(dump_dir, "data_validation.csv"), 'w') as f:
            for index in validation_index:
                f.write(f"{final_data[index]}\n")

        print(f"split data ({len(all_index)}) into train ({len(train_index)}) and validation ({len(validation_index)})")
