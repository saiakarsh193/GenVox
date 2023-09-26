import os
import tqdm
import subprocess
import random
import scipy.io
import numpy as np
import g2p_en
from typing import List, Dict, Callable, Set, Union, Tuple

from configs import TextConfig, AudioConfig
from utils import get_non_silent_boundary, sec_to_formatted_time, dump_json, load_json
from utils.formatters import BaseDataset
from utils.text import base_cleaners
from utils.audio import normalize_signal, get_mel_filter, stft, fft2mel, amplitude_to_db, db_to_amplitude, get_inverse_mel_filter, mel2fft, griffin_lim, combine_magnitude_phase, istft, reduce_noise

# class DownloadProcessor:
#     """
#     For downloading or preparing dataset
#     """
#     def __init__(self, config: DownloadConfig):
#         self.config = config
    
#     def __call__(self):
#         print(self.config)
#         if (self.config.is_youtube):
#             temp_directory_path = os.path.join(self.config.directory_path, f"temp_{get_random_HEX_name()}")
#             os.mkdir(temp_directory_path)
#             createDatasetFromYoutube(self.config.youtube_link, self.config.directory_path, temp_directory_path, self.config.speaker_id, self.config.verbose)
#             shutil.rmtree(temp_directory_path)
#         else:
#             command = ["wget", self.config.download_link, "-P", self.config.directory_path]
#             print("running subprocess command: {comm}".format(comm=" ".join(command)))
#             subprocess.run(command, capture_output=(not self.config.verbose))
#             file_path = os.path.join(self.config.directory_path, os.path.basename(self.config.download_link))
#             file_ext = os.path.splitext(file_path)[1]
#             if (file_ext == ".zip"):
#                 command = ["unzip", file_path, "-d", self.config.directory_path]
#             elif (file_ext == ".tar"):
#                 command = ["tar", "-xzf", file_path, "-C", self.config.directory_path]
#             else:
#                 raise ValueError(f"file_extension ({file_ext}) is not supported for extraction")
#             print("running subprocess command: {comm}".format(comm=" ".join(command)))
#             subprocess.run(command, capture_output=(not self.config.verbose))


class TextProcessor:
    """
    TextProcessor is used to process text and convert it into tokens, and also provides other helper methods.
    """
    def __init__(self, config: TextConfig):
        self.config = config
        self.cleaner_map: Dict[str, Callable[[str], str]] = {
            "base_cleaners": lambda text: base_cleaners(text, self.config.language)
        }
        self.g2p_map: Dict[str, Callable[[str], List[str]]] = {
            "english": g2p_en.G2p()
        }
        self.all_unique_tokens: Set[str] = set()
        self.token_map = self.config.token_map
    
    def tokenize(self, text: str) -> List[str]:
        if isinstance(self.config.cleaners, list):
            for cleaner in self.config.cleaners:
                text = self.cleaner_map[cleaner](text)
        if (self.config.use_g2p):
            tokens = self.g2p_map[self.config.language](text)
        else:
            tokens = list(text)
        self.all_unique_tokens.update(set(tokens))
        return tokens
    
    def generate_token_map(self) -> Dict[str, int]:
        self.token_map = {sym: ind for ind, sym in enumerate(sorted(self.all_unique_tokens))}
        self.config.token_map = self.token_map
        self.config.n_tokens = len(self.token_map)
        return self.token_map
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        assert self.token_map != None, "token_map not yet generated, use TextProcessor.generate_token_map() to generate it"
        tokens_ind = [self.token_map[token] for token in tokens]
        return tokens_ind


class AudioProcessor:
    """
    AudioProcessor is used to process audio and convert it into required features, and also provides other helper methods.
    """
    def __init__(self, config: AudioConfig):
        self.config = config
        self.mel_basis = get_mel_filter(fs=self.config.sampling_rate, n_fft=self.config.filter_length, n_mels=self.config.n_mels, fmin=self.config.mel_fmin, fmax=self.config.mel_fmax)
        self.inverse_mel_basis = get_inverse_mel_filter(mel_basis=self.mel_basis)

    def format_audio2wav(self, input_path: str, output_path: str) -> None:
        """convert any audio file to the proper wav file using ffmpeg"""
        # y: yes to all, i: input path, -ac 1: number of channels is 1, ar: sampling rate
        command = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(self.config.sampling_rate), output_path]
        subprocess.run(command, capture_output=True)

    def convert_wav2mel(self, input_path: str, output_path: str) -> None:
        """convert a wav file to mel spectrogram and save it"""
        fs, signal = scipy.io.wavfile.read(input_path)
        assert fs == self.config.sampling_rate, f"wav file ({input_path}) sampling rate ({fs}) does not match with config ({self.config.sampling_rate})"
        if (self.config.normalize):
            signal = normalize_signal(signal)
        spectrogram = stft(signal, n_fft=self.config.filter_length, hop_length=self.config.hop_length)
        mel_spectrogram = fft2mel(np.abs(spectrogram), self.mel_basis)
        mel_db = amplitude_to_db(mel_spectrogram, log_func=self.config.log_func, ref=self.config.ref_level_db, power=False, scale=1)
        np.save(output_path, mel_db)

    def convert_mel2wav(self, mel: Union[np.ndarray, str]) -> Tuple[int, np.ndarray]:
        """convert a mel spectrogram to waveform"""
        if isinstance(mel, str):
            mel_db = np.load(mel)
        else:
            mel_db = mel
        mel_spectrogram = db_to_amplitude(mel_db, log_func=self.config.log_func, ref=self.config.ref_level_db, power=False, scale=1)
        spectrogram_mag = mel2fft(mel_spectrogram, self.inverse_mel_basis)
        spectrogram_ang = griffin_lim(spectrogram_mag, n_fft=self.config.filter_length, hop_length=self.config.hop_length)
        spectrogram = combine_magnitude_phase(spectrogram_mag, spectrogram_ang)
        signal = istft(spectrogram, n_fft=self.config.filter_length, hop_length=self.config.hop_length)
        signal[(signal > 1) | (signal < -1)] = 0 # to remove spurious points
        signal = signal[500: -500]
        signal = normalize_signal(signal)
        signal = reduce_noise(signal, self.config.sampling_rate)
        return self.config.sampling_rate, signal


class DataPreprocessor:
    """
    Used for converting data into the required format and features.
    Tokenizing text, formatting and extracting features from audio to create dump wav and feature files and data*.csv
    """
    def __init__(
            self,
            datasets: List[BaseDataset],
            text_config: TextConfig,
            audio_config: AudioConfig,
            eval_split: Union[int, float] = 0.1,
            dump_dir: str = "dump",
        ):
        self.data: List[Dict] = []
        for dataset in datasets:
            self.data += dataset.data

        self.text_config = text_config
        self.audio_config = audio_config
        self.text_processor = TextProcessor(self.text_config)
        self.audio_processor = AudioProcessor(self.audio_config)
        self.eval_split = eval_split

        self.dump_dir = dump_dir
        self.wav_dump_dir = os.path.join(self.dump_dir, "wavs")
        self.feature_dump_dir = os.path.join(self.dump_dir, "feats")

        # loading token_map if dump dir is already there
        if self.text_config.token_map == None and os.path.isdir(self.dump_dir):
            print(f"loading token_list from {self.dump_dir} directory into text_config")
            self.text_config.token_map = load_json(os.path.join(self.dump_dir, "token_list.json"))
            self.text_config.n_tokens = len(self.text_config.token_map)

    def _filter_and_format_data(self) -> None:
        valid_data: List[Dict] = []
        total_length = 0
        valid_length = 0
        # filtering data by trimming silence and checking length
        # and then processing audio, extracting features and tokenizing text
        # and writing all of them to the dump directories
        print(f"filtering and formatting data (this step will take time)")
        for sample in tqdm.tqdm(self.data): # sample: {"text", "audio_path", "unique_id"}
            fs, wav = scipy.io.wavfile.read(sample["audio_path"])
            total_length += wav.shape[0] / fs
            if (self.audio_config.trim_silence): # trim silence and update wav_len
                left_ind, right_ind = get_non_silent_boundary(
                    signal=wav, 
                    fs=fs, 
                    silence_threshold=self.audio_config.trim_dbfs
                )
                wav_len: float = (right_ind - left_ind) / fs
            else: # else keep wav_len original
                wav_len: float = wav.shape[0] / fs
            if (self.audio_config.min_wav_duration <= wav_len and wav_len <= self.audio_config.max_wav_duration): # only if wav_len is in the desired range
                valid_length += wav_len
                wav_path = os.path.join(self.wav_dump_dir, sample["unique_id"] + ".wav")
                if (self.audio_config.trim_silence): # if trim, first write trimmed audio then process it and remove intermediate file
                    temp_wav_path = wav_path + "_"
                    scipy.io.wavfile.write(temp_wav_path, fs, wav[left_ind: right_ind])
                    self.audio_processor.format_audio2wav(temp_wav_path, wav_path)
                    os.remove(temp_wav_path)
                else: # else just process it directly
                    self.audio_processor.format(sample["audio_path"], wav_path)
                # extract features from the processed audio
                feature_path = os.path.join(self.feature_dump_dir, sample["unique_id"] + ".npy")
                self.audio_processor.convert_wav2mel(wav_path, feature_path)
                # tokenize text
                tokens = self.text_processor.tokenize(sample["text"])
                valid_data.append({
                    "unique_id": sample["unique_id"],
                    "audio_path": wav_path,
                    "feature_path": feature_path,
                    "tokens": tokens
                })

        print("trimmed and removed long and short wav files -> before: {bf_cnt} ({bf_len}), after: {af_cnt} ({af_len}), removed: {rm_cnt} ({rm_len})".format(
            bf_cnt=len(self.data),
            bf_len=sec_to_formatted_time(total_length),
            af_cnt=len(valid_data),
            af_len=sec_to_formatted_time(valid_length),
            rm_cnt=len(self.data) - len(valid_data),
            rm_len=sec_to_formatted_time(total_length - valid_length)
        ))

        print(f"tokenized text -> token vocabulary size: {len(self.text_processor.all_unique_tokens)}")
        token_map = self.text_processor.generate_token_map()
        dump_json(os.path.join(self.dump_dir, "token_list.json"), token_map)

        self.formatted_data: List[str] = []
        for sample in valid_data:
            tokens_ind = self.text_processor.tokens_to_indices(sample["tokens"])
            tokens_ind = " ".join([str(ind) for ind in tokens_ind])
            self.formatted_data.append(sample["unique_id"] + "|" + sample["audio_path"] + "|" + sample["feature_path"] + "|" + tokens_ind + "\n")

    def run(self) -> None:
        assert not os.path.isdir(self.dump_dir), f"dump directory ({self.dump_dir}) already exists"
        print(f"creating dump directories")
        os.mkdir(self.dump_dir)
        os.mkdir(self.wav_dump_dir)
        os.mkdir(self.feature_dump_dir)

        self._filter_and_format_data()

        if type(self.eval_split) == int: # count of eval samples
            eval_samples_count = self.eval_split
        else: # fraction of eval samples
            eval_samples_count = int(self.eval_split * len(self.formatted_data))
        if eval_samples_count > len(self.formatted_data) * 0.5:
            print(f"eval_samples_count ({eval_samples_count}) is more than 50% of formatted_data ({len(self.formatted_data)})")
            eval_samples_count = int(len(self.formatted_data) * 0.5)
            print(f"capping eval_samples_count to {eval_samples_count}")

        # splitting all indices into train and eval indices
        all_indices = set(range(len(self.formatted_data)))
        eval_indices = set(random.sample(all_indices, k = eval_samples_count))
        train_indices = all_indices - eval_indices
        print(f"split data ({len(all_indices)}) into train ({len(train_indices)}) and eval ({len(eval_indices)})")

        # writing total data to dump dir
        with open(os.path.join(self.dump_dir, "data.csv"), 'w') as f:
            for index in all_indices:
                f.write(self.formatted_data[index])
        # writing train data to dump dir
        with open(os.path.join(self.dump_dir, "data_train.csv"), 'w') as f:
            for index in train_indices:
                f.write(self.formatted_data[index])
        # writing eval data to dump dir
        with open(os.path.join(self.dump_dir, "data_eval.csv"), 'w') as f:
            for index in eval_indices:
                f.write(self.formatted_data[index])
