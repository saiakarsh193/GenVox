# Development Log
## Developers
- **{Akarsh}**
- **{Rakesh}**


## Log Dates
<!-- no toc -->
- [26-03-23](#26-03-23)
- [27-03-23](#27-03-23)
- [28-03-23](#28-03-23)
- [29-03-23](#29-03-23)
- [30-03-23](#30-03-23)
- [01-04-23](#01-04-23)
- [02-04-23](#02-04-23)
- [05-04-23](#05-04-23)
- [08-04-23](#08-04-23)
- [09-04-23](#09-04-23)
- [11-04-23](#11-04-23)
- [12-04-23](#12-04-23)
- [13-04-23](#13-04-23)


### 26-03-23
**{Akarsh}**
- [HindiTTS](https://github.com/saiakarsh193/HindiTTS) for dataset creating, annotating and cleaning.
- [Aeneas_Extended](https://github.com/saiakarsh193/Extended-Forced-Aligner/blob/master/extend_aligner.py) for alignment using aeneas.
- [ESPNET](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/tts1#5-tts-statistics-collection), [Coqui](https://github.com/coqui-ai/TTS) for format, pipeline and code reference.
- [ReadtheDocs](https://readthedocs.org) for documentation preparation.
- First understand logging, argparse and get a defined `<input, output, log>` format for any code.
- Follow `pip` standards, docstrings, and tabspaces for coding.
- Define config file format (separate class -> `yaml` wrapper?):
  - project details
    - name
    - tag
    - language
  - dataset details (for extracting text: `local/data.sh`)
  - preparing details
    - text:
      - token type
      - cleaner
      - g2p (if any)
    - audio:
      - sampling_rate(fs)
      - trim_silence? (trim_db)
      - signal_norm
  - preprocessing details (_TODO_)
  - model details
    - name
    - params
  - trainer details (_TODO_: logging, batch, etc)
  - output details (path for `output_dir/`)
- Learn git branching.
- Decide if bash is used or not: I prefer using it because it is very quick, inclined to write both bash and python variants.
- readme md links:
  - [Basic formatting](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
  - [Collapsible section](https://gist.github.com/pierrejoubert73/902cc94d79424356a8d20be2b382e1ab)
- files:
  - `run.py` (prepare_data, preprocess_data, train_tts)
  - `prepare_data.py`
  - `preprocess_data.py`
  - `make_dataset.py` (dataset creating using raw audio, transcripts or youtube links)
  - `train_tts.py`
> Idealogy: run code with config path
- Added base_config, dataset_config (using [this](https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/audio/processor.py) and [this](https://github.com/coqui-ai/TTS/blob/d309f50e53aaa4fa6fc540f98615f1963d61447f/TTS/config/shared_configs.py#L9) for value and docstring reference).
- Added dataset_processor for making i2t and i2w (created LJSpeech_{small, sample} for experimentation and testing).
- Writing entire runner script in `run.py` and will later divide it into individual components.


### 27-03-23
**{Akarsh}**
- Using coqui for coding format, espnet for data format and [NVIDIA](https://github.com/NVIDIA/tacotron2/) for model code.
- We are using `nn.ModuleList` instead of `list` because `model.parameters()` wont have the necessary parameters as it cannot read a regular python list ([source](https://discuss.pytorch.org/t/the-difference-in-usage-between-nn-modulelist-and-python-list/7744)).
- squeeze and unsqueeze in torch tensor ([source](https://www.geeksforgeeks.org/how-to-squeeze-and-unsqueeze-a-tensor-in-pytorch/)).
- Assumption: single speaker, single language, single GPU (not distributed).
- For distributed read about: workers, jobs, nodes, tasks.
- collate: "collect and combine (texts, information, or data)" ([source](https://plainenglish.io/blog/understanding-collate-fn-in-pytorch-f9d1742647d3), [CREPE-ref](https://github.com/KawshikManikantan/CREPE/blob/main/Code/lstm_base.ipynb))
- MUST-C dataset ([source](https://ict.fbk.eu/must-c-releases/))

**{Rakesh}**
- Using youtube-dl to download the audio (wav) and transcript(.vtt file) of the audio file. (used `pip install git+https://github.com/ytdl-org/youtube-dl.git@master#egg=youtube_dl` as there was a change in youtube metadata)
- Converted the data in vtt file into csv file formatting similar to the LJSpeech dataset format.
- The audio is split into smaller segments by using the timestamps in vtt file and output wav files into a directory.


### 28-03-23
**{Akarsh}**
- `self.training` in `nn.Module` is explained here ([source](https://github.com/google/objax/issues/29)).
- Gate Prediction is basically stop token prediction. It is used to stop inference in decoding stage ([source](https://github.com/NVIDIA/tacotron2/blob/master/model.py#L378)).
- To prevent `Markdown All in One` from auto creating `Table of Contents`, add a `<!-- no toc -->` comment above the list.
- `subprocess` in python to handle `bash` commands ([source](https://docs.python.org/3/library/subprocess.html)).

**{Rakesh}**
- Input is taken to format the naming of the wav files to particular id and the padding the file number to match number of digits using rjust
- Same id format for the id in csv file
- Trim silence of left and right ends of the audio segment by using a silence threshold ([ref](https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub))


### 29-03-23
**{Akarsh}**

**{Rakesh}**
- Learn about STFT and Mel Spectrogram.
- Understand process in the code ([NVIDIA stft](https://github.com/NVIDIA/tacotron2/blob/master/stft.py), [NVIDIA train](https://github.com/NVIDIA/tacotron2/blob/master/train.py)).


### 30-03-23
**{Akarsh}**
- Added `DownloadConfig` and `DownloadProcessor`.
- Added extraction support and changed pipeline -> `i2at, i2aw` contain all values before processing and `i2t, i2w` contain only the values that are valid.
- Added wav dump creation with length filtering and sampling rate modification (and bitrate) in `dump/wavs/` directory.
- **NOTE:** Need to add `trimAudio` for the wavs before checking the length before filtering.
- `i2w` format -> `<utt_id> <wav_path> <wav_shape>`.
- You can now download `wav` and `vtt` files from Youtube (`yt_dlp`) and use the alignment to segment the audio file into `wavs/` directory and also choose a `speaker_id` for `utt_id`. Then we can prepare `transcript.txt` using standard delimiter `|` and finally we have a dataset.
- Added `secToFormattedTime` in `utils` for printing time from seconds to a standard format.
- Check `trimAudio` in general and format `create_dataset`.

**{Rakesh}**
- Added `create_dataset` file to download wav and transcript of the given youtube link and divide it into multiple segments and their respective text data.
- Switched from `youtube_dl` to `yt-dlp` which is a forked version of youtube_dl as there were issues with youtube_dl. ([yt-dlp](https://github.com/yt-dlp/yt-dlp)).
- Added a function to check whether the given link is youtube link or not.


### 01-04-23
**{Akarsh}**
- Added own code for `trim_audio_silence` and removed `pydub` dependency.
- References for calculating dBFS: ([wiki](https://en.wikipedia.org/wiki/DBFS), [src](https://audiointerfacing.com/dbfs-in-audio/), and ChatGPT).
- Changed `None` for default value `str` type. (`typeguard` gives issues for different versions).
- Fix global pip installation issue ([src](https://stackoverflow.com/questions/44552507/conda-environment-pip-is-trying-to-install-dependencies-globally)). **Not able to fix!**
- **Fixed** global pip installation issue. The issue arises because the machine has both `pip3` and `conda` installed. So when we do `/path/to/conda/env/python3 -m pip list` with python already installed in the env, it is looking for the normal pip installation. Hence the simplest solution would be to just remove the `pip3` installation and then `conda` will only look in its own envs for pip.

**{Rakesh}**
- Changed the `vtt_to_csv` file to remove the dependency on reg ex. Also simplified the function to remove redundancy.
- Removed `parse_vtt` function and added the same functionality into `vtt_to_csv` function.
- Added verbose and quiet options for `Download_YT` function.
- Using `os.path.join()` for all directory paths.


### 02-04-23
**{Akarsh}**
- Changed the pipeline.
  - `DatasetProcessor` takes in `AudioConfig` and `TextConfig` and handles preprocessing using `TextProcessor` and `AudioProcessor` inside it.
- **NOTE:** Need to add code for handling multi channel wav audio files (convert to single channel before checking for silence).
- Added simple text tokenization and updated them with calculated index.
- We also need to <SOS/EOS>, <UNK> tokens too along with some standard tokenizers and cleaners.
- Use `random.sample()` as it samples without replacement. `random.choices()` samples with replacement.

**{Rakesh}**
- Check how the text is processed (tokenizers, cleaners) in ([NVIDIA](https://github.com/NVIDIA/tacotron2/tree/master/text)) and ([espnet](https://github.com/espnet/espnet/tree/master/espnet2/text)).
- Learn how the text and mel are processed in `TextMelLoader` and `TextMelCollate`.


### 05-04-23
**{Akarsh}**
- Look into `model.eval()` and `torch.no_grad()` use cases ([src](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2)).
- `np.float32` datatype: 1 sign bit, 23 bits mantissa, 8 bits exponent (single decimal precision float) ([src](https://stackoverflow.com/questions/16963956/difference-between-python-float-and-numpy-float32)).
- `g2p`: we need to add oov (for char level we just ignore those characters).
- Trainable Fourier kernels ([src1](https://github.com/KinWaiCheuk/nnAudio), [src2](https://github.com/pseeth/torch-stft)).
- STFT simple code idea in python ([src](https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/)).
  - We pad zeroes the length of the frame, in order to get first half of FFT the same length of as that of the frame.
  - `np.fft.fft` has even (Hermitian) symmetry ([src](https://stackoverflow.com/questions/70758915/is-a-numpy-fft-on-real-values-actually-hermitian)).
- `ffmpeg` cannot edit existing files in-place. we need to make duplicate file.
- Replaced `pcm_u8` codec to default `pcm_s16le` codec in `ffmpeg`. In unsigned type, the mean is positive (signal moved up into positive axis). Hence mean wont be zero and therefore not ideal.


### 08-04-23
**{Akarsh}**
- `tensor.half()` or `model.half()` will convert all model weights to half precision. This is done to speed up calculations ([src](https://discuss.pytorch.org/t/what-does-pytorch-do-when-calling-tensor-half/58597)).
- why computed frame count and `librosa.stft` frame count is not matching. Because of center padding. Read ([src1](https://github.com/librosa/librosa/issues/530), [src2](https://groups.google.com/g/librosa/c/b5dShgDAkWo?pli=1)). With `center=False` it is the normal calculation, that is
  $$ \textrm{total length} = \textrm{window length} + (\lambda - 1) * \textrm{hop length}$$
  this also assumes that the signal fits perfectly (if extra is there, then we dont consider it).  
  Also if `center=False` in `librosa.stft()` then `pad_mode` is ignored. If `center=True`, then we use the default value of `pad_mode="constant"` to pad zeros to the input signal on **both** sides for framing.
- Nyquist theorem and calculating signal having a particular frequency:
  ```
  nyquist theorem states:
  sampling frequency >= 2 * max_frequency to be preserved
  #points per sec >= 2 * #waves per sec
  ---
  if coeff increases wavelength decreases (inversely related)
  coeff -> #values for one wave (continuous)
  1 -> 2 * pi values
  x -> 1 / f values
  1 * 2 * pi = x * 1 / f
  => x = (2 * pi * f)
  use sin(x * points) for f frequency signal
  ```
- Important concepts for STFT and mel spectrogram (the way `librosa` does `librosa.amplitude_to_db()`, `librosa.stft()`):
  - signal -> [filter_length, hop_length, window_length] -> frames
  - frames (real) -> [stacking fft(frame) => stft] -> spectrogram (complex)
  - spectrogram (complex) -> [np.abs] -> amplitude_spectrogram -> [np.power, usually ^2] -> power_spectrogram
  - power_spectrogram -> [10 * log10(max(amin, sig)) - 10 * log10(max(amin, ref))] -> dB_scale_spectrogram
  - power_spectrogram -> [np.matmul(mel_transform_filter, spectrogram)] -> mel_spectrogram
  - `np.abs()` of spectrogram gives magnitude while `np.angle()` gives **phase**
- Reading about mel and its use ([src](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)).


### 09-04-23
**{Akarsh}**
- `librosa.stft()` uses `np.fft.rfft()`, hence we get `1 + (n_fft // 2)` values as the output ([src](https://stackoverflow.com/questions/52387673/what-is-the-difference-between-numpy-fft-fft-and-numpy-fft-rfft)).  
  `np.fft.fft()` for real input gives Hermitian-symmetric output, where the negative frequencies are the complex conjugates of positive frequencies and are hence redundant.
- Completely implemented `stft()` and `istft()` from scratch using `numpy`. Heavily used `librosa` docs and other resources, but greatly simplified the code by making to quite task specific.
- Added above explored code along with other audio helper functions in `audio.py`.
- **Removed** `librosa` dependency.
- Added `AudioProcessor.convert2mel()` that takes in a wav_path and extracts features (mel spectrogram) from it and saves it as a `.npy` file in `dump/feats` directory.  
  `.npy` file is `numpy` format for saving arrays as data ([src](https://numpy.org/doc/stable/reference/generated/numpy.save.html)).


### 11-04-23
**{Akarsh}**
- Explore the idea of calculating features on the go. We can reduce load and increase parallelization during loading. This is handled natively by `torch.utils.data.DataLoader` which can be accessed by `num_workers` value. The only upside is to **save memory**.
- **Assumption**: single speaker, single language on single GPU
- Added data splitting into training and validation inside `DatasetProcessor`.
- Check `pin_memory=False` and `drop_last=True` along with `shuffle=True/False` in `torch.utils.data.DataLoader`.
- Add collate function for DataLoader.


### 12-04-23
**{Akarsh}**
- `torch.nn` vs `torch.nn.functional` ([src](https://discuss.pytorch.org/t/what-is-the-difference-between-torch-nn-and-torch-nn-functional/33597)).
- `torch.utils.data.DataLoader` attributes: ([src](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader))
  - pin_memory (bool, optional) – If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below. (default: False)
  - drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
- Added `TextMelCollate()` for `torch.utils.data.DataLoader.collate_fn` in `Trainer`.


### 13-04-23
**{Akarsh}**
- Added `CheckpointManager()` in `Trainer` that will handle file saving for top n checkpoints based on the loss value.
- Added `WandbLogger()` in `Trainer` that will handle `wandb` logging and syncing.
- Added `use_g2p` and updated `TextProcessor` pipeline.
- Added `g2p_en` by Park Kyu Byong ([src](https://github.com/Kyubyong/g2p)).
- *NOTE*:
  - Consider on the fly loading of tokens and mel feats generation (along with resampling and trimming). This will save space, but we wont have control over size of audio.


### 18-04-23
**{Akarsh}**
- Omitting param groups of optimizer update before every iteration. Later on add and experiment. ([src](https://github.com/NVIDIA/tacotron2/blob/185cd24e046cc1304b4f8e564734d2498c6e2e6f/train.py#L210)).
- `optimizer.zero_grad()` vs `model.zero_grad()` ([src](https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/3)). They are the same ([src2](https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)).
- Decide variable to log (gpu, etc) and config values.
- Add timers for training loop and estimators.


### 19-04-23
**{Akarsh}**
- Added evaluation in training loop with timers and estimators.
- Integrated wandb that supports images. Updated `WandbLogger` to abstract direct `wandb` API.
  - `commit=True` in `wandb.log()` also increments the iteration value ([src](https://docs.wandb.ai/ref/python/log)).
- Added `inference.py` that contains the `TTSModel()` class for handling the inference.
- Added time formatting in `utils.py` and log printing.
