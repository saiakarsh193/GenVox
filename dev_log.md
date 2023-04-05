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
- Learn STFT and Mel Spectrogram
- Understand process in the code ([NVIDIA stft](https://github.com/NVIDIA/tacotron2/blob/master/stft.py), [NVIDIA train](https://github.com/NVIDIA/tacotron2/blob/master/train.py))


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


### 01-04-23
**{Akarsh}**
- Added own code for `trim_audio_silence` and removed `pydub` dependency.
- References for calculating dBFS: ([wiki](https://en.wikipedia.org/wiki/DBFS), [src](https://audiointerfacing.com/dbfs-in-audio/), and ChatGPT).
- Changed `None` for default value `str` type. (`typeguard` gives issues for different versions).
- Fix global pip installation issue ([src](https://stackoverflow.com/questions/44552507/conda-environment-pip-is-trying-to-install-dependencies-globally)). **Not able to fix!**
- **Fixed** global pip installation issue. The issue arises because the machine has both `pip3` and `conda` installed. So when we do `/path/to/conda/env/python3 -m pip list` with python already installed in the env, it is looking for the normal pip installation. Hence the simplest solution would be to just remove the `pip3` installation and then `conda` will only look in its own envs for pip.


### 02-04-23
**{Akarsh}**
- Changed the pipeline.
  - `DatasetProcessor` takes in `AudioConfig` and `TextConfig` and handles preprocessing using `TextProcessor` and `AudioProcessor` inside it.
- **NOTE:** Need to add code for handling multi channel wav audio files (convert to single channel before checking for silence).
- Added simple text tokenization and updated them with calculated index.
- We also need to <SOS/EOS>, <UNK> tokens too along with some standard tokenizers and cleaners.
- Use `random.sample()` as it samples without replacement. `random.choices()` samples with replacement.


### 05-04-23
**{Akarsh}**
- Look into `model.eval()` and `torch.no_grad()` use cases ([src](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2)).
- `np.float32` datatype: 1 sign bit, 23 bits mantissa, 8 bits exponent (single decimal precision float) ([src](https://stackoverflow.com/questions/16963956/difference-between-python-float-and-numpy-float32)).
- `g2p`: we need to add oov (for char level we just ignore those characters).
- Trainable Fourier kernels ([src1](https://github.com/KinWaiCheuk/nnAudio), [src2](https://github.com/pseeth/torch-stft))