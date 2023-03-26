# Development Log
## Developers
- **{Akarsh}**
- **{Rakesh}**

## Log Dates
- [26-03-2020](#26-03-23)

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