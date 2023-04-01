# GenVox
Pipeline for building TTS models using Tacotron2.

## Usage
```bash
conda create --prefix ./ttsenv
conda activate ./ttsenv
conda install pip
./ttsenv/bin/python3 -m pip install -r requirements.txt

python3 run.py
```

## Development
Started on (26-03-23)  
Read [devlog](dev_log.md) for more details.  

### Phase 1 (1 week)
- [x] Getting Dataset (2 days)
  - [x] Download
  - [x] Create using YT_Transcript
  - [ ] Create using Aeneas/WhisperX
- [x] Preparing Dataset (2 days)
  - [x] Format of text and path
  - [x] Format to wav and fs
  - [x] Remove long/short
- [ ] Preprocessing (3 days)
  - [ ] Tokens/Vocab generation
  - [ ] Features generation

### Phase 2
- [ ] Torch Model
- [ ] Trainer
- [ ] Distributed

### Explore
- [ ] argparse
- [ ] logging
- [ ] pip standards (with docstrings)
- [ ] aeneas
- [x] youtubedl
- [ ] readme badges
- [x] config (yaml?)
- [ ] pip package (export)