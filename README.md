# GenVox
Pipeline for building TTS models using Tacotron2.

## Usage
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development
Started on (26-03-23)  
Read [devlog](dev_log.md) for more details.  

### Phase 1 (1 week)
- [ ] Getting Dataset (2 days)
  - [ ] Download
  - [ ] Create using Aeneas/WhisperX/YT_Transcript
- [ ] Preparing Dataset (2 days)
  - [ ] Format of text and path
  - [ ] Format to wav and fs
  - [ ] Remove long/short
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
- [ ] youtubedl
- [ ] readme badges
- [x] config (yaml?)
- [ ] pip package (export)