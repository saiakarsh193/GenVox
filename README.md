# GenVox
Pipeline for building TTS

Supported Text2Mel:
- Tacotron2 - [paper](https://arxiv.org/pdf/1712.05884.pdf)

Supported Vocoder:
- MelGAN - [paper](https://arxiv.org/pdf/1910.06711.pdf)

## Usage
```bash
# create and activate environment
conda create --prefix ./ttsenv
conda activate ./ttsenv

# install python (using 3.8 as it is tested and works)
conda install python=3.8
# install dependencies
./ttsenv/bin/python3 -m pip install -r requirements.txt

python3 run.py
```

## Development
Read [devlog](dev_log.md) for more details.  
Inspired by [NVIDIA](https://github.com/NVIDIA/tacotron2), [Coqui-AI](https://github.com/coqui-ai/TTS), [ESPNET](https://github.com/espnet/espnet).
