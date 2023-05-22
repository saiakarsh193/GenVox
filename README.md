# GenVox
Pipeline for building TTS

Supported Text2Mel:
- Tacotron2

Supported Vocoder:
- MelGAN

## Usage
```bash
# for CPU torch installation
conda create --prefix ./ttsenv pytorch -c pytorch

# for GPU torch installation
conda create --prefix ./ttsenv pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

# activate environment
conda activate ./ttsenv

# for other requirements
./ttsenv/bin/python3 -m pip install -r requirements.txt

python3 run.py
```

## Development
Read [devlog](dev_log.md) for more details.  
Inspired by [NVIDIA](https://github.com/NVIDIA/tacotron2), [Coqui-AI](https://github.com/coqui-ai/TTS), [ESPNET](https://github.com/espnet/espnet).
