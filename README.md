# GenVox
Pipeline for building TTS

Supported Text2Mel:
- Tacotron2 - [paper](https://arxiv.org/pdf/1712.05884.pdf)

Supported Vocoder:
- MelGAN - [paper](https://arxiv.org/pdf/1910.06711.pdf)

## Usage

### Set up the repo (dev branch)
```bash
# recommended method to clone repo and track branch
git clone https://github.com/saiakarsh193/GenVox
# will automatically track to upstream remote dev
git checkout dev

# you can also clone using these methods
# 1. to set dev as the local branch but still track other branches.
git clone -b dev https://github.com/saiakarsh193/GenVox
# 2. to only clone dev branch
git clone -b dev --single-branch https://github.com/saiakarsh193/GenVox

# to check everything is correctly tracked
git remote show origin

# enter the repo
cd GenVox
```

### Set up the env
```bash
# NOTE: if running on ADA
# go to interactive shell (to prevent out of memory issue)
sinteractive -c 8 -g 1 .
# and then load cuda, ffmpeg modules
module load u18/cuda/10.2
module load u18/ffmpeg/5.0.1


# create and activate environment
conda create --prefix ./ttsenv
conda activate ./ttsenv

# install python (using 3.8 as it is tested and works)
conda install python=3.8
# install dependencies
./ttsenv/bin/python3 -m pip install -r requirements.txt
```

### Set up the data and run the code
```bash
# create data directory
mkdir data
# move/download the dataset into data directory and then edit run.py accordingly

# once everything is all setup, you can run the code
python3 run.py
```

## Development
Read [devlog](dev_log.md) for more details.  
Inspired by [NVIDIA](https://github.com/NVIDIA/tacotron2), [Coqui-AI](https://github.com/coqui-ai/TTS), [ESPNET](https://github.com/espnet/espnet).
