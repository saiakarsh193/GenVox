<div align="center">
<!--<a href="https://github.com/saiakarsh193/GenVox"><img src="https://i.ibb.co/y5jqFFJ/genvox-logo1.png" alt="genvox-logo1" width="300"/></a>-->
<a href="https://github.com/saiakarsh193/GenVox"><img src="https://i.ibb.co/nRyJBGk/genvox-logo2.png" alt="genvox-logo2" width="250"/></a>
</div>

# GenVox
GenVox is an end-to-end Python-based neural speech synthesis toolkit that provides both Text-to-Speech (TTS) and Vocoder models. GenVox uses [Pytorch](http://pytorch.org/) for its neural backend. GenVox coding style is heavily inspired by [Coqui-AI](https://github.com/coqui-ai/TTS), and the data flow is inspired by [ESPNET](https://github.com/espnet/espnet). The model architecture codes were taken from various places but heavily modified for better readability, quality and optimization. Read [devlog](dev_log.md) for more details regarding the development.  

🥣 Recipes for training your model from scratch with [WandB](https://wandb.ai/) support for logging.  
⚒ Tools for creating and processing datasets, and data analysis.  
⏱ Codes for optimized audio and text processing, logging, and pipelines for server-specific optimization.  
🔥 Designed to add your own custom architectures easily.  

Supported TTS (Text2Mel):
- Tacotron2 - [paper](https://arxiv.org/pdf/1712.05884.pdf) - [original repo](https://github.com/NVIDIA/tacotron2)

## 📖 Installation
Clone the [repo](https://github.com/saiakarsh193/GenVox) using any one of the methods
```bash
# recommended method
git clone https://github.com/saiakarsh193/GenVox
git checkout dev

git clone -b dev https://github.com/saiakarsh193/GenVox

git clone -b dev --single-branch https://github.com/saiakarsh193/GenVox

git remote show origin # to check everything is correctly tracked
```

**NOTE**: If running on ADA cluster
```bash
sinteractive -c 8 -g 1 # to prevent out of memory issue
module load u18/cuda/10.2
module load u18/ffmpeg/5.0.1
```

Setting up the environment
```bash
# create environment with python 3.8 (as it is tested and working) and activate it
conda create --name genvox python=3.8
conda activate genvox
# or
conda create --prefix ./genvox python=3.8
conda activate ./genvox

pip install -r requirements.txt # to install the dependencies
```

## 🚀 Training your model
```bash
# to use wandb for logging, you need to login (only once)
wandb login # then type your API key (you can find your API key in your browser at https://wandb.ai/authorize)

# after all the hard work, you can finally run the code
python3 run.py
```

## 📢 Generate speech using a pretrained model
Check [demo.py](demo.py) for more details
```python3
import scipy.io
from core.synthesizer import Synthesizer
from models.tts.tacotron2 import Tacotron2

syn = Synthesizer(
    tts_model_class=Tacotron2,
    tts_config_path=<path/to/config>,
    tts_checkpoint_path=<path/to/checkpoint>
)
outputs = syn.tts(text="Hello world! This is a test sentence.")
scipy.io.wavfile.write('pred_sig.wav', outputs["sampling_rate"], outputs["waveform"])
```
