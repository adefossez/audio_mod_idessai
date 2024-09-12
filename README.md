# IDESSAI 2024 - Auto-regressive modeling of discrete audio tokens

This repository provides the code for my class at IDESSAI 2024 about auto-regressive modeling of discrete
audio tokens. We use [Audiocraft](https://github.com/facebookresearch/audiocraft) to fine tune a pre-trained
[MusicGen](https://arxiv.org/abs/2306.05284) model on a small dataset of tracks from a given style.

## Requirements

First clone this repository and cd the root folder:
```bash
git clone https://github.com/adefossez/audio_mod_idessai.git
cd audio_mod_idessai
```

Make sure to have an environment with `ffmpeg` installed, the easiest is with
conda/mamba: `conda install -c conda-forge ffmpeg`.

Then we install `audiocraft` with slightly
different requirements to allow more recent versions of PyTorch (especially on Colab).
Note that I had some issues with python3.10 getting a bus error, so maybe try to use python3.12.

```bash
# If you need a specific version of cuda, first install it along with torchaudio, for instance
# xformers can be a bit tricky to get when pytorch releases a new version, so we pin 2.4.0.
pip install torch==2.4.0 torchaudio==2.4.0 xformers
pip install -r requirements.txt

# If you want to run locally the notebook, and maybe have some VIM binding ;)
pip install jupyter # jupyterlab-vim
```


## Setup

Edit `audio_mod_idessai/config.py` with the proper URL.

### Download the dataset

```bash
python -m audio_mod_idessai.config
```

### Launch notebook

```bash
jupyter notebook
```
