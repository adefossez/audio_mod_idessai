# IDESSAI 2024 - Auto-regressive modeling of discrete audio tokens

This repository provides the code for my class at IDESSAI 2024 about auto-regressive modeling of discrete
audio tokens. We use [Audiocraft](https://github.com/facebookresearch/audiocraft) to fine tune a pre-trained
[MusicGen](https://arxiv.org/abs/2306.05284) model on a small dataset of tracks from a given style.

## Requirements

Make sure to have an environment with `ffmpeg` installed, the easiest is with
conda/mamba: `conda install -c conda-forge ffmpeg`. Then we install `audiocraft` with slightly
different requirements to allow more recent versions of PyTorch:

```
# If you need a specific version of torch and cuda, first install it along with torchaudio, for instance
pip install torch torchaudio
pip install -r requirements.txt
pip install --no-deps audiocraft
```


## Download the dataset


