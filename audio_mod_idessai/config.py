# MIT License, copyright 2024 Alexandre Défossez.
"""Config and setting up of the dataset."""
import os
import subprocess as sp
from pathlib import Path
import sys


def fatal(msg):
    print('Fatal:', msg, file=sys.stderr)
    sys.exit(1)


DATASET_ROOT = Path("./data")
DATASET_URL = "TO_FILL"

EGS_FILE = DATASET_ROOT / 'tracks.jsonl'

if 'DATASET_URL' in os.environ:
    DATASET_URL = os.environ['DATASET_URL']


def prepare_dataset():
    DATASET_ROOT.mkdir(exist_ok=True, parents=True)
    out_file = DATASET_ROOT / 'dataset.zip'
    if not out_file.exists():
        tmp_file = Path(str(out_file) + '.tmp')
        sp.run(['curl', '-L', DATASET_URL, '-o', tmp_file], check=True)
        size_mb = tmp_file.stat().st_size / 1e6
        if size_mb < 1:
            # Less than 1 MB of data, we probably  mistyped the URL.
            fatal(f"Dataset from {DATASET_URL} is only {size_mb:.3f}MB, "
                  "please double check URL.")
        tmp_file.rename(out_file)

    done_file = DATASET_ROOT / 'extracted'
    if not done_file.exists():
        sp.run(['unzip', out_file.relative_to(DATASET_ROOT)],
               cwd=DATASET_ROOT, check=True)
        done_file.touch()

    if not EGS_FILE.exists():
        sp.run(['python', '-m', 'audiocraft.data.audio_dataset',
                DATASET_ROOT, EGS_FILE], check=True)


if __name__ == '__main__':
    prepare_dataset()
