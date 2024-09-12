import os
import subprocess as sp
from pathlib import Path
import sys


def fatal(msg):
    print('Fatal:', msg, file=sys.stderr)
    sys.exit(1)


DATASET_ROOT = Path("./data")
DATASET_URL = Path("TO_FILL")

EGS_FILE = DATASET_ROOT / 'tracks.jsonl'

if 'DATASET_URL' in os.environ:
    DATASET_URL = Path(os.environ['DATASET_URL'])


def prepare_dataset():
    DATASET_ROOT.mkdir(exist_ok=True, parents=True)
    out_file = DATASET_ROOT / 'dataset.zip'
    if not out_file.exists():
        tmp_file = Path(str(out_file) + '.tmp')
        sp.call(['curl', DATASET_URL, '-o', tmp_file], check=True)
        tmp_file.rename(out_file)

    done_file = DATASET_ROOT / 'extracted'
    if not done_file.exists():
        sp.call(['unzip', out_file], cwd=DATASET_ROOT, check=True)
        done_file.touch()

    if not EGS_FILE.exists():
        sp.call(['python', '-m', 'audiocraft.data.audio_dataset', DATASET_ROOT, EGS_FILE], check=True)


if __name__ == '__main__':
    prepare_dataset()
