from pathlib import Path

import numpy as np
import torch

from src import paths
from src.get_features import load_precomputed_features

EXP_NAME = "0206_wordembed"

keys = [
    "sum-gpt2-9",
    "sum-gpt2-0",
    "sum-gpt2-9.equiv-random-mean-10",
    "sum-gpt2-0.equiv-random-mean-10",
]
FILES = {}
for key in keys:
    label = f"6-delayed-{key}"
    FILES[label] = key

print(FILES)


def delay_features(input_files, audio_task, save_files):
    # Extract features from stimulus
    feats = load_precomputed_features(audio_task, input_files, idx=None)
    assert len(feats) == len(save_files)
    for feat, save_file in zip(feats, save_files):
        feat = torch.cat(
            [torch.roll(feat, shifts=shift, dims=1) for shift in range(7)], axis=-1
        )  # add the words BEFORE the current word
        feat = torch.FloatTensor(feat)
        if save_file:
            save_file.parent.mkdir(exist_ok=True, parents=True)
            print(f"saving to : {save_file}")
            torch.save(feat, save_file)

    # return feats


if __name__ == "__main__":

    save_dir = paths.embeddings / EXP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build params
    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))]
    params = []
    for task in tasks:
        print(task)
        input_files = [save_dir / task / f"{in_name}.pth" for in_name in FILES.values()]
        save_files = [save_dir / task / f"{name}.pth" for name in FILES.keys()]
        delay_features(input_files, task, save_files)
