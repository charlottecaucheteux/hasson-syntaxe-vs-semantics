from pathlib import Path

import numpy as np
import torch

from src import paths
from src.get_features import load_precomputed_features

EXP_NAME = "0225-gpt2-errors"
EXP_NAME = "0308-gpt2-errors"

# NEW with valid sentences

keys = [
    "phon-syn-sem-1dim",
]

names = {}
for key in keys:
    to_catch = key.replace("-1dim", "")
    names[key] = to_catch.split("-")

FILES = {}
for name, files in names.items():
    FILES[name] = [
        str(paths.embeddings / EXP_NAME / "%s" / f"{file_name}.pth")
        for file_name in files
    ]

print(FILES)


def select_features(files, audio_task):
    # Extract features from stimulus
    task_files = [Path(str(f) % str(audio_task)) for f in files]
    assert np.all(
        [f.is_file() for f in task_files]
    ), f"!!!!!! NOT EXIXTS : {task_files}"

    feats = load_precomputed_features(audio_task, task_files, idx=None)

    # lens = [f.shape[1] for f in feats]
    feats = [f[:, -1] for f in feats]

    idx = torch.arange(len(feats))
    # idx = torch.LongTensor(idx)
    feats = np.stack(feats, axis=-1)
    feats = torch.FloatTensor(feats)
    return feats, idx


if __name__ == "__main__":

    save_dir = paths.embeddings / EXP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build params
    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))]
    params = []
    for task in tasks:
        for name, files in FILES.items():
            save_file = save_dir / task / f"{name}.pth"
            # if not save_file.is_file():
            print(save_file)
            feat, idx = select_features(files, task)
            idx_file = save_dir / task / f"{name}.idx.pth"
            save_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(feat, save_file)
            torch.save(idx, idx_file)
