from pathlib import Path

import numpy as np
import torch

from src import paths
from src.get_features import load_precomputed_features

EXP_NAME = "0201_wiki_valid"

names = {}
for layer in [0, 9]:
    names[f"semantic-gpt2-{layer}.equiv-random-mean-10"] = [
        f"sum-gpt2-{layer}.equiv-random-mean-10",
        f"sum-gpt2-{layer}",
    ]

FILES = {}
for name, files in names.items():
    FILES[name] = [
        str(paths.embeddings / EXP_NAME / "%s" / f"{file_name}.pth")
        for file_name in files
    ]

print(FILES)


def substract_features(files, audio_task):
    # Extract features from stimulus
    assert len(files) == 2
    task_files = [Path(str(f) % str(audio_task)) for f in files]
    assert np.all(
        [f.is_file() for f in task_files]
    ), f"!!!!!! NOT EXIXTS : {task_files}"

    feats = load_precomputed_features(audio_task, task_files, idx=None)
    feats = feats[1] - feats[0]
    feats = torch.FloatTensor(feats)
    return feats


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
            feat = substract_features(files, task)
            save_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(feat, save_file)
