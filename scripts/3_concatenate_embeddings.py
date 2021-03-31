from pathlib import Path

import numpy as np
import torch

from src import paths
from src.get_features import load_precomputed_features

EXP_NAME = "0206_wordembed"
# EXP_NAME = "0321_hugg_models"
# EXP_NAME = "0323_hugg_models_bidir"

names = {}
keys = ["sum-gpt2-%s.equiv-random-mean-10", "sum-gpt2-%s"]
for key in keys:
    for layer in range(13):
        k = key % layer
        print(k)
        names[f"phone_{k}"] = ["n_words", "n_phones", "phones", k]

for key in ["equal_len_sentence", "shuffle_in_sentence", "shuffle_in_task"]:
    names[f"phone_sum-gpt2-9.{key}"] = [
        "n_words",
        "n_phones",
        "phones",
        f"sum-gpt2-9.{key}",
    ]


models = [
    # "sum-albert-base-v1",
    "sum-bert-base-uncased",
    # "sum-roberta-base",
    # "sum-xlnet-base-cased",
]
models = [
    ("bert-base-uncased", 8),  # 12
    ("xlnet-base-cased", 8),  # 12
    ("roberta-base", 8),  # 12
    # "longformer-base-4096",  # 12
    # "squeezebert-mnli",
    # "layoutlm-base-uncased",
    ("albert-base-v1", 8),  # 12
    ("distilgpt2", 4),  # 6 (layer 4)
    # ("transfo-xl-wt103", 12), # 18 (-> layer 12?)
    # ("distilbert-base-uncased", 4),  6 (layer 4)
]

models = [
    ("transfo-xl-wt103", 12),  # 18 (-> layer 12?)
    ("distilbert-base-uncased", 4),  # 6 (layer 4)
]

names = {"3_phone_features": ["n_words", "n_phones", "phones"]}
for model, layer in models:
    for ext in ["", ".equiv-random-mean-10"]:
        key = "sum-" + model + "-" + str(layer) + ext
        names[f"phone_{key}"] = [
            "n_words",
            "n_phones",
            "phones",
            key,
        ]

names = {}
for key in ["6-delayed-sum-gpt2-0", "6-delayed-sum-gpt2-0.equiv-random-mean-10"]:
    names[f"phone_{key}"] = [
        "n_words",
        "n_phones",
        "phones",
        key,
    ]

FILES = {}
for name, files in names.items():
    FILES[name] = [
        str(paths.embeddings / EXP_NAME / "%s" / f"{file_name}.pth")
        for file_name in files
    ]

print(FILES)


def merge_features(files, audio_task):
    # Extract features from stimulus
    task_files = [Path(str(f) % str(audio_task)) for f in files]
    assert np.all(
        [f.is_file() for f in task_files]
    ), f"!!!!!! NOT EXIXTS : {task_files}"

    feats = load_precomputed_features(audio_task, task_files, idx=None)
    lens = [f.shape[1] for f in feats]
    idx = np.repeat(np.arange(len(lens)), lens)
    idx = torch.LongTensor(idx)
    feats = np.concatenate(feats, axis=-1)
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
            feat, idx = merge_features(files, task)
            idx_file = save_dir / task / f"{name}.idx.pth"
            save_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(feat, save_file)
            torch.save(idx, idx_file)
