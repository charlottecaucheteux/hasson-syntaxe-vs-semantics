from pathlib import Path

import numpy as np
import torch

from src import paths
from src.get_features import load_precomputed_features

# EXP_NAME = "0225-gpt2-errors"
EXP_NAME = "0311-gpt2-errors"
EXP_NAME = "0316-gpt2-errors"

# NEW with valid sentences

keys = ["freq", "freq-lex", "freq-lex-sem", "freq-lex-syn", "freq-lex-sem-syn"]
keys += [
    "freq-lex",
    "freq-syn",
    "freq-phon",
    "freq-sem",
    "freq-phon-lex-sem-syn",
    "freq-phon-sem-syn",
]
keys = [
    "freq-phon",
    "freq-phon-lex",
    # "freq-phon-lex-syn",
    # "freq-phon-lex-sem",
    "freq-phon",
    "freq-phon-lex",
    "freq-phon-lex-syn",
    "freq-phon-lex-sem",
    "freq-phon-lex-syn-sem",
]

keys = [
    "phon",
    "phon-syn",
    "phon-sem",
    "phon-syn-sem",
    "phon-syn-sem-lex",
    "syn-sem-lex",
    "syn-sem",
    "syn-lex",
    "phon-lex",
    "phon-syn-lex",
]

keys = [
    "syn-logsyn",
    "sem-logsem",
    "wordrate-syn-logsyn",
    "syn-logsyn-sem-logsem",
    "wordrate-syn-logsyn-sem-logsem",
]

keys = ["wordrate-syn", "wordrate-syn_norm", "wordrate-syn_trunc"]
keys = ["wordrate-syn", "wordrate-syn_norm-synlog_norm", "wordrate-syn_trunc"]
keys = ["wordrate-logsyn_norm-sem"]
keys = [
    "n_phones-n_words-phon_freq-freq",
    "n_phones-n_words-logsyn_norm-sem",
    "n_phones-n_words-phon_norm-logsyn_norm-sem",
    "n_phones-n_words-phon_norm-logsyn_norm-sem",
    "phon_norm-logsyn_norm-sem",
    "phon-logsyn-sem",
    "phon-logsyn_norm-sem",
    "n_phones-n_words-phon-logsyn-sem",
]
keys = [
    "logsyn-sem",
    "logsyn_norm-sem",
    "syn-sem",
    "n_phones-n_words-phon_freq-freq",
    "n_phones-n_words-logsyn_norm-sem",
    "n_phones-n_words-phon_norm-logsyn_norm-sem",
    "n_phones-n_words-phon_norm-logsyn_norm-sem",
    "phon_norm-logsyn_norm-sem",
    "phon-logsyn-sem",
    "phon-logsyn_norm-sem",
    "n_phones-n_words-phon-logsyn-sem",
]

keys = [
    "phon_freq",
    "phon_freq-logsyn",
    "phon_freq-logsyn-sem",
    "logsyn",
    "logsyn-sem",
]

names = {}
for key in keys:
    names[key] = key.split("-")

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
