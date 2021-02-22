import pandas as pd
import torch
from submitit import AutoExecutor

from src import paths
from src.get_features import get_features
from src.preprocess_stim import get_stimulus

TEST_LOCAL = False
MAX_RUN = None
EXP_NAME = "0206_wordembed"

FEATURE_NAMES = [
    "gpt2..equal_len_sentence",
    "gpt2..shuffle_in_sentence",
    "gpt2..shuffle_in_task",
    "n_words",
    "n_phones",
    "phones",
    "wordpos",
    "seqlen",
    "gpt2",
    # "gpt2.shuffled-posdep-1",
    # "gpt2.shuffled-posdep-5",
    # "gpt2.shuffled-posdep-10",
]

FEATURE_NAMES = [
    "gpt2.syn-equiv-1",
    "gpt2..equal_len_sentence",
    "gpt2..shuffle_in_sentence",
    "gpt2..shuffle_in_task",
    "gpt2",
    "n_words",
    "n_phones",
    "phones",
    "wordpos",
    "seqlen",
    "gpt2.syn-equiv-5",
    "gpt2.syn-equiv-10",
]

FEATURE_NAMES = [
    "gpt2.syn-equiv-1.no_punc",
    "gpt2..equal_len_sentence",
    "gpt2..shuffle_in_sentence",
    "gpt2..shuffle_in_task",
    "gpt2",
    "n_words",
    "n_phones",
    "phones",
    "wordpos",
    "seqlen",
    "gpt2.syn-equiv-5",
    "gpt2.syn-equiv-10",
]

FEATURE_NAMES = ["wordpos", "seqlen", "gpt2.syn-equiv-5", "gpt2.syn-equiv-10"]
FEATURE_NAMES = ["gpt2.syn-equiv-1"]

FEATURE_NAMES = ["spacy-w2v.syn-equiv-1", "spacy-w2v-sm.syn-equiv-1"]

FEATURE_NAMES = [
    "gpt2..nopunc",
    "gpt2.nopunc-syn-equiv-1",
]

FEATURE_NAMES = [
    # "gpt2..nopunc",
    # "gpt2.nopunc-syn-equiv-1",
    # "sum-gpt2",
    "sum-gpt2..nopunc",
    "sum-gpt2.syn-equiv-1",
    "sum-gpt2.nopunc-syn-equiv-1",
]

FEATURE_NAMES = ["last-gpt2"]
FEATURE_NAMES = ["last-gpt2.syn-equiv-1"]
FEATURE_NAMES = ["spacy-w2v-sm..nopunc", "spacy-w2v-sm.nopunc-syn-equiv-1"]
FEATURE_NAMES = ["spacy-w2v-sm.nopunc-syn-equiv-1"]

FEATURE_NAMES = [
    "sum-gpt2.equiv-1",
    "sum-gpt2",
    "sum-gpt2.equiv-10",
    # "n_words",
    # "n_phones",
    # "phones",
    # "wordpos",
    "seqlen",
    "sum-gpt2..equal_len_sentence",
    "sum-gpt2..shuffle_in_sentence",
    "sum-gpt2..shuffle_in_task",
]

FEATURE_NAMES = [f"sum-gpt2.equiv-random-{i}" for i in range(5)]
FEATURE_NAMES += [
    # "sum-gpt2.equiv-random-2",
    # "sum-gpt2.equiv-random-1",
    "sum-gpt2",
    "n_words",
    "n_phones",
    "phones",
    # "wordpos",
    # "seqlen",
    # "sum-gpt2..equal_len_sentence",
    # "sum-gpt2..shuffle_in_sentence",
    # "sum-gpt2..shuffle_in_task",
]


FEATURE_NAMES = [
    # POS TAG
    # "postag",
    # GPT2
    "sum-gpt2",
    "sum-gpt2.equiv-random-mean-10",
    # Low levels
    "n_words",
    "n_phones",
    "phones",
    "wordpos",
    "seqlen",
    # Controls
    "sum-gpt2..equal_len_sentence",
    "sum-gpt2..shuffle_in_sentence",
    "sum-gpt2..shuffle_in_task",
]
# FEATURE_NAMES += [f"sum-gpt2.equiv-random-idx-{i}" for i in range(5)]


def generate_embeddings(gentle_task, feature_names):

    # Extract features if necessary
    features, labels = get_features(gentle_task, feature_names)

    return features, labels


def job(gentle_task, feature_names, save_path):
    save_path.mkdir(exist_ok=True, parents=True)
    features, labels = generate_embeddings(gentle_task, feature_names)
    for feat, lab in zip(features, labels):
        torch.save(feat, save_path / f"{lab}.pth")


if __name__ == "__main__":

    save_dir = paths.data / "embeddings" / EXP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build params
    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))][::-1]
    params = []
    for feature_name in FEATURE_NAMES:
        for task in tasks:
            save_path = save_dir / task
            save_path.mkdir(exist_ok=True, parents=True)
            params.append(
                dict(
                    task=task,
                    feature_names=[feature_name],
                    save_path=save_path,
                )
            )

    # Launch with submitit
    name = "generate-embeddings"
    executor = AutoExecutor(f"submitit_jobs/submitit_jobs/{name}")
    executor.update_parameters(
        slurm_partition="learnfair",
        slurm_array_parallelism=150,
        timeout_min=60 * 72,
        # cpus_per_tasks=4,
        name=name,
        cpus_per_task=4,
    )

    df = pd.DataFrame(params)
    df.to_csv(save_dir / "embeddings_paths.csv")
    print(f"{len(df)} params")

    key_args = ["task", "feature_names", "save_path"]
    MAX_RUN = len(df) if not MAX_RUN else MAX_RUN
    print(f"Cut to  : {MAX_RUN} runs")

    if TEST_LOCAL:
        for i in range(len(df)):
            job(*[df[k].values[i] for k in key_args])
    else:
        executor.map_array(job, *[df[k].values[:MAX_RUN] for k in key_args])
