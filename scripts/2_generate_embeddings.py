import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from src import paths
from src.get_features import get_features
from src.preprocess_stim import get_stimulus

TEST_LOCAL = True
MAX_RUN = None

EXP_NAME = "0206_wordembed"
EXP_NAME = "0321_hugg_models"
EXP_NAME = "0323_hugg_models_bidir"
SELECT_TASKS = ["slumlordreach", "21styear", "sherlock", "tunnel"]
# SELECT_TASKS = []

RUN_PARAMS = dict(cuda=True, force_causal=False)

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

FEATURE_NAMES = ["sum-gpt2.equiv-random-mean-10"]


models = [
    "albert-base-v1",
    "bert-base-uncased",
    "squeezebert-mnli",
    "xlnet-base-cased",
    "roberta-base",
    "transfo-xl-wt103",
    "longformer-base-4096",
]

models = [
    "bert-base-uncased",
    "xlnet-base-cased",
    "roberta-base",
    "longformer-base-4096",
    "squeezebert-mnli",
    "transfo-xl-wt103",
    "distilbert-base-uncased",
    "distilgpt2",
    "layoutlm-base-uncased",
    "albert-base-v1",
]

models = [
    "transfo-xl-wt103",
    "distilbert-base-uncased",
]

# models = ["bert-base-uncased"]
# "roberta-base-openai-detector",
# "distilbert-base-uncased",
# "distilgpt2",
# "microsoft/layoutlm-base-uncased",


FEATURE_NAMES = []
for model in models:
    FEATURE_NAMES += [f"sum-{model}.equiv-random-mean-10", f"sum-{model}"]
    # FEATURE_NAMES += [f"sum-{model}"]  # f"sum-{model}",

# FEATURE_NAMES += [f"sum-gpt2.equiv-random-idx-{i}" for i in range(5)]


def generate_embeddings(gentle_task, feature_names):

    # Extract features if necessary
    features, labels = get_features(gentle_task, feature_names, **RUN_PARAMS)

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

    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))]
    if len(SELECT_TASKS):
        assert np.all([i in tasks for i in SELECT_TASKS])
        tasks = SELECT_TASKS

    params = []
    for feature_name in FEATURE_NAMES[::-1]:
        for task in tasks[::-1]:
            save_path = save_dir / task
            save_path.mkdir(exist_ok=True, parents=True)
            if not (
                save_dir
                / task
                / (
                    feature_name.replace(
                        ".equiv-random-mean-10", "-0.equiv-random-mean-10.pth"
                    )
                )
            ).is_file():
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
        slurm_partition="dev",
        slurm_array_parallelism=150,
        timeout_min=60 * 72,
        # cpus_per_tasks=4,
        name=name,
        gpus_per_node=1,
    )

    df = pd.DataFrame(params)
    df.to_csv(save_dir / "embeddings_paths.csv")
    np.save(save_dir / "params.npy", RUN_PARAMS)
    print(f"{len(df)} params")

    key_args = ["task", "feature_names", "save_path"]
    MAX_RUN = len(df) if not MAX_RUN else MAX_RUN
    print(f"Cut to  : {MAX_RUN} runs")

    if TEST_LOCAL:
        for i in range(len(df)):
            job(*[df[k].values[i] for k in key_args])
    else:
        executor.map_array(job, *[df[k].values[:MAX_RUN] for k in key_args])
