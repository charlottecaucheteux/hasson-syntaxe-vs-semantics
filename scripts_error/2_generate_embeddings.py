import pandas as pd
import torch
from submitit import AutoExecutor

from src import paths
from src.get_features import get_features
from src.preprocess_stim import get_stimulus

TEST_LOCAL = True
MAX_RUN = None
EXP_NAME = "0316-gpt2-errors"
FEATURE_NAMES = [
    # "phon_freq",
    # "gpt2-errors",
    # "random-gpt2-errors",
    # "phonfreq",
    # "freq",
    "n_phones",
    "n_words",
    "freq",
    "phon_freq",
]

FEATURE_NAMES = ["gpt2-errors"]

# SELECTED_TASKS = ["21styear", "slumlordreach", "forgot", "tunnel"]
# SELECTED_TASKS = ["21styear", "forgot"]
# SELECTED_TASKS = ["tunnel"]

# SELECTED_TASKS = ["pieman"]
SELECTED_TASKS = []

ERR_NAMES = [
    "syn_norm",
    "logsyn_norm",
    "logsem",
    "lex_trunc",
    "syn_trunc",
    "sem_trunc",
    "phon",
    "lex",
    "syn",
    "sem",
    "loglex",
    "logsyn",
]

ERR_NAMES = [
    "phon",
    "sem_norm",
    "lex_norm",
    "phon_norm",
    "logsyn_norm",
    "logsem_norm",
    "loglex_norm",
    "logphon_norm",
]
ERR_NAMES = [
    "phon",
    "phon_norm",
]

# FEATURE_NAMES = ["freq"]


def generate_embeddings(gentle_task, feature_names):

    # Extract features if necessary
    features, labels = get_features(gentle_task, feature_names, err_names=ERR_NAMES)

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
    if len(SELECTED_TASKS):
        tasks = SELECTED_TASKS.copy()
    else:
        tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))]
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
    name = "generate-embeddings/" + EXP_NAME
    executor = AutoExecutor(f"submitit_jobs/submitit_jobs/{name}")
    executor.update_parameters(
        slurm_partition="dev",
        slurm_array_parallelism=150,
        timeout_min=60 * 72,
        # cpus_per_task=8 * 2,
        name=name,
        # gpus_per_node=2,
        # cpus_per_task=8 * 2,
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
        executor.map_array(job, *[df[k].values for k in key_args])
