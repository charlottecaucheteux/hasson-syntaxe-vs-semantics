import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from src import paths
from src.exp_singlesubject_mutifeats import run_exp_singlesub_multifeats
from src.task_dataset import get_task_df

EXP_NAME = "0317-gpt2-errors-multifeats-cvbetas-sum"
DIR_NAME = EXP_NAME
FEATURE_FOLDER = "0316-gpt2-errors"

TEST_LOCAL = False

FEATURES = [
    "freq",
    "freq-lex",
    "freq-phon",
    "freq-phon-syn",
    "freq-phon-sem",
    "freq-phon-sem-syn",
    "freq-phon-lex-sem-syn",
    "sem",
    "syn",
    "freq",
    "lex",
    "phon",
]


FEATURES = [
    "phon-syn-lex",
    "phon-syn-sem-lex",
]

FEATURES = ["lex", "syn-sem"]

FEATURES = [
    # "wordrate-syn-logsyn-sem-logsem",
    "wordrate-logsyn_norm-sem",
    "logsyn_norm-sem",
    "phon-logsyn_norm-sem",
    "wordrate-logsyn-logsyn_norm-sem",
]

FEATURES = [
    "syn",
    "logsyn",
    "logsyn_norm",
    "phon",
    "phon_norm",
    "sem",
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

OTHER_TASK = [False] * len(FEATURES)
MAX_RUN = None

RUN_PARAMS = dict(
    high_pass=None, convolve_model="fir", encoding_type="cvbetas", merge_func="sum"
)

# RUN_PARAMS = {"feature_names": ["wordpos"]}
MAX_RUN = None


def job(
    subject,
    feature_file,
    save_file,
    hemi="L",
    other_task=False,
):

    # Info
    logging.warning(f"running for : {(subject, feature_file, save_file)}")
    save_file.parent.mkdir(exist_ok=True, parents=True)

    # Run
    r = run_exp_singlesub_multifeats(
        subject,
        feature_files=[feature_file],
        hemi=hemi,
        **RUN_PARAMS,
    )

    # Save
    logging.warning(f"saving to :{save_file}")
    np.save(save_file, r)


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    save_dir = paths.scores / EXP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build params
    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))][::-1]
    dataset = get_task_df()

    params = []

    for subject in dataset.subject.unique():
        tasks = dataset.query("subject == @subject").audio_task.unique()

        for hemi in ["L", "R"]:
            for feat, other_task in zip(FEATURES, OTHER_TASK):
                # Select feature files
                feature_file = paths.embeddings / FEATURE_FOLDER / "%s" / f"{feat}.pth"
                ext = "other_task" if other_task else ""
                save_file = save_dir / (feat + ext) / f"{subject}_{hemi}.npy"
                save_file.parent.mkdir(exist_ok=True, parents=True)

                # if not save_file.is_file():

                # Check files
                for task in tasks:
                    assert Path(str(feature_file) % task).is_file(), (
                        str(feature_file) % task
                    )

                save_file.parent.mkdir(exist_ok=True, parents=True)
                params.append(
                    dict(
                        subject=subject,
                        feature_file=feature_file,
                        save_file=save_file,
                        hemi=hemi,
                        other_task=other_task,
                        tasks=",".join(tasks),
                        n_tasks=len(tasks),
                    )
                )

    print(FEATURES)

    if len(params) == 0:
        print("NO PARAMS LEFT")
    else:

        # Save params
        df = pd.DataFrame(params)
        df.to_csv(save_dir / "results_path.csv")
        np.save(save_dir / "params.npy", {"FEATURES": FEATURES, **RUN_PARAMS})

        print(
            f"""
            total number of:\n\
            parameters : {len(df)}\n\
            subjects : {df.subject.nunique()},\n\
            """
        )

        # Launch with submitit
        name = EXP_NAME
        executor = AutoExecutor(f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition="dev",
            slurm_array_parallelism=100,
            timeout_min=60 * 72,
            # cpus_per_tasks=3,
            name=name,
            # cpus_per_task=2,
        )

        key_args = [
            "subject",
            "feature_file",
            "save_file",
            "hemi",
            "other_task",
        ]
        MAX_RUN = len(df) if not MAX_RUN else MAX_RUN
        print(f"Cut to  : {MAX_RUN} runs")

        if TEST_LOCAL:
            for i in range(len(df)):
                job(*[df[k].values[i] for k in key_args])
        else:
            executor.map_array(job, *[df[k].values[:MAX_RUN] for k in key_args])
