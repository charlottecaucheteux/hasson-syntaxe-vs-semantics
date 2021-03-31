import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from src import paths
from src.exp_multitasks import run_exp_multitasks
from src.task_dataset import get_task_df

EXP_NAME = "0317-gpt2-errors-singlesubjects"
# EXP_NAME = "0311-gpt2-errors-singlesubjects"
# EXP_NAME = "0312-errors-singlesubjects"
DIR_NAME = EXP_NAME
# FEATURE_FOLDER = "0225-gpt2-errors"
# FEATURE_FOLDER = "0311-gpt2-errors"
# FEATURE_FOLDER = "0312-errors"
FEATURE_FOLDER = "0316-gpt2-errors"

LOOP = 0

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
    "phon",
    "phon-syn",
    "phon-sem",
    "phon-syn-sem",
    "phon-syn-sem-lex",
]

FEATURES = [
    "phon",
    "syn",
    "sem",
    "lex",
    "syn-lex",
    "phon-syn",
    "phon-lex",
    "phon-syn-lex",
]

FEATURES = [
    "phon",
    # "random-phon",
    # "random-syn",
    # "random-sem",
    # "random-lex",
    "phonfreq",
    "lex",
    "syn",
    "sem",
]

FEATURES = [
    # "random-lex",
    # "random-syn",
    "syn-sem",
    "lex",
    "sem",
    "syn",
    # "syn-lex",
    "wordrate",
    # "random-logsyn",
    # "random-syn",
    # "random-lex",
    "wordrate-syn-logsyn",
    "wordrate-syn-logsyn-sem-logsem",
    "syn-logsyn",
    "syn-logsyn-sem-logsem",
    "wordrate-syn",
    "wordrate-syn_norm",
    "wordrate-syn_trunc",
]

FEATURES = [
    "phon_freq",
    "phon_freq-logsyn",
    "phon_freq-logsyn-sem",
    "logsyn",
    "logsyn-sem",
]

# "syn", "lex", "sem", "syn"]

OTHER_TASK = [False] * len(FEATURES)
REGRESS_OUT = [0] * len(FEATURES)
# REGRESS_OUT = [0] * len(FEATURES)

MAX_RUN = None

RUN_PARAMS = dict(
    fit_intercept=True,
    n_folds=5,
    high_pass=None,
    convolve_model="fir",
    average_folds=True,
)

# RUN_PARAMS = {"feature_names": ["wordpos"]}
MAX_RUN = None


def job(
    subject,
    feature_file,
    save_file,
    hemi="L",
    index_regress_out=-1,
    other_task=False,
):

    # Info
    logging.warning(f"running for : {(subject, feature_file, save_file)}")
    save_file.parent.mkdir(exist_ok=True, parents=True)

    # Load regress_out indices
    if index_regress_out > 0:
        regress_out = {
            feature_file: torch.load(
                (str(feature_file) % "pieman").replace(".pth", ".idx.pth")
            )
        }
        regress_out = {lab: (v < index_regress_out) for lab, v in regress_out.items()}
    else:
        regress_out = None

    # Run
    r = run_exp_multitasks(
        subject,
        feature_files=[feature_file],
        zero_out=None,
        hemi=hemi,
        regress_out=regress_out,
        other_task=other_task,
        **RUN_PARAMS,
    )

    # Save
    logging.warning(f"saving to :{save_file}")
    np.save(save_file, r)


def job_loop(
    subject,
    feature_file,
    save_file,
    hemi=["L"],
    index_regress_out=[-1],
    other_task=[False],
):

    for sub, ff, sf, h, iro, ot in zip(
        subject, feature_file, save_file, hemi, index_regress_out, other_task
    ):
        try:
            job(
                sub,
                ff,
                sf,
                hemi=h,
                index_regress_out=iro,
                other_task=ot,
            )
        except Exception as e:
            continue


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
            for feat, regress_out, other_task in zip(FEATURES, REGRESS_OUT, OTHER_TASK):
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
                        index_regress_out=regress_out,
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
            "index_regress_out",
            "other_task",
        ]
        MAX_RUN = len(df) if not MAX_RUN else MAX_RUN
        print(f"Cut to  : {MAX_RUN} runs")

        if TEST_LOCAL:
            for i in range(len(df)):
                job(*[df[k].values[i] for k in key_args])
        elif LOOP:
            loop_params = [
                np.array_split(df[k].values[:MAX_RUN], LOOP) for k in key_args
            ]
            print("len(loop params)", len(loop_params[0]))
            executor.map_array(job_loop, *loop_params)
        else:
            executor.map_array(job, *[df[k].values[:MAX_RUN] for k in key_args])
