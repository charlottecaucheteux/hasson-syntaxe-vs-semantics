import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from src import paths
from src.exp_single_task import run_exp_single_tasks
from src.task_dataset import get_task_df

EXP_NAME = "concat-single-task-0206"
DIR_NAME = EXP_NAME
CONCAT = True
FEATURE_FOLDER = "0206_wordembed"
LOOP = 0



FEATURES = [
    # "phone_sum-gpt2-9",
    # Base
    "3_phone_features",
    # Layer 0
    "phone_sum-gpt2-0",
    "phone_sum-gpt2-0.equiv-random-mean-10",
    # Layer 9
    "phone_sum-gpt2-9",
    "phone_sum-gpt2-9.equiv-random-mean-10",
    # Controls
    # "phone_sum-gpt2-9.equal_len_sentence",
    # "phone_sum-gpt2-9.shuffle_in_sentence",
    # "phone_sum-gpt2-9.shuffle_in_task",
]

OTHER_TASK = [False] * len(FEATURES)
if CONCAT:
    REGRESS_OUT = [0] * len(FEATURES)
else:
    REGRESS_OUT = [0] + [3] * len(FEATURES)

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
    task,
    feature_file,
    save_file,
    hemis=["L", "R"],
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
    r = run_exp_single_tasks(
        subject,
        task,
        feature_files=[feature_file],
        zero_out=None,
        hemis=hemis,
        regress_out=regress_out,
        other_task=other_task,
        **RUN_PARAMS,
    )

    # Save
    logging.warning(f"saving to :{save_file}")
    np.save(save_file, r)


"""def job_loop(
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
                ssub,
                ff,
                sf,
                hemi=h,
                index_regress_out=iro,
                other_task=ot,
            )
        except Exception as e:
            continue"""


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    save_dir = paths.scores / EXP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build params
    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))][::-1]
    dataset = get_task_df()
    dataset = dataset.dropna(subset=["comprehension"])

    params = []

    for feat, regress_out, other_task in zip(FEATURES, REGRESS_OUT, OTHER_TASK):
        # Select feature files
        feature_file = paths.embeddings / FEATURE_FOLDER / "%s" / f"{feat}.pth"

        for subject in dataset.subject.unique():
            tasks = dataset.query("subject == @subject").audio_task.unique()

            for task in tasks:
                assert Path(str(feature_file) % task).is_file(), (
                    str(feature_file) % task
                )

                ext = "other_task" if other_task else ""
                save_file = save_dir / task / (feat + ext) / f"{subject}.npy"
                save_file.parent.mkdir(exist_ok=True, parents=True)

                # if not save_file.is_file():

                save_file.parent.mkdir(exist_ok=True, parents=True)
                params.append(
                    dict(
                        subject=subject,
                        task=task,
                        feature_file=feature_file,
                        save_file=save_file,
                        hemis=["L", "R"],
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
        np.save(
            save_dir / "params.npy",
            {"FEATURES": FEATURES, "FEATURE_FOLDER": FEATURE_FOLDER, **RUN_PARAMS},
        )

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
            slurm_partition="learnfair",
            slurm_array_parallelism=100,
            timeout_min=60 * 72,
            # cpus_per_tasks=3,
            name=name,
            # cpus_per_task=2,
        )

        key_args = [
            "subject",
            "task",
            "feature_file",
            "save_file",
            "hemis",
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
