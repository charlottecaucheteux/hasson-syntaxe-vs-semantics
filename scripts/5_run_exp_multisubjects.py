import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from src import paths
from src.exp_multisubjects import run_exp_multisubjects

# EXP_NAME = "200-concat-multisubjects-0201-valid"
# EXP_NAME = "100-concat-multisubjects-0206-wordemb"
EXP_NAME = "concat-multisubjects-0201-valid"
# EXP_NAME = "icml-review-multisubjects-newmodels-[0321_hugg_models]"
EXP_NAME = "icml-review-multisubjects-newmodels-[0321_hugg_models_bidir]"
EXP_NAME = "icml-review-multisubjects-delayed-[0206_wordembed]"
# EXP_NAME = "icml-review-multisubjects-jr_2v2-[0206_wordembed]"
# EXP_NAME = "icml-review-multisubjects-VALID-v2v_per_voxel-[0206_wordembed]"

# EXP_NAME = "icml-review-multisubjects-v2v-[0206_wordembed]"
# EXP_NAME = "icml-review-multisubjects-v2v-[0206_wordembed]-10folds-concat"

DIR_NAME = EXP_NAME
# FEATURE_FOLDER = "0201_wiki_valid"
FEATURE_FOLDER = "0206_wordembed"
# FEATURE_FOLDER = "0321_hugg_models"
# FEATURE_FOLDER = "0323_hugg_models_bidir"

MAX_RUN = None
RUN_PARAMS = dict(
    fit_intercept=True,
    n_folds=100,
    average_folds=False,
    high_pass=None,
    convolve_model="fir",
    alpha_per_target=True,
    metric="correlate",  # v2v
)


TEST_LOCAL = True
FILTER_ON_EXIST = False

FEATURES = [
    "phone_sum-gpt2-9",
    "phone_sum-gpt2-9.equiv-random-mean-10",
    # Base
    "3_phone_features",
    # "phone_postag",
    # Layer 0
    "phone_sum-gpt2-0",
    "phone_sum-gpt2-0.equiv-random-mean-10",
    #  "phone_semantic-0.equiv-random-mean-10",
    # Layer 9
    # "phone_semantic-9.equiv-random-mean-10",
    # "phone_embedding_semantic-9.equiv-random-mean-10",
    # Controls
    # "phone_sum-gpt2-9.equal_len_sentence",
    # "phone_sum-gpt2-9.shuffle_in_sentence",
    # "phone_sum-gpt2-9.shuffle_in_task",
]

# for layer in [0, 9]:
#    for i in range(5):
#        FEATURES += [f"phone_sum-gpt2-{layer}.equiv-random-idx-{i}"]
#        FEATURES += [f"phone_embedding_sum-gpt2-{layer}.equiv-random-idx-{i}"]

FEATURES += [
    "phone_embedding_sum-gpt2-9.equiv-random-mean-10",
]


FEATURES = ["3_phone_features"]
for layer in range(13):
    FEATURES.append(f"phone_sum-gpt2-{layer}.equiv-random-mean-10")
    FEATURES.append(f"phone_sum-gpt2-{layer}")

FEATURES += [
    "phone_sum-gpt2-9.equal_len_sentence",
    "phone_sum-gpt2-9.shuffle_in_sentence",
    "phone_sum-gpt2-9.shuffle_in_task",
]

FEATURES = [
    "3_phone_features",
    # Layer 0
    "phone_sum-gpt2-0",
    "phone_sum-gpt2-0.equiv-random-mean-10",
    # Layer 9
    "phone_sum-gpt2-9",
    "phone_sum-gpt2-9.equiv-random-mean-10",
    # "phone_embedding_sum-gpt2-9.equiv-random-mean-10",
]


# FEATURES = ["3_phone_features"]
models = [
    ("bert-base-uncased", 8),  # 12
    ("xlnet-base-cased", 8),  # 12
    ("roberta-base", 8),  # 12
    # "longformer-base-4096",  # 12
    # "squeezebert-mnli",
    # "layoutlm-base-uncased",
    ("albert-base-v1", 8),  # 12
    ("distilgpt2", 4),  # 6 (layer 4)
    ("transfo-xl-wt103", 12),  # 18 (-> layer 12?)
    ("distilbert-base-uncased", 4),  # 6 (layer 4)
]

# for model, layer in models:
#     k = f"phone_sum-{model}-{layer}"
#     FEATURES.extend([k, f"{k}.equiv-random-mean-10"])

FEATURES = [
    "6-delayed-sum-gpt2-0",
    "6-delayed-sum-gpt2-0.equiv-random-mean-10",
    "phone_6-delayed-sum-gpt2-0",
    "phone_6-delayed-sum-gpt2-0.equiv-random-mean-10",
]

# FEATURES = ["sum-gpt2-0", "6-delayed-sum-gpt2-0"]

# OTHER_TASK = [False, False] + [True] * len(LAYERS) + [False] * len(FEATURES)
CONCAT = True
OTHER_TASK = [False] * len(FEATURES)
if CONCAT:
    REGRESS_OUT = [0] * len(FEATURES)
else:
    REGRESS_OUT = [0] + [3] * len(FEATURES)


AGG_BOLD = "mean"


def job(feature_file, save_file, hemi, index_regress_out=0, other_task=False):

    # Info
    logging.warning(f"running for : {(feature_file, save_file, hemi)}")

    Path(save_file).parent.mkdir(exist_ok=True, parents=True)

    # Load bold
    if AGG_BOLD == "mean":
        bold = np.load(str(paths.mean_bolds) % hemi, allow_pickle=True).item()
    else:
        bold = np.load(str(paths.median_bolds) % hemi, allow_pickle=True).item()["bold"]
    print("Loaded for hemi {}", feature_file)

    # Load regress_out indices
    if index_regress_out > 0:
        regress_out = {
            feature_file: torch.load(
                (str(feature_file) % "pieman").replace(".pth", ".idx.pth")
            )
        }
        # import pdb

        # pdb.set_trace()
        regress_out = {lab: (v < index_regress_out) for lab, v in regress_out.items()}
    else:
        regress_out = None

    print(regress_out)

    # Run
    r = run_exp_multisubjects(
        bold,
        feature_files=[feature_file],
        zero_out=None,
        regress_out=regress_out,
        other_task=other_task,
        **RUN_PARAMS,
    )

    # Save
    logging.warning(f"saving to :{save_file}")
    np.save(save_file, r)

    return r

    # Save
    logging.warning(f"saving to :{save_file}")
    np.save(save_file, r)


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    save_dir = paths.scores / DIR_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))][::-1]

    params = []

    for hemi in ["L", "R"]:
        for feat, regress_out, other_task in zip(FEATURES, REGRESS_OUT, OTHER_TASK):
            # Select feature files
            feature_file = paths.embeddings / FEATURE_FOLDER / "%s" / f"{feat}.pth"
            ext = "other_task" if other_task else ""
            save_file = save_dir / (feat + ext) / f"{hemi}.npy"

            if not save_file.is_file() or not FILTER_ON_EXIST:

                # Check files
                for task in tasks:
                    assert Path(str(feature_file) % task).is_file(), (
                        str(feature_file) % task
                    )
                save_file.parent.mkdir(exist_ok=True, parents=True)
                params.append(
                    dict(
                        feature_file=feature_file,
                        save_file=save_file,
                        hemi=hemi,
                        index_regress_out=regress_out,
                        other_task=other_task,
                    )
                )

    print(FEATURES)

    # Save params
    df = pd.DataFrame(params)
    df.to_csv(save_dir / "results_path.csv")
    np.save(
        save_dir / "params.npy",
        {
            "FEATURES": FEATURES,
            "FEATURE_FOLDER": FEATURE_FOLDER,
            "AGG_BOLD": AGG_BOLD,
            **RUN_PARAMS,
        },
    )

    print(
        f"""
        total number of:\n\
        parameters : {len(df)}\n\n
        """
    )

    # Launch with submitit
    name = EXP_NAME
    executor = AutoExecutor(f"submitit_jobs/submitit_jobs/{name}")
    executor.update_parameters(
        slurm_partition="learnfair",
        # comment="ICML",
        slurm_array_parallelism=100,
        timeout_min=60 * 72,
        cpus_per_task=8 * 4,
        name=name,
        # cpus_per_task=2,
    )

    key_args = ["feature_file", "save_file", "hemi", "index_regress_out", "other_task"]

    MAX_RUN = len(df) if not MAX_RUN else MAX_RUN
    print(f"Cut to  : {MAX_RUN} runs")

    if TEST_LOCAL:
        for i in range(len(df)):
            job(*[df[k].values[i] for k in key_args])
    else:
        executor.map_array(job, *[df[k].values[:MAX_RUN] for k in key_args])
