from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from submitit import AutoExecutor

from src import paths
from src.preprocess_stim import get_stimulus
from src.syntactic_equivalences import (
    build_equivalences,
    generate_sentences,
    shuffle_tokens,
)

MAX_RUN = None
TEST_LOCAL = False
EXP_NAME = "10_sampling"
PARAMS = dict(
    n_seeds=200,
    min_n_shuff=2,  # minimum 400 sentences shuffled
    min_n_valid=10,  # minimum valid sentences
    n_final_sent=10,  # final number of sentences to keep
    valid_threshold=0.8,  # minimum .8 sp correlation between distances
    max_n_shuff=50,  # max 10,000 sentences tested
)


def combine_generate_sentences(stimulus, result, sampling_method="order_random"):

    assert sampling_method in ["order", "random", "order_random"]

    # Combine generated sentences into new stories
    story_tokens = stimulus.word_raw.values
    story_len = len(story_tokens)
    max_len = np.max([res["tokens"].shape[1] for _, res in result.items()])
    n = len(result[0]["tokens"])

    # Checks
    assert n == PARAMS["n_final_sent"]
    assert np.all([len(res["tokens"]) == n for _, res in result.items()])
    assert np.sum([res["tokens"].shape[1] for _, res in result.items()]) == story_len

    # Generate story
    shuffled_story = np.empty((n, story_len), dtype="<U30")
    distances_story = np.ones((n, max_len), dtype=np.float) * np.nan
    curr = 0
    for sid, res in result.items():
        l = res["tokens"].shape[1]
        for i in range(n):
            if sampling_method == "order":
                idx = i
            elif sampling_method == "random":
                idx = np.random.randint(0, n)
            elif sampling_method == "order_random":
                idx = np.random.randint(0, i + 1)
            shuffled_story[i, curr : (curr + l)] = res["tokens"][idx]
            distances_story[i, curr : (curr + l)] = res["sp_dist"][idx]
        curr += l

    return shuffled_story, distances_story


def job(task, save_dir, sampling_method="order_random"):

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    assert sampling_method in ["order", "random", "order_random"]

    # Load nlp
    nlp = spacy.load("en_core_web_sm")

    # Load transcripts
    transcripts = list(paths.gentle_path.glob("*/transcript.txt"))
    transcripts = [open(f).read() for f in transcripts]

    # Build quivalences
    equival, counts = build_equivalences(nlp, transcripts)
    np.save(save_dir / "equival_dic.npy", equival)
    np.save(save_dir / "equival_counts.npy", counts)

    # Load stimulus
    stimulus = get_stimulus(task)

    # Generate sentence per sentence
    result = generate_sentences(
        stimulus,
        nlp,
        equival,
        equival_counts=counts,
        **PARAMS,
    )

    np.save(
        save_dir / "equival_sentences.npy",
        result,
    )

    shuffled_story, distances_story = combine_generate_sentences(
        stimulus, result, sampling_method=sampling_method
    )

    np.save(
        save_dir / "equival_story.npy",
        {
            "tokens": shuffled_story,
            "distances": distances_story,
        },
    )


if __name__ == "__main__":

    save_dir = paths.data / "syntactic_equivalences" / EXP_NAME
    save_dir.mkdir(exist_ok=True, parents=True)

    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/transcript.txt"))]

    tasks = ["tunnel"]
    params = []
    for task in tasks:
        params.append(
            dict(
                task=task,
                save_dir=str(save_dir / task),
            )
        )

    # Save params
    df = pd.DataFrame(params)
    df.to_csv(save_dir / "summary_paths.csv")
    np.save(save_dir / "params.npy", PARAMS)

    print(
        f"""
        total number of:\n\
        parameters : {len(df)}\n\
        """
    )

    # Launch with submitit
    name = EXP_NAME
    executor = AutoExecutor(f"submitit_jobs/submitit_jobs/{name}")
    executor.update_parameters(
        slurm_partition="dev",
        slurm_array_parallelism=80,
        timeout_min=60 * 48,
        # cpus_per_tasks=4,
        name=name,
        # cpus_per_task=2,
    )

    key_args = ["task", "save_dir"]

    MAX_RUN = len(df) if not MAX_RUN else MAX_RUN
    print(f"Cut to  : {MAX_RUN} runs")

    if TEST_LOCAL:
        job(*[df[k].values[0] for k in key_args])
    else:
        executor.map_array(job, *[df[k].values[:MAX_RUN] for k in key_args])
