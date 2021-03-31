import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from src import paths
from src.transformer_block_embeddings import get_block_embeddings

TEST_LOCAL = True
MAX_RUN = None
EXP_NAME = "0323_wiki_500_seeds_200_valid"
SENTS_FOLDER = "only_10m_wikisent"
# MODEL_NAMES = ["gpt2", "bert-large-cased", "bert-base-uncased"]
MODEL_NAMES = ["ptpb-bert-large-cased", "gpt2"]
N_BLOCKS = 15
RUN_PARAMS = dict(cuda=True, agg="mean")


def load_wiki_sents(input_file, min_len=5, max_len=50):
    # Load sentences and filter on valid sentences
    tmp = np.load(input_file, allow_pickle=True).item()
    gram = np.where([g.all() for g in tmp["is_grammatical"]])[0]
    dist = np.where([g.all() for g in tmp["distances"]])[0]
    lens = np.where(
        [(len(g) > min_len and len(g) < max_len) for g in tmp["base_tokens"]]
    )[0]

    idx = [i for i in gram if i in lens and i in dist]

    base = [tmp["base_tokens"][i] for i in idx]
    shuffled = [tmp["shuffled_tokens"][i] for i in idx]
    return base, shuffled


def generate_wiki_embeddings(input_file, model_name):
    base_tok, shuffled_tok = load_wiki_sents(input_file)
    base_tok = [np.array(b)[None] for b in base_tok]
    logging.warning(f"{len(base_tok)} sents")
    bar = get_block_embeddings(shuffled_tok, model_name=model_name, **RUN_PARAMS)
    base = get_block_embeddings(base_tok, model_name=model_name, **RUN_PARAMS)

    return base, bar, base_tok, shuffled_tok


def job(input_file, model_name, bar_save_file, base_save_file, tok_save_file):
    # Build dir
    for file in [bar_save_file, base_save_file, tok_save_file]:
        file.parent.mkdir(exist_ok=True, parents=True)

    # Run
    base, bar, base_tok, shuffled_tok = generate_wiki_embeddings(input_file, model_name)

    # Save
    logging.warning(f"Saving to {str(bar_save_file)}")
    torch.save(bar, bar_save_file)
    torch.save(base, base_save_file)
    np.save(tok_save_file, {"base": base_tok, "shuff": [list(i) for i in shuffled_tok]})
    # np.asarray(base_tok, dtype=object))
    # np.save(shuf_save_file, np.asarray([list(sh) for sh in shuffled_tok], dtype=object))
    # np.save(shuf_save_file, [list(sh) for sh in shuffled])


if __name__ == "__main__":

    save_dir = paths.data / "wiki_bar_embeddings" / EXP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build params

    params = []
    for model_name in MODEL_NAMES:
        n = 0
        i = 0
        while n < N_BLOCKS:
            input_file = (
                paths.syn_equiv_dir / SENTS_FOLDER / f"{i}_equival_sentences.npy"
            )
            if Path(input_file).is_file():
                save_path = save_dir / model_name / SENTS_FOLDER / f"{i}_equival_emb"
                save_path.mkdir(exist_ok=True, parents=True)
                params.append(
                    dict(
                        input_file=input_file,
                        model_name=model_name,
                        bar_save_file=save_path / "bar.pth",
                        base_save_file=save_path / "base.pth",
                        tok_save_file=save_path / "tokens.npy",
                    )
                )
                n += 1
            else:
                logging.warning(f"{input_file} does not exists")
            i += 1

    # Launch with submitit
    name = "bar-embeddings"
    executor = AutoExecutor(f"submitit_jobs/submitit_jobs/{name}")
    executor.update_parameters(
        slurm_partition="learnfair",
        slurm_array_parallelism=150,
        timeout_min=60 * 72,
        # cpus_per_tasks=4,
        name=name,
        gpus_per_node=1,
    )

    df = pd.DataFrame(params)
    df.to_csv(save_dir / "embeddings_paths.csv")
    print(f"{len(df)} params")

    np.save(
        save_dir / "params.npy",
        dict(N_BLOCKS=N_BLOCKS, SENTS_FOLDER=SENTS_FOLDER, RUN_PARAMS=RUN_PARAMS),
    )

    key_args = [
        "input_file",
        "model_name",
        "bar_save_file",
        "base_save_file",
        "tok_save_file",
    ]
    MAX_RUN = len(df) if not MAX_RUN else MAX_RUN
    print(f"Cut to  : {MAX_RUN} runs")

    if TEST_LOCAL:
        for i in range(len(df)):
            job(*[df[k].values[i] for k in key_args])
    else:
        executor.map_array(job, *[df[k].values[:MAX_RUN] for k in key_args])
