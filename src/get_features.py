import logging
import re

import numpy as np
import spacy
import torch

from . import paths
from .constants import POSSIBLE_FEATURES, TRANSFORMER_NAMES
from .get_control_features import add_controls

# from .manning_proj_embeddings import (
#    get_manning_proj_embeddings,
#    get_ptpb_transformer_embeddings,
# )
from .preprocess_stim import (
    format_text,
    format_tokens,
    gentle_tokenizer,
    get_phone_dic,
    get_pos_dic,
    get_stimulus,
)
from .transformer_embeddings import get_transformer_embeddings


def load_precomputed_features(task, feature_files, idx=None, ext_name="features"):
    features = []
    for file in feature_files:
        feat = torch.load(file)
        if idx is not None:
            feat = feat[idx]
        features.append(feat)
    return features


def get_base_embeddings(tokens, model_name):
    print("np.array(tokens) shape", np.array(tokens).shape)
    tokens = format_tokens(np.array(tokens))
    print("np.array(tokens) shape", np.array(tokens).shape)

    if model_name in [
        "gpt2",
        "bert",
        "bert-large",
        "sum-gpt2",
        "sum-bert",
        "sum-bert-large",
        "last-gpt2",
        "last-bert",
        "last-bert-large",
    ]:
        if model_name.startswith("sum-"):
            agg = "sum"
            model_name = model_name.split("sum-")[1]
        elif model_name.startswith("last-"):
            agg = "last"
            model_name = model_name.split("last-")[1]
        else:
            agg = "mean"

        feats = get_transformer_embeddings(
            tokens,
            model_name=TRANSFORMER_NAMES[model_name],
            agg=agg,
        )

    if model_name == "manning":
        feats = get_manning_proj_embeddings(tokens)[None]
    if model_name == "bert-large-ptpb":
        feats = get_ptpb_transformer_embeddings(tokens, model_name="bert-large-cased")
    if "spacy-w2v" in model_name:
        if "sm" in model_name:
            nlp = spacy.load("en_core_web_sm")
        else:
            nlp = spacy.load("en_core_web_md")
        feats = [np.stack([i.vector for i in nlp(str(w))]).mean(0) for w in tokens]
        feats = np.stack(feats)[None]
        feats = torch.FloatTensor(feats)
    return feats


def get_average_embeddings(
    shuffleds,
    model_name,
    remove_punc=False,
):  # FIXEME BETTER DEAL WITH PUNC
    mean_hiddens = []
    for toks in shuffleds:
        if remove_punc:
            toks = np.array(
                [" ".join([w[0] for w in gentle_tokenizer(i)]).lower() for i in toks]
            ).astype(toks.dtype)
        hiddens = get_base_embeddings(toks, model_name)
        mean_hiddens.append(hiddens)
    mean_hiddens = torch.stack(mean_hiddens).mean(0)
    return mean_hiddens


def get_features(
    task,
    names=["wordpos", "seqlen", "gpt2", "bert"],
    equiv_sampling_method="order",
):

    assert equiv_sampling_method in ["order", "random", "order_random"]
    # assert np.all([i in POSSIBLE_FEATURES for i in names])
    add_pos = np.any([name == "postag" for name in names])
    stimulus = get_stimulus(task, add_phones=True, add_pos=add_pos)

    labels = []
    features = []
    for name in names:
        print(name)

        # Parse name
        if len(name.split(".")) <= 3:
            name += "..."
        model_name, shuffle_name, control_name, *_ = name.split(
            "."
        )  # model_name, shuffled, controls
        print(
            f"model_name : {model_name}, shuffle_name  : {shuffle_name}, control_name : {control_name}"
        )

        # Low level features
        labs = [model_name]
        if model_name == "wordpos":
            feats = stimulus["wordpos_in_seq"].values[None, :, None]
        elif model_name == "seqlen":
            feats = stimulus["seq_len"].values[None, :, None]
        elif model_name in ["n_words", "n_phones"]:
            feats = stimulus[model_name].values[None, :, None]
        elif model_name == "postag":
            assert "postag" in stimulus.columns
            pos_dic = get_pos_dic()
            feats = np.stack(
                [
                    np.sum([pos_dic[i] for i in postag.split("|")], axis=0)
                    for postag in stimulus["postag"]
                ]
            )
            feats = feats[None]

        elif model_name == "phones":
            phone_dic = get_phone_dic()
            feats = np.stack(
                [
                    np.sum([phone_dic[i] for i in phones.split(",")], axis=0)
                    for phones in stimulus["phones"]
                ]
            )
            feats = feats[None]

        else:  # Transformer features

            # Control tokens (if necessary)
            if control_name:
                logging.warning(f"Applying control {control_name}")
                control_stimulus = add_controls(stimulus, task, [control_name])[
                    control_name
                ]
                tokens = control_stimulus.word_raw.values
                last_suffix = f".{control_name}"
            else:
                tokens = stimulus.word_raw.values
                last_suffix = ""

            # Base Transformer features
            if "equiv" not in shuffle_name:
                suffix = ""
                if "nopunc" in shuffle_name:
                    print("LOWER")
                    tokens = np.array(
                        [
                            " ".join([w[0] for w in gentle_tokenizer(i)]).lower()
                            for i in tokens
                        ]
                    ).astype(tokens.dtype)
                    suffix = ".nopunc"
                feats = get_base_embeddings(tokens, model_name)
                if len(feats) > 0:
                    labs = [
                        f"{model_name}-{i}{suffix}{last_suffix}"
                        for i in range(len(feats))
                    ]
                else:
                    labs = [f"{model_name}{suffix}{last_suffix}"]

            # Shuffled Transformer features
            if "equiv" in shuffle_name:
                # n_seeds = #shuffle_name.split("-")[-1]
                how = shuffle_name.split("-")[1]
                assert how in ["sorted", "random"]
                print(f"EQUIVAL FROM {paths.syn_equiv_file}")
                equival = np.load(
                    str(paths.syn_equiv_file) % (task, how), allow_pickle=True
                ).item()
                equival = equival["shuffled_tokens"]

                if "idx" in shuffle_name:
                    shuffle_idx = int(shuffle_name.split("-")[-1])
                    assert shuffle_idx < len(equival)
                    equival = equival[[shuffle_idx]]
                elif "mean" in shuffle_name:
                    idx = int(shuffle_name.split("-")[-1])
                    equival = equival[:idx]
                else:
                    raise

                print("equival size", task, equival.shape)
                remove_punc = "nopunc" in shuffle_name

                print(
                    equival[0][:10],
                    equival.shape,
                )

                feats = get_average_embeddings(
                    equival,
                    model_name,
                    remove_punc=remove_punc,
                )  # DIRTY DIRTY

                if len(feats) > 0:
                    labs = [
                        f"{model_name}-{i}.{shuffle_name}" for i in range(len(feats))
                    ]
                else:
                    labs = [f"{model_name}.{shuffle_name}" for i in range(len(feats))]

        feats = torch.FloatTensor(feats)
        features.extend(feats)
        labels.extend(labs)
    return features, labels


def combine_equivalent_sentences(stimulus, result, sampling_method="order_random"):

    assert sampling_method in ["order", "random", "order_random"]

    # Combine generated sentences into new stories
    story_tokens = stimulus.word_raw.values
    story_len = len(story_tokens)
    max_len = np.max([res["tokens"].shape[1] for _, res in result.items()])
    n = len(result[0]["tokens"])

    # Checks
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
