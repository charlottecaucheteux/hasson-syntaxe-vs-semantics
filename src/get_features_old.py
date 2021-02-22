import numpy as np
import torch

from . import paths
from .constants import POSSIBLE_FEATURES, TRANSFORMER_NAMES
from .manning_proj_embeddings import (
    get_manning_proj_embeddings,
    get_ptpb_transformer_embeddings,
)
from .preprocess_stim import get_phone_dic
from .shuffled_embeddings import get_shuffled_transformer_embeddings
from .transformer_embeddings import get_transformer_embeddings


def load_precomputed_features(task, feature_files, idx=None, ext_name="features"):
    features = []
    for file in feature_files:
        feat = torch.load(file)
        if idx is not None:
            feat = feat[idx]
        features.append(feat)
    return features


def get_features(events, names=["wordpos", "seqlen", "gpt2", "bert"]):
    assert np.all([i in POSSIBLE_FEATURES for i in names])
    word_events = events.query("condition=='word'")

    labels = []
    features = []
    for name in names:
        if name in ["gpt2", "bert", "bert-large"]:
            feats = get_transformer_embeddings(
                word_events, model_name=TRANSFORMER_NAMES[name]
            )
            labs = [f"{name}.{i}" for i in range(len(feats))]
        if name == "manning":
            feats = get_manning_proj_embeddings(word_events)[None]
            labs = ["manning-bert16"]
        if name == "bert-large-ptpb":
            feats = get_ptpb_transformer_embeddings(
                word_events, model_name="bert-large-cased"
            )
            labs = [f"{name}.{i}" for i in range(len(feats))]
        if "shuffled" in name:
            content_only = "posdep" not in name
            pos = "pos" if content_only else "posdep"
            name = name.split("_")[0]
            if "manning" in name:
                model_name = None
                manning = True

            else:
                model_name = TRANSFORMER_NAMES[name]
                manning = False
            feats = get_shuffled_transformer_embeddings(
                word_events,
                n_seeds=10,
                manning=manning,
                content_only=content_only,
                model_name=model_name,
            )
            if manning:
                labs = [f"manning-bert16.shuffled-{pos}"]
            else:
                labs = [f"{name}.{i}.shuffled-{pos}" for i in range(len(feats))]
        if name == "wordpos":
            feats = word_events["wordpos_in_seq"].values[None, :, None]
            feats = torch.FloatTensor(feats)
            labs = ["wordpos"]
        if name == "seqlen":
            feats = word_events["seq_len"].values[None, :, None]
            feats = torch.FloatTensor(feats)
            labs = ["seqlen"]
        if name in ["n_nords", "n_phones"]:
            feats = word_events[name].values[None, :, None]
            feats = torch.FloatTensor(feats)
            labs = [name]
        if name == "phones":
            phone_dic = get_phone_dic()
            feats = np.stack(
                [
                    np.sum([phone_dic[i] for i in phones.split(",")], axis=0)
                    for phones in word_events["phones"]
                ]
            )
            assert len(feats) == len(word_events)
            feats = torch.FloatTensor(feats)
            labs = ["phones"]

        features.extend(feats)
        labels.extend(labs)
    return features, labels
