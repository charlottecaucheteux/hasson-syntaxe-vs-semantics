import numpy as np
import torch
from pytorch_pretrained_bert import BertModel
from structural_probes import probe
from transformers import AutoTokenizer

from . import paths
from .transformer_embeddings import map_word_to_inputs

"""
Almost similar to transformer_embeddings but with pytorch pretrained
bert to be sure to have the same model between probe and not probe
To change in the future (either huggingface or manning model)
#differences are marked with #DIFF
"""


def yield_hidden_states(model, inputs, max_len=256):
    idx = np.arange(inputs["input_ids"].size(1))
    splits = np.array_split(idx, len(idx) // max_len)
    for idx in splits:
        batch_inputs = {k: v[:, idx] for k, v in inputs.items()}
        outputs = model(**batch_inputs)  # DIFF no need to output_hidden_states
        hidden_states = torch.stack(outputs[0]).squeeze(
            1
        )  # DIFF (outputs[2] => outputs[0])
        yield hidden_states


def get_ptpb_transformer_embeddings(events, model_name="bert-large-cased"):

    # Load
    words = events.word_raw.values

    print(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)  # DIFF

    # Inputs
    inputs, mapping = map_word_to_inputs(words, tokenizer)

    # Inference
    with torch.no_grad():
        hidden_states = torch.cat(list(yield_hidden_states(model, inputs)), dim=1)

        # Mapping
        word_level_hidden_states = torch.stack(
            [
                torch.mean(hidden_states[:, mapping[i][0] : mapping[i][-1]], dim=1)
                for i in range(len(words))
            ],
            dim=1,
        )

    assert word_level_hidden_states.size(1) == len(events)
    return word_level_hidden_states.detach()


def get_manning_proj_embeddings(word_events):

    probe_layer = 16

    # Params
    args = {
        "model": {
            "hidden_dim": 1024,
            "model_type": "BERT-disk",
            "use_disk": False,
            "model_layer": probe_layer,
        },
        "probe": {
            "task_name": "demo",
            "maximum_rank": 1024,
            "psd_parameters": True,
            "depth_params_path": paths.probe_path
            / "example/data/bertlarge16-depth-probe.params",
            "distance_params_path": paths.probe_path
            / "example/data/bertlarge16-distance-probe.params",
        },
        "reporting": {"root": paths.probe_path / "example/results"},
        "device": "cpu",
    }

    # Define the distance probe
    with torch.no_grad():
        distance_probe = probe.TwoWordPSDProbe(args)
        distance_probe.load_state_dict(
            torch.load(
                args["probe"]["distance_params_path"], map_location=args["device"]
            )
        )

        # Extract 16 layer representation from bert-large-cased ONLY (pretrained)
        feats = get_ptpb_transformer_embeddings(
            word_events, model_name="bert-large-cased"
        )

        assert feats.shape == (
            24,
            len(word_events),
            1024,
        )  # check right model, with hugginface add embedding layer
        feats = feats[probe_layer]  # only layer 16
        feats = feats @ distance_probe.proj

    return feats.detach()
