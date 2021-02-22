import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def yield_hidden_states(model, inputs, max_len=256):
    idx = np.arange(inputs["input_ids"].size(1))
    splits = np.array_split(idx, len(idx) // max_len) + 1
    for idx in splits:
        batch_inputs = {k: v[:, idx] for k, v in inputs.items()}
        outputs = model(**batch_inputs, output_hidden_states=True)

        # FIX
        outputs = list(outputs[2])
        outputs[0] = model.base_model.wte.forward(batch_inputs["input_ids"])
        print("HERE OK ")
        hidden_states = torch.stack(outputs).squeeze(1)
        yield hidden_states


def map_word_to_inputs(words, tokenizer):
    mapping = {}
    idx = 0
    inputs = tokenizer("", return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.long() for k, v in inputs.items()}
    for i, word in enumerate(words):
        word_inpt = tokenizer(word, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = torch.cat([inputs[k], word_inpt[k]], dim=1).long()
        ntok = word_inpt[k].size(1)
        mapping[i] = torch.arange(idx, idx + ntok + 1)
        idx += ntok
    return inputs, mapping


def get_transformer_embeddings(words, model_name="gpt2", agg="mean"):

    # Load
    # words = events.word_raw.values
    assert agg in ["sum", "mean", "last"]

    print(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Inputs
    print("Processed words", words[:5])
    inputs, mapping = map_word_to_inputs(words, tokenizer)
    # import pdb

    # pdb.set_trace()

    # Inference
    with torch.no_grad():
        hidden_states = torch.cat(list(yield_hidden_states(model, inputs)), dim=1)

        if agg == "mean":
            logging.warning("Averaging BPE")
            # Mapping
            word_level_hidden_states = torch.stack(
                [
                    torch.mean(hidden_states[:, mapping[i][0] : mapping[i][-1]], dim=1)
                    for i in range(len(words))
                ],
                dim=1,
            )
        elif agg == "sum":
            logging.warning("Summing BPE")
            # Mapping
            word_level_hidden_states = torch.stack(
                [
                    torch.sum(hidden_states[:, mapping[i][0] : mapping[i][-1]], dim=1)
                    for i in range(len(words))
                ],
                dim=1,
            )

        elif agg == "last":
            logging.warning("Cutting to the last BPE")
            # Mapping
            word_level_hidden_states = torch.stack(
                [hidden_states[:, (mapping[i][-1] - 1)] for i in range(len(words))],
                dim=1,
            )

    assert word_level_hidden_states.size(1) == len(words)
    return word_level_hidden_states.detach()
