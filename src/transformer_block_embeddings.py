import logging
from pathlib import Path

import numpy as np
import torch
from pytorch_pretrained_bert import BertModel  # useless  for most models !!
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def extract_hiddens(words, model, tokenizer, agg="sum", cuda=False):
    inputs, mapping = map_word_to_inputs(words, tokenizer)
    assert len(inputs["input_ids"]) == 1

    # Inference
    with torch.no_grad():
        if cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        output = model(**inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states).squeeze(1)
        if cuda:
            hidden_states = hidden_states.cpu()

        if agg == "mean":
            agg_fun = torch.mean
        else:
            agg_fun = torch.sum
        # Mapping
        word_level_hidden_states = torch.stack(
            [
                agg_fun(hidden_states[:, mapping[i][0] : mapping[i][-1]], dim=1)
                for i in range(len(words))
            ],
            dim=1,
        )

    return word_level_hidden_states


def extract_hiddens_ptpb(words, model, tokenizer, agg="sum", cuda=False):
    inputs, mapping = map_word_to_inputs(words, tokenizer)
    assert len(inputs["input_ids"]) == 1

    # Inference
    with torch.no_grad():
        if cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        output = model(**inputs)  # DIFF without output_hidden_states
        hidden_states = torch.stack(output[0]).squeeze(
            1
        )  # DIFF hidden_states => outputs[0]
        if cuda:
            hidden_states = hidden_states.cpu()

        if agg == "mean":
            agg_fun = torch.mean
        else:
            agg_fun = torch.sum
        # Mapping
        word_level_hidden_states = torch.stack(
            [
                agg_fun(hidden_states[:, mapping[i][0] : mapping[i][-1]], dim=1)
                for i in range(len(words))
            ],
            dim=1,
        )

    return word_level_hidden_states


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


def get_block_embeddings(blocks, model_name="gpt2", agg="sum", cuda=False):

    """
    Block is a list of "blocks" of sentences of equal len
    Each sentence is a list of words
    """

    # Load
    assert agg in ["sum", "mean"]

    print(f"Loading model {model_name}")
    if model_name == "ptpb-bert-large-cased":
        tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
        model = BertModel.from_pretrained("bert-large-cased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    # For inference
    if cuda:
        model.to("cuda")
        print("Model CUDA")

    bar = []
    for sentences in blocks:
        if len(sentences):
            print(sentences[0][:2])
        outputs = []
        for words in tqdm(sentences):
            if model_name == "ptpb-bert-large-cased":
                hiddens = extract_hiddens_ptpb(
                    words, model, tokenizer, agg=agg, cuda=cuda
                )
            else:
                hiddens = extract_hiddens(words, model, tokenizer, agg=agg, cuda=cuda)
            assert hiddens.shape[1] == len(words)
            outputs.append(hiddens.detach())

        # Gather all sentences embeddings
        outputs = torch.stack(outputs, dim=1)  # all sentences the same len

        # Append to shuffled embeddings
        bar.append(outputs)

    return bar
