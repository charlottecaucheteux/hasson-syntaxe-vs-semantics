import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def yield_hidden_states(model, inputs, max_len=256, fix_embeddings=True):
    idx = np.arange(inputs["input_ids"].size(1))
    splits = np.array_split(idx, len(idx) // max_len + 1)
    for idx in splits:
        batch_inputs = {k: v[:, idx] for k, v in inputs.items()}
        print(batch_inputs["input_ids"].shape)
        outputs = model(**batch_inputs, output_hidden_states=True)
        outputs = list(outputs.hidden_states)

        # FIX

        if fix_embeddings:
            outputs[0] = model.base_model.wte.forward(batch_inputs["input_ids"])
            print("HERE OK ")

        hidden_states = torch.stack(outputs).squeeze(1)
        yield hidden_states


def yield_hidden_states_new(
    model, inputs, max_len=256, fix_embeddings=True, cuda=False, force_causal=False,
):
    assert len(inputs["input_ids"]) == 1
    inputs = inputs["input_ids"][0]
    idx = np.arange(len(inputs))
    splits = np.array_split(idx, len(idx) // max_len + 1)
    for idx in splits:

        x = inputs[idx]

        # Only for causal 
        if force_causal:
            nw = len(x)
            x = x.expand((nw, nw))
            mask = 1 - torch.triu(torch.ones_like(x))

            if cuda:
                outs = []
                for i, (bi, mk) in enumerate(zip(x, mask)):
                    out = model(
                        bi[None], output_hidden_states=True, attention_mask=mk[None]
                    )
                    out = out.hidden_states
                    out = torch.stack([x[0, i] for x in out])
                    out = out.cpu()
                    outs.append(out)
                outs = torch.stack(outs, dim=1)
                print(outs.shape)
                assert outs.shape[1] == nw, f"{outs.shape[1]}, {nw}, {i}"
                outputs = list(outs)
            else:
                out = model(
                    x, output_hidden_states=True, attention_mask=mask
                )

                out = out.hidden_states
                outputs = [xi[torch.arange(nw), torch.arange(nw)][None] for xi in out]

        # Usual
        else:
            x = inputs[idx]
            if cuda:
                x = x.cuda()
            outputs = model(x[None], output_hidden_states=True)
            outputs = list(outputs.hidden_states)

        # For gpt2 only
        if fix_embeddings:
            outputs[0] = model.base_model.wte.forward(x)
            print("HERE OK ")
        hidden_states = torch.stack(outputs).squeeze(1)

        if cuda:
            hidden_states = hidden_states.cpu()

        yield hidden_states


def yield_hidden_states_cuda(
    model,
    inputs,
    max_len=256,
    force_causal=False,
    fix_embeddings=True,
    cuda=False,
):
    inputs = inputs["input_ids"]
    assert inputs.size(0) == 1
    idx = np.arange(inputs.size(1))  # np.arange(inputs["input_ids"].size(1))
    splits = np.array_split(idx, len(idx) // max_len + 1)
    with torch.no_grad():
        for j, idx in enumerate(splits):
            print(j)
            batch_inputs = inputs[:, idx].clone()
            if cuda:
                batch_inputs = batch_inputs.to("cuda")

            if force_causal:
                nw = len(idx)
                batch_inputs = batch_inputs.expand((nw, nw))
                mask = 1 - torch.triu(torch.ones_like(batch_inputs))

                if cuda:
                    outs = []
                    for i, (bi, mk) in enumerate(zip(batch_inputs, mask)):
                        out = model(
                            bi[None], output_hidden_states=True, attention_mask=mk[None]
                        )
                        out = out.hidden_states
                        out = torch.stack([x[0, i] for x in out])
                        out = out.cpu()
                        outs.append(out)
                    outs = torch.stack(outs, dim=1)
                    print(outs.shape)
                    assert outs.shape[1] == nw, f"{outs.shape[1]}, {nw}, {i}"
                    print(outs.shape)
                    out = list(outs)
                    # out = [x[torch.arange(nw), torch.arange(nw)][None] for x in out]

                else:
                    out = model(
                        batch_inputs, output_hidden_states=True, attention_mask=mask
                    )

                    out = out.hidden_states
                    out = [x[torch.arange(nw), torch.arange(nw)][None] for x in out]

            else:
                out = model(batch_inputs, output_hidden_states=True)
                out = list(out.hidden_states)

            if fix_embeddings:  # TOFIX, not clean, not general
                out[0] = model.base_model.wte.forward(batch_inputs)
            hidden_states = torch.stack(out).squeeze(1)
            yield hidden_states


def yield_hidden_states_general(
    model,
    inputs,
    max_len=256,
    force_causal=False,
    fix_embeddings=True,
    cuda=False,
):
    inputs = inputs["input_ids"]
    assert inputs.size(0) == 1
    idx = np.arange(inputs.size(1))  # np.arange(inputs["input_ids"].size(1))
    splits = np.array_split(idx, len(idx) // max_len + 1)
    with torch.no_grad():
        for idx in splits:

            batch_inputs = inputs[:, idx].clone()
            if cuda:
                batch_inputs = batch_inputs.to("cuda")

            if force_causal:
                nw = len(idx)
                batch_inputs = batch_inputs.expand((nw, nw))
                mask = 1 - torch.triu(torch.ones_like(batch_inputs))

                if cuda:
                    outs = []
                    for i, (bi, mk) in enumerate(zip(batch_inputs, mask)):
                        out = model(
                            bi[None], output_hidden_states=True, attention_mask=mk[None]
                        )
                        out = out.hidden_states
                        out = torch.stack([x[0, i] for x in out])
                        out = out.cpu()
                        outs.append(out)
                    outs = torch.stack(outs, dim=1)
                    print(outs.shape)
                    assert outs.shape[1] == nw, f"{outs.shape[1]}, {nw}, {i}"
                    out = list(outs)
                    # out = [x[torch.arange(nw), torch.arange(nw)][None] for x in out]

                else:
                    out = model(
                        batch_inputs, output_hidden_states=True, attention_mask=mask
                    )

                    out = out.hidden_states
                    out = [x[torch.arange(nw), torch.arange(nw)][None] for x in out]

            else:
                out = model(batch_inputs, output_hidden_states=True)
                out = list(out.hidden_states)

            if fix_embeddings:  # TOFIX, not clean, not general
                out[0] = model.base_model.wte.forward(batch_inputs)
            hidden_states = torch.stack(out).squeeze(1)
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


def get_transformer_embeddings(
    words, model_name="gpt2", agg="mean", cuda=False, force_causal=False
):

    # Load
    # words = events.word_raw.values
    assert agg in ["sum", "mean", "last"]

    print(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # For inference
    if model_name == "gpt2":
        opts = dict(force_causal=False, fix_embeddings=True, cuda=cuda)
    else:
        opts = dict(force_causal=force_causal, fix_embeddings=False, cuda=cuda)

    if cuda:
        print("CUDA")
        model.to("cuda")
        print("after model")

    # Inputs
    print("Processed words", words[:5])
    inputs, mapping = map_word_to_inputs(words, tokenizer)

    # Inference
    with torch.no_grad():
        # hidden_states = torch.cat(list(yield_hidden_states(model, inputs)), dim=1)
        hidden_states = torch.cat(
            list(yield_hidden_states_new(model, inputs, **opts)), dim=1
        )

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
