import logging
from pathlib import Path

import cmudict
import numpy as np
import spacy
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from . import paths
from .preprocess_stim import (
    format_text,
    format_tokens,
    gentle_tokenizer,
    get_phone_dic,
    get_pos_dic,
    get_stimulus,
)


def get_vocab_features(
    tokenizer=None,
    file=paths.gpt2_vocab_spacy_feats,
    content_only=True,
    overwrite=False,
):

    if Path(file).is_file() and not overwrite:
        output = np.load(file, allow_pickle=True).item()
        return (
            output["semantic"],
            output["syntactic"],
            output["tag_labels"],
            output["phonological"],
            output["phone_labels"],
        )
    else:
        Path(file).parent.mkdir(exist_ok=True)
        assert tokenizer is not None
        nlp = spacy.load("en_core_web_sm")
        n = tokenizer.vocab_size
        dim = len(nlp(" ")[0].vector)

        # Gather labels
        tag_labels = [""] + list(nlp.get_pipe("tagger").labels)
        tag_labels = {lab: i for i, lab in enumerate(tag_labels)}

        # import pdb
        # pdb.set_trace()
        phone_labels = ["", " "] + list(cmudict.symbols())
        phone_labels = {lab: i for i, lab in enumerate(phone_labels)}

        postags = torch.zeros((n, len(tag_labels))).long()
        phonetags = torch.zeros((n, 100)).long()
        max_n_phones = 0
        mismatch = []
        embeddings = torch.zeros((n, dim)).float()
        phonedic = dict(cmudict.entries())
        for i in trange(n):
            word = tokenizer.decode(i).strip()
            if len(word) == 0:
                word = " "
            # phones = pronouncing.phones_for_word(word)
            if word.lower() in phonedic:
                word = word.lower()
                phones = phonedic[word]
                max_n_phones = np.max([max_n_phones, len(phones)])
                for j, p in enumerate(phones):
                    phonetags[i, j] = phone_labels[p]
            else:
                mismatch.append(word)

            word = nlp(str(word))
            for w in word:
                postags[i, tag_labels[w.tag_]] += 1
                embeddings[i] += torch.Tensor(w.vector)
            embeddings[i] /= len(word)
        postags = postags.long()
        phonetags = phonetags.long()[:, :max_n_phones]

        np.save(
            file,
            {
                "semantic": embeddings,
                "syntactic": postags,
                "phonological": phonetags,
                "phone_labels": phone_labels,
                "tag_labels": tag_labels,
            },
        )

        return embeddings, postags, tag_labels, phonetags, phone_labels


def compute_error(probs, y, name="lex", vocab=None):

    assert name in ["lex", "syn", "sem", "phon"]

    # --- Lexical ---
    if name == "lex":
        error = torch.gather(probs, -1, y[:, None]).squeeze()

    # --- Semantic ---
    elif name == "sem":
        assert vocab is not None
        semantic_pred = probs[:, :, None] * vocab[None]
        semantic_pred = semantic_pred.sum(-2)
        semantic_true = vocab[y]
        error = 1 - nn.functional.cosine_similarity(semantic_pred, semantic_true)
        del semantic_true, semantic_pred

    # --- Syntactic ---
    elif name == "syn":
        assert vocab is not None

        syntactic_pred = (
            probs[:, :, None] * vocab[None]
        )  # of shape (bs, vocab_size, num_pos)
        syntactic_pred = syntactic_pred.sum(-2)  # (bs, num_pos)
        true_index = torch.argmax(vocab[y], dim=-1)
        error = torch.gather(syntactic_pred, -1, true_index[:, None]).squeeze()
        del syntactic_pred, true_index

    # --- Phon ---
    else:
        assert vocab is not None

        # 1 if similar phone between y and vocab word at position j
        error = vocab[y].unsqueeze(1) == vocab[None]

        # Ignore unmatched phones
        error &= vocab[y].unsqueeze(1) > 0

        # 1 if all similar phone between y vocab word at position <=j
        error = error.cumprod(-1).float()

        # Corresponding prob of vocab words
        error *= probs[:, :, None]

        # Sum over vocab words (=> cumulative freq of words in the remaining cohort f(C_phi_t = A))
        error = error.sum(-2).float()

        # Divide by the sum of probability of the previous phonemes
        denom = error.clone()
        denom[denom == 0] = 1.0
        error[:, 1:] /= denom[:, :-1]

        # Probability of one when only one possibility
        error[error == 0] = 1.0

        # The probability of the sequence of phoneme is the product of probability of each phoneme # => mean error per phoneme
        error[error > 0] = np.log10(error[error > 0])
        error = error.mean(-1)
        # error = error.prod(-1)
        del denom

    return error


def extract_sub_level_probs(
    probs,
    inputs,
    sem_vocab,
    syn_vocab,
    ph_vocab,
    err_names=["lex"],
    words=None,
):
    possible = ["lex", "sem", "syn", "phon", "loglex", "logsyn", "logsem", "logphon"]
    possible = (
        possible + [w + "_norm" for w in possible] + [w + "_trunc" for w in possible]
    )
    possible = possible + [w + "_trunc" for w in possible]
    issues = [i for i in err_names if i not in possible]
    assert np.all([i in possible for i in err_names]), f"{issues} not in {err_names}"

    y = inputs["input_ids"].reshape(-1).clone()
    del inputs

    # Truncate distribution
    trunc_probs = truncate_topk_logits(probs, k=0.9, mink=40)
    print("trunc_probs", trunc_probs.shape)

    all_errors = []
    for name in err_names:

        if "norm" in name:
            syntax_vocab = torch.FloatTensor(StandardScaler().fit_transform(syn_vocab))
            semantic_vocab = torch.FloatTensor(
                StandardScaler().fit_transform(sem_vocab)
            )
            phon_vocab = torch.FloatTensor(StandardScaler().fit_transform(ph_vocab))
        else:
            syntax_vocab = syn_vocab.clone()
            semantic_vocab = sem_vocab.clone()
            phon_vocab = ph_vocab.clone()

        if "trunc" in name:
            curr_probs = trunc_probs.clone()
        else:
            curr_probs = probs.clone()

        # error = compute_error(probs, y, name="lex", vocab=None)

        if "lex" in name:

            # Lexical
            error = torch.gather(curr_probs, -1, y[:, None]).squeeze()

        elif "sem" in name:
            # Semantic pred
            semantic_pred = curr_probs[:, :, None] * semantic_vocab[None]
            semantic_pred = semantic_pred.sum(-2)

            # Semantic error
            semantic_true = semantic_vocab[y]
            error = 1 - nn.functional.cosine_similarity(semantic_pred, semantic_true)
            del semantic_true, semantic_pred

        elif "syn" in name:

            syntactic_pred = (
                curr_probs[:, :, None] * syntax_vocab[None]
            )  # of shape (bs, vocab_size, num_pos)
            syntactic_pred = syntactic_pred.sum(-2)  # (bs, num_pos)
            true_index = torch.argmax(syntax_vocab[y], dim=-1)
            error = torch.gather(syntactic_pred, -1, true_index[:, None]).squeeze()
            del syntactic_pred, true_index

        elif "phon" in name:

            # 1 if similar phone between y and vocab word at position j
            error = phon_vocab[y].unsqueeze(1) == phon_vocab[None]

            # Ignore unmatched phones
            error &= phon_vocab[y].unsqueeze(1) > 0

            # 1 if all similar phone between y vocab word at position <=j
            error = error.cumprod(-1).float()

            # Corresponding prob of vocab words
            error *= curr_probs[:, :, None]

            # Sum over vocab words (=> cumulative freq of words in the remaining cohort f(C_phi_t = A))
            error = error.sum(-2).float()

            # Divide by the sum of probability of the previous phonemes
            denom = error.clone()
            denom[denom == 0] = 1.0
            error[:, 1:] /= denom[:, :-1]

            # Probability of one when only one possibility
            error[error == 0] = 1.0

            # The probability of the sequence of phoneme is the product of probability of each phoneme
            error = error.prod(-1)  # to change. MEAN surprisal.

            del denom

        else:
            raise

        print("extract_sub_level_errors", error.shape)
        # else : if na

        if "log" in name:
            # log_error = torch.zeros(error.shape)
            error[error > 0] = torch.log10(error[error > 0] * 1e9)
            # error = -log_error.clone()

        print("ok")
        all_errors.append(error)
        del error, syntax_vocab, phon_vocab, semantic_vocab, curr_probs

    del trunc_probs, probs

    all_errors = torch.stack(all_errors)
    print(all_errors.shape)
    # all_errors = np.nan_to_num(all_errors.numpy())
    all_errors = torch.FloatTensor(all_errors)

    print("end error")

    return all_errors


def yield_probs(causal_model, inputs, max_len=256, bos_token_id=50256):
    bos_token_id = torch.LongTensor([[bos_token_id]])
    x = inputs["input_ids"]
    idx = np.arange(x.size(1))
    print(idx)
    splits = np.array_split(idx, len(idx) // (max_len - 1) + 1)
    for idx in splits:
        batch_inputs = x[:, idx]
        batch_inputs = torch.cat(
            [bos_token_id, batch_inputs], dim=-1
        )  # Add begin of sentence to predict first token
        logits = causal_model(batch_inputs)["logits"]
        probs = nn.functional.softmax(logits, dim=-1)
        vocab_size = probs.shape[-1]
        probs = probs.reshape((-1, vocab_size))
        probs = probs[:-1]  # Only keep actual tokens
        assert len(probs) == len(idx)
        yield probs


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


def map_outputs_to_word(output, mapping, words, agg="sum"):
    agg_fun = torch.mean if agg == "mean" else torch.sum
    word_level = torch.stack(
        [
            agg_fun(output[mapping[i][0] : mapping[i][-1]], dim=0)
            for i in range(len(words))
        ]
    )
    return word_level


def truncate_topk_logits(logits, k=0.9, mink=40):
    # Filter on probs > 90
    sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
    # assert torch.isclose(sorted_logits.sum(-1), torch.ones(sorted_logits.shape[:-1])).all()

    # Method 1
    sorted_mask = sorted_logits.cumsum(-1) > k
    sorted_mask[:, :mink] = False  # minimum 40 values per word
    out = logits.clone()
    for i in range(idx.shape[0]):
        to_zero = idx[i][sorted_mask[i]]
        out[i, to_zero] = 0

    """# Method 2
    sorted_mask = sorted_logits.cumsum(-1) <= k
    out = torch.zeros(*sorted_logits.shape[:-1], sorted_logits.shape[-1]+1)
    out.scatter_add_(dim=-1, index=(idx.long() + 1) * sorted_mask, src=sorted_logits)
    out = out[:,1:]"""

    # assert torch.isclose(out1, out).all()
    return out


def get_transformer_errors(
    words,
    model_name="gpt2",
    err_names=["lex"],
    random=False,
):

    # Load
    # words = events.word_raw.values

    print(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if random:
        config = AutoConfig.from_pretrained(model_name)
        causal_model = AutoModelForCausalLM.from_config(config)
    else:
        causal_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Inputs
    print("Processed words", words[:5])
    inputs, mapping = map_word_to_inputs(words, tokenizer)

    # Vocabulary mapping with POS TAG / word embeddings / (+ ADD PHONES ?)
    (
        semantic_vocab,
        syntax_vocab,
        tag_labels,
        phon_vocab,
        phon_labels,
    ) = get_vocab_features(tokenizer, file=paths.gpt2_vocab_spacy_feats)

    # Inference
    with torch.no_grad():
        probs = torch.cat(list(yield_probs(causal_model, inputs)))
        errors = extract_sub_level_probs(
            probs,
            inputs,
            semantic_vocab,
            syntax_vocab,
            phon_vocab,
            err_names=err_names,
        )
        print("get_transformer_errors", errors.shape)
        agg = "sum"  # if log else "prod"
        print(agg)
        word_level_errors = [
            map_outputs_to_word(err, mapping, words, agg=agg) for err in errors
        ]
        print("after map_outputs_to_word")
        word_level_errors = torch.stack(word_level_errors)
        print("get_transformer_errors after mapping", word_level_errors.shape)

    assert word_level_errors.size(1) == len(words)
    return word_level_errors.detach()
