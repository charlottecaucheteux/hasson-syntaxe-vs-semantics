import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import torch
from scipy.stats import spearmanr
from tqdm.notebook import tqdm

from src import paths


def build_equivalences(nlp, sentences):
    """
    Build a dictionary that aggregate all words that have the same
    tag (enhanced POS), such that the plural / infinitive etc are
    respected.
    """

    equival = dict()
    counts = dict()

    # For each sentence
    for sentence in tqdm(sentences):

        # for each word
        doc = nlp(sentence)
        for word in doc:
            for non_shuffled in range(2):

                info = word.tag_ + "|" + word.pos_
                if not non_shuffled:
                    info += "|" + word.dep_
                word_str = word.text
                # identify enhanced part of speech information
                if info not in equival.keys():
                    equival[info] = set()
                    counts[info] = defaultdict(lambda: 0)

                # Clean formatting
                word_str = word_str.lower()
                if word_str[0] == " ":
                    word_str = word_str[1:]

                equival[info].add(word_str)
                counts[info][word_str] += 1

    equival = {k: list(v) for k, v in equival.items()}  # dirty
    counts = {
        k: {k2: v[k2] for k2 in equival[k]} for k, v in counts.items()
    }  # reorder to match equival order # dirty
    return equival, counts


def shuffle_tokens(
    nlp,
    equival,
    tokens,
    counts=None,
    n=1,
    non_shuffled="POS+DEP",
    keep_punc=True,
    keep_function_word=False,
    force_replace=True,
):
    """
    Return a list of tokens with similar POS // POS+TAG
    Remark: we use as list of tokens as input in order to return the extact same number of tokens as output
    That is why we use spacy 'spans'.
    """

    # Text
    text = " ".join(tokens)

    # Get original dependency tree
    tree = nlp(text).get_lca_matrix()

    # Compute spacy spans : only useful for alignment between generated and initial tokens
    # Some `token` contain one word, but several spacy `unit words` (e.g when there is a dot
    # e.g at the end 'on the mat.')
    # Easier to do like this because 1) I could not find the spacy tokenizer 2) it allows us to use
    # the gentle tokenizer for all steps (model inference, matching with gentle dataframe, spacy)
    spans = []
    start = 0
    for tok in tokens:
        end = start + len(tok)
        spans.append((tok, start, end))
        start = end + 1

    # Convert to spacy
    doc = nlp(text)
    spacy_spans = [doc.char_span(start, end) for (_, start, end) in spans]
    assert len(spacy_spans) == len(spans) == len(tokens)

    # Shuffle each span
    shuffled_tokens = []
    for span in spacy_spans:

        shuffled = ["" for i in range(n)]
        for word in span:
            if (
                word.pos_ in ("AUX", "ADP", "SYM", "NUM", "PRON", "DET")
                and keep_function_word
            ) or (word.pos_ in ("PUNCT") and keep_punc):
                for i in range(n):
                    shuffled[i] += " " + str(word)
                continue

            # get alternative words
            word_info = word.tag_ + "|" + word.pos_
            if non_shuffled == "POS+DEP":
                word_info += "|" + word.dep_
            # elif non_shuffled == 'POS':
            #    raise

            word_text = word.text.lower().replace(" ", "")
            proba = None
            if word_info not in equival:
                logging.warning(f"{word_info} not in equival dic")
                alternatives = [word_text]
            else:
                alternatives = deepcopy(equival[word_info])
                if counts is not None:
                    assert (
                        np.array(list(counts[word_info].keys()))
                        == np.array(alternatives)
                    ).all()
                    proba = list(counts[word_info].values())
                if len(alternatives) == 1:
                    logging.warning(
                        f"Only one word in category for word {word_text}:{alternatives[0]}"
                    )
            if force_replace and len(alternatives) > 1:
                if proba is not None:
                    proba = [p for w, p in zip(alternatives, proba) if w != word_text]
                alternatives = [w for w in alternatives if w != word_text]

            # pick with/witout prob
            if proba is not None:
                proba = np.array(proba) / sum(proba)
            random_word = [
                np.random.choice(alternatives, 1, p=proba)[0] for i in range(n)
            ]

            # add upper case
            if str(word)[0].isupper():
                for i in range(n):
                    random_word[i] = random_word[i][0].upper() + random_word[i][1:]

            # add to new sentence
            for i in range(n):
                shuffled[i] += " " + random_word[i]

        shuffled_tokens.append([s.strip() for s in shuffled])

    assert len(shuffled_tokens) == len(spans)
    shuffled_tokens = np.transpose(shuffled_tokens)

    # ---- Distances
    doc = nlp(" ".join(tokens))

    # Distances LCA
    lca = doc.get_lca_matrix()
    lca_distances = list()
    for shuffle in shuffled_tokens:
        lca_shuffled = nlp(" ".join(shuffle)).get_lca_matrix()
        if len(lca) != len(lca_shuffled):
            lca_distances.append(np.inf)
            continue
        lca_distances.append(((lca - lca_shuffled) ** 2).sum())
    lca_distances = np.array(lca_distances)

    # Manning check
    true_distances = get_true_distance_matrix(doc)
    sp_distances = list()
    for shuffle in shuffled_tokens:
        shuff_distances = get_true_distance_matrix(nlp(" ".join(shuffle)))
        if len(shuff_distances) != len(true_distances):
            sp_distances.append(-np.inf)
            continue
        sp_distances.append(vector_spearmanr(true_distances, shuff_distances))
    sp_distances = np.array(sp_distances)

    # Grammatical check

    return shuffled_tokens, lca_distances, sp_distances


def vector_spearmanr(true, pred):
    assert true.shape == pred.shape
    corr = [spearmanr(true[i], pred[i])[0] for i in range(len(true))]
    corr = np.array(corr)
    return corr.mean()


def get_true_distance_matrix(doc):
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((token.i, child.i))
    graph = nx.Graph(edges)

    distances = list(nx.all_pairs_shortest_path_length(graph))
    nodes = [n for (n, _) in distances]

    distances = {(node, k): v for (node, d) in distances for (k, v) in d.items()}
    if len(nodes) == 0:
        return np.zeros((0, 0))
    matrix = np.zeros([int(max(nodes) + 1)] * 2)
    for (i, j), v in distances.items():
        matrix[i, j] = v
    assert np.allclose(matrix, matrix.T)
    return matrix


def generate_sentences(
    stimulus,
    nlp,
    equival,
    equival_counts=None,
    n_seeds=50,
    min_n_shuff=40, # minimum 20 * 50 sentences shuffled
    min_n_valid=50, # minimum valid sentences
    n_final_sent=10, # final number of sentences to keep
    valid_threshold=0.8,
    max_n_shuff=1000,
):
    print("changed")
    result = {}
    for sid, tokens in tqdm(
        stimulus.groupby("sequ_index")["word_raw", "wordpos_in_stim"]
    ):
        tokens = tokens.sort_values("wordpos_in_stim")["word_raw"].values
        result[sid] = {"tokens": [], "lca_dist": [], "sp_dist": []}
        n_valid = 0
        n_shuff = 0
        while n_valid < min_n_valid or n_shuff < min_n_shuff:
            shuffled_tokens, lca_dist, sp_dist = shuffle_tokens(
                nlp,
                equival,
                tokens,
                counts=equival_counts,
                n=n_seeds,
                non_shuffled="POS+DEP",
                keep_punc=True,
                keep_function_word=False,
                force_replace=True,
            )
            if n_shuff < max_n_shuff:
                # import pdb

                # pdb.set_trace()
                valids = np.where(np.array(sp_dist).astype(float) > valid_threshold)[
                    0
                ].astype(int)
            else:
                valids = np.arange(len(sp_dist))
            result[sid]["tokens"].extend(shuffled_tokens[valids])
            result[sid]["lca_dist"].extend(lca_dist[valids])
            result[sid]["sp_dist"].extend(sp_dist[valids])
            n_valid += len(valids)
            n_shuff += 1
            print(n_valid)
            print(n_shuff)

        order = np.argsort(result[sid]["sp_dist"])[::-1]
        for k, v in result[sid].items():
            result[sid][k] = np.array(v)[order][:n_final_sent]
            result[sid][k] = np.stack(result[sid][k])

    return result
