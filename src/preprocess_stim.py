import json
import re
from itertools import groupby

import numpy as np
import pandas as pd
import spacy

from . import constants, paths

# add their own script to path for simplicity


def format_text(text):
    text = text.replace("\n", " ")
    text = text.replace(" -- ", ". ")
    text = text.replace(" – ", ", ")
    text = text.replace("–", "-")
    text = text.replace(' "', ". ")
    text = text.replace(' "', ". ")
    text = text.replace('" ', ". ")
    text = text.replace('". ', ". ")
    text = text.replace('." ', ". ")
    text = text.replace("?. ", "? ")
    text = text.replace(",. ", ", ")
    text = text.replace("...", ". ")
    text = text.replace(".. ", ". ")
    text = text.replace(":", ". ")
    text = text.replace("…", ". ")
    text = text.replace("-", " ")
    text = text.replace("  ", " ")
    text = text.lower()
    return text


def replace_special_character_chains(text):
    text = text.replace("-", " ")
    text = text.replace('laughs:"You', 'laughs: "You')
    return text


def gentle_tokenizer(raw_sentence):
    seq = []
    for m in re.finditer(constants.REGEX_GENTLE_TOKENIZER, raw_sentence, re.UNICODE):
        start, end = m.span()
        word = m.group()
        seq.append((word, start, end))
    return seq


def split_with_index(s, c=" "):
    p = 0
    for k, g in groupby(s, lambda x: x == c):
        q = p + sum(1 for i in g)
        if not k:
            yield p, q  # or p, q-1 if you are really sure you want that
        p = q


def format_tokens(x):
    x = np.array(x)
    fx = [format_text(" " + xi + " ").strip() for xi in x.reshape(-1)]
    fx = np.array(fx).reshape(x.shape)
    return fx


def space_tokenizer(text):
    return [(text[i:j], i, j) for i, j in split_with_index(text, c=" ")]


def match_transcript_tokens(transcript_tokens, gentle_tokens):
    transcript_line = np.array([i[1] for i in transcript_tokens])  # begin of each word
    raw_words = []
    for word, start, end in gentle_tokens:
        middle = (start + end) / 2
        diff = (middle - transcript_line).copy()
        diff[diff < 0] = np.Inf
        matching_idx = np.argmin(diff).astype(int)
        raw_words.append(transcript_tokens[matching_idx])

    return raw_words


def preproc_stim(df, text_fname):
    text = open(text_fname).read()

    text = format_text(text)
    transcript_tokens = space_tokenizer(text)
    gentle_tokens = gentle_tokenizer(text)
    assert len(gentle_tokens) == len(df)

    spans = match_transcript_tokens(transcript_tokens, gentle_tokens)
    assert len(spans) == len(gentle_tokens)

    tokens = [w[0] for w in spans]
    tokens = format_tokens(tokens)
    # text = replace_special_character_chains(text)
    # text = text.split()
    # text = [w for w in text if w if len(w.strip(string.punctuation))>0]
    # assert len(text) == len(df), f"len(transcript): {len(text)}, len(gentle_df): {len(df)}"

    # word raw
    df["word_raw"] = tokens

    # is_final_word
    begin_of_sentences_marks = [".", "!", "?"]
    df["is_eos"] = [np.any([k in i for k in begin_of_sentences_marks]) for i in tokens]

    # is_bos
    df["is_bos"] = np.roll(df["is_eos"], 1)

    # seq_id
    df["sequ_index"] = df["is_bos"].cumsum() - 1

    # wordpos_in_seq
    df["wordpos_in_seq"] = df.groupby("sequ_index").cumcount()

    # wordpos_in_stim
    df["wordpos_in_stim"] = np.arange(len(tokens))

    # seq_len
    df["seq_len"] = df.groupby("sequ_index")["word_raw"].transform(len)

    # end of file
    df["is_eof"] = [False] * (len(df) - 1) + [True]
    df["is_bof"] = [True] + [False] * (len(df) - 1)

    df["word_raw"] = df["word_raw"].fillna("")
    df["word"] = df["word"].fillna("")


def get_timing(task, subtask):
    event_meta = json.load(open(paths.event_meta_path))
    onset = event_meta[task][subtask]["onset"]
    duration = event_meta[task][subtask]["duration"]
    return onset, duration


def get_stimulus(task, add_phones=False, add_pos=False):
    stim_fname = paths.gentle_path / task / "align.csv"
    text_fname = paths.gentle_path / task / "transcript.txt"
    stimuli = pd.read_csv(stim_fname, names=["word", "word_low", "onset", "offset"])
    preproc_stim(stimuli, text_fname)
    if add_phones:
        json_name = paths.gentle_path / task / "align.json"
        dico = json.load(open(json_name, "r"))
        stimuli["phones"] = [
            [v2["phone"] for v2 in v["phones"]] if "phones" in v else []
            for v in dico["words"]
        ]
        stimuli["phones"] = [",".join(i) for i in stimuli["phones"]]
        stimuli["n_phones"] = [len(i.split(",")) for i in stimuli["phones"]]
        stimuli["n_words"] = 1

    if add_pos:
        nlp = spacy.load("en_core_web_sm")
        pos = []
        for word in stimuli["word_low"]:
            if type(word) is float:
                pos.append("")
            else:
                tok = nlp(word)
                p = [w.tag_ for w in tok]
                pos.append("|".join(p))
        stimuli["postag"] = pos

    return stimuli


def add_pulses_to_stim(stimuli, n_pulse, TR=1.5, reset_onset=False):
    """
    Return a dataframe with volumes and stimulus (similar to MOUS)
    """
    events = stimuli.copy().interpolate()
    events["word_index"] = range(len(events))
    if reset_onset:
        min_time = events.iloc[0].onset
        if min_time >= 1:  # TO CHECK WHEN IT IS STORY VS MUSIC FILE
            events[["onset", "offset"]] -= events.iloc[0].onset
    events["condition"] = "word"
    events["type"] = "word"
    pulses = pd.DataFrame(
        {
            "condition": "Pulse",
            "type": "Pulse",
            "onset": np.arange(n_pulse) * TR,
            "volume": np.arange(n_pulse),
        }
    )
    pulses["offset"] = pulses["onset"]
    events = pd.concat([events, pulses], axis=0).sort_values(["onset", "offset"])
    events["volume"] = events["volume"].fillna(method="ffill").astype(int)
    events["volume_delay"] = events.groupby("volume").cumcount() / events.groupby(
        "volume"
    )["volume"].transform("count")
    events["volume"] += events["volume_delay"]
    return events


def trim_pulses_and_stim(subj_data, events, offset, trim_init=6):
    """
    Cut extra pulses
    """
    subj_data = subj_data[trim_init:offset]
    events = events.query("volume<@offset and volume>=@trim_init")
    events["volume"] -= trim_init
    assert len(subj_data) == len(events.query("condition=='Pulse'"))
    return subj_data, events


def get_phone_dic(overwrite=False):
    """
    If does not exists, generate and save the dictionnary
    {phone: id vector} for each possible phoneme of the
    tasks.
    """

    if paths.phone_dic.is_file():
        return np.load(paths.phone_dic, allow_pickle=True).item()

    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))]
    phones = []
    for task in tasks:
        stimuli = get_stimulus(task, add_phones=True)
        phones.extend([f.split(",") for f in stimuli["phones"]])
    phones = np.concatenate(phones)
    phones = [ph for ph in phones if len(ph) > 0]
    phones = np.unique(phones)

    phone_dic = {k: np.eye(len(phones))[i] for i, k in enumerate(phones)}
    phone_dic[""] = np.zeros(len(phones))

    paths.phone_dic.parent.mkdir(exist_ok=True, parents=True)
    np.save(paths.phone_dic, phone_dic)
    return phone_dic


def get_pos_dic(overwrite=False):
    if paths.pos_dic.is_file():
        return np.load(paths.pos_dic, allow_pickle=True).item()

    print("LOADING SPACY")
    nlp = spacy.load("en_core_web_sm")
    labels = nlp.get_pipe("tagger").labels

    pos_dic = {k: np.eye(len(labels))[i] for i, k in enumerate(labels)}
    pos_dic[""] = np.zeros(len(labels))

    paths.pos_dic.parent.mkdir(exist_ok=True, parents=True)
    np.save(paths.pos_dic, pos_dic)
    return pos_dic
