import re
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy

from . import paths
from .preprocess_stim import format_text, gentle_tokenizer, get_stimulus
from .task_dataset import get_task_df


def build_sentence_dic(overwrite=True):
    if paths.wiki_seq_len_dic.is_file() and not overwrite:
        return np.load(paths.wiki_seq_len_dic, allow_pickle=True).items()
    text = open(paths.wiki_100m_path).read().split("\n")
    text = [par for par in text if len(par.split()) > 30]
    text = " ".join(text)
    text = format_text(text)
    text = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    #doc = nlp(text)

    dic = defaultdict(lambda: [])
    for sent in text:
        ln = len(sent.split())
        dic[ln].append(sent.strip().split())
    #np.save(paths.wiki_seq_len_dic, dic)

    keys = np.array(list(dic.keys()))
    for k in range(150):
        if k not in keys:
            closest = int(np.argmin(np.abs(k - keys)))
            closest = keys[closest]
            dic[k] = [dic[closest][0][:closest] + ["."] * (k - closest)]

    np.save(paths.wiki_seq_len_dic, dict(dic), allow_pickle=True)
    return dic

    
def add_punc(words_without_punc, words_with_punc):
    new = [re.sub("\w+", str(first), str(second)) for first, second in zip(words_without_punc, words_with_punc)]
    return new


def add_upper(words, idx):
    new = pd.Series(words)
    new[idx] = new[idx].str.slice(stop=1).str.upper() + new[idx].str.slice(start=1)
    return new.values


def add_controls(stimuli, audio_task, control_names=["nopunc",
                           "shuffle", 
                           "shuffle_in_task", 
                           "other_task", 
                           "equal_len_sentence",
                           "shuffle_in_sentence", 
                           "shuffle_in_sentence_nopunc",
                           "shuffle_with_punc"]):
    control_stim = {}
    
    df_tasks = get_task_df()
    all_tasks = df_tasks.audio_task.unique()
    np.random.shuffle(all_tasks)
    all_words = np.concatenate([get_stimulus(task).word_raw.dropna().values for task in all_tasks])
    all_words_low = np.concatenate([get_stimulus(task).word_low.dropna().values for task in all_tasks])
    np.random.shuffle(all_words)

    for control in control_names:
        print(control)
        assert control in ["nopunc",
                           "shuffle", 
                           "shuffle_in_task", 
                           "other_task", 
                           "shuffle_in_sentence",
                           "shuffle_in_sentence_nopunc",
                           "equal_len_sentence",
                           "shuffle_with_punc"] + [f"shuffle_D{d}" for d in range(2000)]
        stim = stimuli.copy()
        n = len(stim)
        if control == "nopunc":
            new = stim.word_low.str.lower().values
        if control == "shuffle":
            new = np.random.choice(all_words_low, n)
        if control == "shuffle_in_task":
            new = np.random.choice(stim["word_low"].dropna().values, n)
        if control == "other_task":
            all_other_tasks = df_tasks.query("audio_task!=@audio_task").audio_task.unique()
            np.random.shuffle(all_other_tasks)
            words = np.concatenate([get_stimulus(task).word_raw.values for task in all_other_tasks])
            new = words[:n]
        if control == "shuffle_in_sentence": # keeps punctuation
            new = stimuli.groupby("sequ_index")["word_raw"].transform(lambda x: np.random.permutation(x)).values
            new = add_punc(new, stimuli.word_raw.values)
            new = add_upper(new, stimuli.is_bos)
        if control == "shuffle_in_sentence_nopunc": # keeps punctuation
            new = stimuli.groupby("sequ_index")["word_low"].transform(lambda x: np.random.permutation(x)).values
        if control == "shuffle_with_punc":
            new = np.random.choice(all_words_low, n)
            new = add_punc(new, stimuli.word_raw.values)
            new = add_upper(new, stimuli.is_bos)
        if "shuffle_D" in control:
            distance = int(control.split("shuffle_D")[1])
            stim["dist_group"] = np.repeat(np.arange(n), distance)[:n]
            new = stim.groupby("dist_group")["word_low"].transform(lambda x: np.random.permutation(x)).values
            new = add_punc(new, stimuli.word_raw.values)
            new = add_upper(new, stimuli.is_bos)
        if control == "equal_len_sentence":
            dico = build_sentence_dic(overwrite=True)
            lens = stim.seq_len.unique()
            print(lens.max())
            new = stim.groupby("sequ_index")["seq_len"].transform(lambda x: np.random.permutation(dico[x.iloc[0]])[0])
            new = new.values

        stim["word_raw"] = new.copy()
        stim["word_raw"] = stim["word_raw"].astype(str)
        control_stim[control] = stim
        
    return control_stim


if __name__=="__main__":
    audio_task = "shapessocial"
    stimuli = get_stimulus(audio_task)
    stimuli["condition"] = "word"
    control_stimulis = add_controls(stimuli, audio_task, control_names=["nopunc",
                            "shuffle", 
                            "shuffle_in_task", 
                            "other_task", 
                            "shuffle_in_sentence", 
                                                        "shuffle_in_sentence_nopunc",
                            "shuffle_with_punc"] + [f"shuffle_D{k}" for k in [5, 10, 30, 100, 1000]])
