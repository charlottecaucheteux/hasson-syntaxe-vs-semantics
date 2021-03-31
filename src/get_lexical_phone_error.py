from pathlib import Path

import cmudict
import numpy as np
from tqdm import tqdm
from wordfreq import word_frequency, zipf_frequency

from . import paths


def get_lexical_phone_table(
    save_file=paths.data / "lexical_phone_table.npy", overwrite=False
):  # ~1h

    if Path(save_file).is_file() and not overwrite:
        phone_prob = np.load(save_file, allow_pickle=True).item()
        return phone_prob["prob_table"], phone_prob["words"]

    # Get (word: phones) pairs with CMU
    phonedic = dict(cmudict.entries())
    words = list(phonedic.keys())  # = cmudict.words()

    # Corresponding freq of words
    freqtable = np.array([word_frequency(w, "en") for w in words])

    # Init phone prob tabe
    maxlen = max([len(i) for _, i in phonedic.items()])
    phonetable = np.zeros((len(phonedic), maxlen), dtype=int)
    symbols2id = {lab: i for i, lab in enumerate(cmudict.symbols())}

    # Fill phone table for each word of cmudic
    for i, (k, v) in enumerate(phonedic.items()):
        for j, symb in enumerate(v):
            phonetable[i, j] = symbols2id[symb]
    assert (phonetable[:, 0]).all()  # at least one phoneme

    # Compute, for each word of CMU, the prob of its phonemes
    phone_prob = np.zeros(phonetable.shape)
    for w, word_phones in enumerate(
        tqdm(phonetable)
    ):  # One line per word # Could be done by batch with broadcasting

        # Words with similar first, second, third ... phonemes as the current word
        equal = word_phones[:, None] == phonetable.T
        equal = equal.cumprod(0)
        weighted = equal * freqtable[None]

        # C_phi_t=A
        cumulative = weighted.sum(1)

        # C_phi_t=A / C_phi_t-1
        cumulative[1:] /= cumulative[:-1]
        cumulative = np.nan_to_num(cumulative, nan=1)

        phone_prob[w] = cumulative

    Path(save_file).parent.mkdir(exist_ok=True, parents=True)
    np.save(save_file, {"prob_table": phone_prob, "words": words})
    return phone_prob, words


def get_wordfreq(stimulus, log=True):
    tokens_low = stimulus.word.str.replace("’", "'").str.lower().values
    freq = [zipf_frequency(w) for w in tokens_low]
    return np.array(freq)


def get_phon_freq_dic(agg="mean"):  # mean phoneme frequency per word

    # Get (word: phones) pairs with CMU
    phonedic = dict(cmudict.entries())
    words = list(phonedic.keys())  # = cmudict.words()

    # Corresponding freq of words
    freqtable = np.array([word_frequency(w, "en") for w in words])

    phonetable = np.zeros((len(phonedic), len(cmudict.symbols())), dtype=int)
    symbols2id = {lab: i for i, lab in enumerate(cmudict.symbols())}

    # Fill phone table for each word of cmudic
    for i, (k, v) in enumerate(phonedic.items()):
        for j, symb in enumerate(v):
            phonetable[i, symbols2id[symb]] += 1

    res = phonetable * freqtable[:, None]
    res = res.sum(0)
    res /= res.sum()
    # res = np.log10(res * 1e9)
    res = (np.nan_to_num(res[None]) * phonetable).sum(1)
    if agg == "mean":
        res /= phonetable.sum(1)
    res = {w: res[i] for i, w in enumerate(words)}
    return res


def get_phone_freq(stimulus, agg="mean"):
    phon_freq_dic = get_phon_freq_dic()
    tokens_low = stimulus.word.str.replace("’", "'").str.lower().values
    phon_freq = np.zeros((len(stimulus)))
    for i, tok in enumerate(tqdm(tokens_low)):
        if tok in phon_freq_dic:
            phon_freq[i] = phon_freq_dic[tok]
        else:
            phon_freq[i] = 0

    return phon_freq


def get_lexical_phone_error(stimulus, log=True):
    tokens_low = stimulus.word.str.replace("’", "'").str.lower().values

    phone_prob, words = get_lexical_phone_table()
    phone_prob = np.concatenate(
        [phone_prob, np.ones(phone_prob.shape[1])[None]], axis=0
    )

    idx = []
    for tok in tqdm(tokens_low):
        if tok in words:
            idx.append(words.index(tok))
        else:
            idx.append(-1)
    idx = np.array(idx)

    error = phone_prob[idx]
    error = phone_prob[idx].prod(-1)
    if log:
        error[error == 0] = error[error > 0].min()
        error = np.log10(error * 1e9)
        # error = -np.log(error)

    return error
