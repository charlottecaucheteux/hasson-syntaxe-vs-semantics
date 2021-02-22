REGEX_GENTLE_TOKENIZER = r"(\w|\’\w|\'\w)+"

POSSIBLE_FEATURES = [
    "gpt2",
    "wordpos",
    "seqlen",
    "bert",
    "bert-large",
    "gpt2.shuffled",
    "bert.shuffled",
    "bert-large.shuffled",
    "bert.shuffled-posdep",
    "bert-large.shuffled-posdep",
    "gpt2.shuffled-posdep",
    "manning",
    "manning.shuffled",
    "manning.shuffled-posdep",
    "bert-large-ptpb",
    "phones",
    "n_words",
    "n_phones",
]


TRANSFORMER_NAMES = {
    "gpt2": "gpt2",
    "bert": "bert-base-cased",
    "bert-large": "bert-large-cased",
}
