from pathlib import Path

# Data to load
base_dir = Path("/private/home/ccaucheteux/narratives")
deriv_dir = base_dir / "derivatives"
afni_dir = base_dir / "derivatives" / "afni-smooth"
afni_dir_nosmooth = base_dir / "derivatives" / "afni-nosmooth"
event_meta_path = base_dir / "code" / "event_meta.json"
task_meta_path = base_dir / "code" / "task_meta.json"
scan_exclude_path = base_dir / "code" / "scan_exclude.json"
checked_gentle_path = base_dir / "stimuli" / "gentle_checked"
gentle_path = base_dir / "stimuli" / "gentle"
probe_path = Path("/private/home/ccaucheteux/structural-probes")
pos_equiv_file = (
    "/private/home/ccaucheteux/drafts/data/english_equivalence_pos_test.npy"
)
posdep_equiv_file = (
    "/private/home/ccaucheteux/drafts/data/english_equivalence_pos_dep_1M.npy"
)
wiki_100m_path = "/checkpoint/ccaucheteux/train-xlm-models/XLM/data/wiki/txt/en.100m"


# Brain map
surf_dir = base_dir / "derivatives/freesurfer/fsaverage6/surf/"
sulc_left = str(surf_dir / "lh.sulc")
sulc_right = str(surf_dir / "rh.sulc")
inf_left = str(surf_dir / "lh.inflated")
inf_right = str(surf_dir / "rh.inflated")

# Repo
root = Path("/private/home/ccaucheteux/hasson-syntaxe-vs-semantics")

# Data to generate
data = root / "data"
phone_dic = data / "phone_dic.npy"
pos_dic = data / "pos_dic.npy"
gpt2_vocab_spacy_feats = data / "spacy_feats" / "gpt2_vocab_spacy_features.npy"

wiki_seq_len_dic = data / "wiki_seq_len_dic.npy"
syn_equiv_dir = data / "syntactic_equivalences"
syn_equiv_file = str(
    data
    / "syntactic_equivalences"
    / "0201_wiki_valid"
    / "%s"
    / "%s"
    / "equival_story.npy"
)
embeddings = data / "embeddings"
mean_bolds = data / "bold" / "mean_bolds_concat_tasks_%s.npy"
median_bolds = data / "bold" / "median_bold_concat_tasks_%s.npy"

# Results
scores = root / "scores"

# Wiki syntactic embeddings (no brain)
wiki_bar_embeddings = data / "wiki_bar_embeddings"
