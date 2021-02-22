import os
import pathlib

import mne
import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
from nilearn import surface
from nilearn.image import clean_img
from nistats.hemodynamic_models import compute_regressor

from .paths import confounds_fname, deriv_path, events_fname, preproc_path
# from .modeling import get_embeddings, get_model, scale
from .scaler import scale


def read_events(subject):
    # Read MRI events
    events = pd.read_csv(events_fname % (subject, subject), sep="\t")

    # Add context: sentence or word list?
    contexts = dict(WOORDEN="word_list", ZINNEN="sentence")
    for key, value in contexts.items():
        sel = events.value.str.contains(key)
        events.loc[sel, "context"] = value
        events.loc[sel, "condition"] = value

    # Clean up MRI evenst mess
    sel = ~events.context.isna()
    start = 0
    context = "init"
    for idx, row in events.loc[sel].iterrows():
        events.loc[start:idx, "context"] = context
        start = idx
        context = row.context
    events.loc[start:, "context"] = context

    # Add event condition: word, blank, inter stimulus interval etc
    conditions = (("50", "pulse"), ("blank", "blank"), ("ISI", "isi"))
    for key, value in conditions:
        sel = events.value == key
        events.loc[sel, "condition"] = value

    events.loc[events.value.str.contains("FIX "), "condition"] = "fix"

    # Extract words from file
    sel = events.condition.isna()
    words = events.loc[sel, "value"].apply(lambda s: s.strip("0123456789 "))
    events.loc[sel, "word"] = words

    # Remove empty words
    sel = (events.word.astype(str).apply(len) == 0) & (events.condition.isna())
    events.loc[sel, "word"] = np.nan
    events.loc[sel, "condition"] = "blank"
    events.loc[~events.word.isna(), "condition"] = "word"

    # Define sequence
    events.loc[events.word == "QUESTION", "condition"] = "question"
    events.loc[events.word == "QUESTION", "word"] = np.nan
    events["sequence"] = np.cumsum(events.condition == "fix")
    events[
        "sequ_pos"
    ] = events.sequence.values  # for backward compatibility with word embeddings

    for s, words in events.query('condition=="word"').groupby("sequence"):
        events.loc[words.index, "word_position"] = range(len(words))

    # Fix bids
    events["trial_type"] = events["type"]

    # volume start at 1!
    events["volume"] -= 1.0

    return events


def compute_fir(events, features, frame_times, merge_func="sum", delays=4):
    assert len(events) == len(features)
    events = events.copy().reset_index()
    events["volume_int"] = events.volume.apply(int)

    # Find all events for each frame time (fMRI TR)
    indices = [[]] * len(frame_times)
    for v, e in events.groupby("volume_int"):
        indices[v] = e.index

    # Concatenate feature within each TR
    if merge_func == "sum":
        merge_func = lambda x: x.sum(0)
    elif merge_func == "mean":
        merge_func = lambda x: x.mean(0)

    X = np.zeros((len(frame_times), features.shape[1]))
    for j, idx in enumerate(indices):
        if not len(idx):
            continue
        X[j, :] = merge_func(features[idx])

    # Build FIR
    X = np.concatenate([np.roll(X, k, axis=0) for k in range(delays)], axis=-1)

    return X


def clean_fmri(
    img,
    confounds,
    tr,
    subject,
    detrend=True,
    high_pass=None,
    crop=None,
    standardize=False,
    confounds_fname=confounds_fname,
):

    if crop:
        start, stop = crop
        data = img.get_fdata()
        img_shape = data.shape
        data = data[..., start:stop]
        img = nib.Nifti1Image(data, img.affine, img.header)

    if confounds:

        if confounds == "all":
            confounds = ["comp", "cos", "pos"]
        if isinstance(confounds, str):
            confounds = [confounds]

        selected_confounds = list()
        for c in confounds:
            if c == "comp":

                # https://neurostars.org/t/confounds-from-fmriprep-which-one-would-you-use-for-glm/326/32
                selected_confounds += [
                    "a_comp_cor_00",
                    "a_comp_cor_01",
                    "a_comp_cor_02",
                    "a_comp_cor_03",
                    "a_comp_cor_04",
                    "a_comp_cor_05",
                ]
            elif c == "cos":
                selected_confounds += [
                    "cosine00",
                    "cosine01",
                    "cosine02",
                    "cosine03",
                    "cosine04",
                    "cosine05",
                ]
            elif c == "pos":
                selected_confounds += [
                    "trans_x",
                    "trans_y",
                    "trans_z",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                ]

        confounds = pd.read_csv(confounds_fname % (subject, subject), sep="\t")
        confounds = confounds[selected_confounds].to_numpy()

        if crop:
            confounds = confounds[start:stop]

    if confounds is not None or detrend or high_pass or standardize:

        img = clean_img(
            img,
            sessions=None,
            detrend=detrend,
            standardize=standardize,
            confounds=confounds,
            low_pass=None,
            high_pass=high_pass,
            t_r=tr,
            ensure_finite=False,
        )

    if crop:
        data = img.get_fdata()
        x, y, z, t = data.shape
        data = np.concatenate(
            [
                np.zeros((x, y, z, start), dtype=data.dtype),
                data,
                np.zeros((x, y, z, -stop), dtype=data.dtype),
            ],
            axis=3,
        )
        img = nib.Nifti1Image(data, img.affine, img.header)
        assert np.array_equal(data.shape, img_shape)

    return img


def vol_to_surf(
    img, subject, deriv_path=deriv_path, tr=2.0, decim=10, subject_to="fsaverage5"
):
    from nilearn.surface.surface import _sample_locations

    # get function data
    data = img.get_fdata()

    # Sample line normal to from pial/white
    bold = list()
    vertices = list()
    for hemi in ("left", "right"):
        print(hemi, end=" ")
        # Read pial mid surface
        mesh_pial = "%s_hemi-%s_midthickness.surf.gii" % (subject, hemi[0].upper())
        mesh_pial = str(deriv_path / subject / "anat" / mesh_pial)
        mesh = surface.load_surf_mesh(mesh_pial)

        # Sample normal to surface
        sample_locations = _sample_locations(
            mesh, img.affine, radius=3.0, kind="line", n_points=None
        )
        sample_locations = sample_locations.astype(int)

        # Make vertices for later morph to fsaverage
        vertices.append(np.arange(0, len(sample_locations), decim))

        # Decimate to avoid redundancy
        sample_locations = sample_locations[::decim]

        # Average project onto surface
        x, y, z = sample_locations.T
        _bold = data[x, y, z, :].mean(0)
        bold.append(scale(_bold.T).T)
    bold = np.vstack(bold)

    # Build morph to fsaverage
    print("fit morph to fsaverage")
    stc = mne.SourceEstimate(bold, vertices, subject=subject, tmin=0, tstep=1 / tr)

    morph = mne.compute_source_morph(
        stc,
        subject_from=subject,
        subject_to=subject_to,
        subjects_dir=str(deriv_path / ".." / "freesurfer"),
    )
    return stc, morph


def preproc_fmri(
    subject,
    confounds=None,
    deriv_path=deriv_path,
    preproc_path=preproc_path,
    detrend=True,
    high_pass=None,
    crop=None,
    overwrite=False,
    standardize=False,
    hemi="both",
):

    # Analyze in T1 space within subjects
    space = "T1w"
    func = "-preproc_bold.nii.gz"
    assert hemi in ["both", "rh", "lh"]

    if hemi == "both":
        ext = ""
    else:
        ext = f"-{hemi}.stc"

    stc_fname = str(pathlib.Path(preproc_path) / (subject + "-stc.fif" + ext))
    # stc_fname = str(pathlib.Path(preproc_path) / (subject + "-stc.fif"))
    morph_fname = str(pathlib.Path(preproc_path) / (subject + "-morph.h5"))
    if os.path.isfile(morph_fname) and not overwrite:
        print("load preprocessed bold and morph %s" % morph_fname)

        stc = mne.read_source_estimate(stc_fname)
        morph = mne.read_source_morph(morph_fname)
        return stc, morph

    # Preproc Bold
    files = [
        f
        for f in os.listdir(deriv_path / subject / "func")
        if f.endswith(func) and "_task-visual_space-%s" % space in f
    ]
    assert len(files) == 1
    print("load fMRI %s" % subject)
    img = nib.load(str(deriv_path / subject / "func" / files[0]))
    tr = 2.0

    # clean confounds, filter, detrend
    print("clean fMRI %s" % subject)
    img = clean_fmri(
        img, confounds, tr, subject, detrend, high_pass, crop, standardize,
    )

    # project to surface
    print("project fMRI to surface %s" % subject)
    stc, morph = vol_to_surf(img, subject, deriv_path, tr, decim=10)

    # save
    print("save preprocessed bold and morph %s" % subject)
    stc.save(stc_fname)
    morph.save(morph_fname, overwrite=True)

    return stc, morph


def detrend_segment(bold, events, segment="sequence", extend=1, center=True):
    from scipy.signal import detrend

    # find sequence starts and stops
    words = events.query('condition=="word"').copy()
    if segment == "sequence":

        for s, ev in words.groupby("sequence"):
            words.loc[ev.index, "sequ_length"] = len(ev)

        starts = words.query("word_position==0")
        stops = words.query("(word_position+1)==sequ_length")
        assert len(stops) == len(starts) == len(words.sequence.unique())
    else:
        words["block"] = np.cumsum(words.shift(1).context != words.context) - 1
        for s, ev in words.groupby("block"):
            words.loc[ev.index, "block_length"] = len(ev)
            words.loc[ev.index, "block_position"] = range(len(ev))

        starts = words.query("block_position==0")
        stops = words.query("(block_position+1)==block_length")
        assert len(stops) == len(starts) == len(words.block.unique())

    starts = starts.volume.values.astype(int)
    stops = stops.volume.values.astype(int)

    # Check

    assert all(starts < stops)
    assert all((starts[1:] - stops[:-1]) >= (2 * extend))

    # Detrend each sequence individually
    bold_seq = np.nan * np.zeros_like(bold)
    for start, stop in zip(starts, stops):
        sel = slice(start - extend, stop + extend)
        bold_seq[sel] = detrend(bold[sel], axis=0)

    # Center
    if center:
        bold_seq -= np.nanmean(bold_seq, axis=0)
        bold_seq = np.nan_to_num(bold_seq)
    return bold_seq


def convolve_features(
    events, features, model="fir", pca=False, feature_pca=None, n_delays=4,
):
    assert model in ["fir", "glover", "glover + derivative"]
    # Get fMRI frame times
    frame_times = events.query('type=="Pulse"').onset

    # Define potential causal factors
    word_events = events.query('condition=="word"').copy()

    assert len(word_events) == len(features)

    if model == "fir":
        reg_signals = compute_fir(word_events, features, frame_times, delays=n_delays)

    else:
        reg_signals = list()
        for column, v in enumerate(features.T):
            signal, name_ = compute_regressor(
                np.c_[word_events.onset, np.ones(len(word_events)), v].T,
                hrf_model=model,
                frame_times=frame_times.values,
                oversampling=16,
            )
            reg_signals.append(signal)
        reg_signals = np.concatenate(reg_signals, 1)

    return reg_signals


def restrict_indices(events, restrict, indices, extend=3):
    if restrict is None:
        return indices
    volumes = events.query(restrict)

    # we keep only indices for which there is a file on
    volumes = volumes.loc[events.sequence != 0].volume.astype(int).values

    # we extend it a bit to take delay into account
    volumes = np.unique([volumes + i for i in range(extend)])
    return np.intersect1d(volumes, indices)
