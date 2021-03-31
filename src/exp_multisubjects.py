from pathlib import Path

import numpy as np
import torch
from nilearn import signal
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch_ridge import RidgeCV as TorchRidgeCV

from . import paths
from .encode import (
    correlate,
    encode_with_folds,
    jr_2v2,
    t_correlate,
    v2v,
    v2v_per_voxel,
)
from .fir import convolve_features
from .get_bold import get_bold
from .get_features import get_features, load_precomputed_features
from .preprocess_stim import (
    add_pulses_to_stim,
    get_stimulus,
    get_timing,
    trim_pulses_and_stim,
)


def run_exp_multisubjects(
    average_bold,
    feature_files=[],
    TR=1.5,
    trim_init=2,
    convolve_model="fir",
    concatenate_tasks=True,
    cross_val_groups=False,
    fit_intercept=True,
    average_folds=False,
    n_folds=5,
    high_pass=None,
    other_task=False,
    zero_out=None,
    regress_out=None,
    n_delays=4,
    metric="correlate",
    alpha_per_target=False,
    scaler=RobustScaler(quantile_range=(0.1, 99.9)),
):
    assert metric in ["correlate", "t_correlate", "v2v", "v2v_per_voxel", "jr_2v2"]
    r = {}
    X = {lab: [] for lab in feature_files}
    Y = []
    tasks = []
    for audio_task, subj_data in average_bold.items():

        if subj_data is None:
            continue

        # ----  Load stimulus ----
        if other_task:
            tasks = list(average_bold.keys())
            tasks = [t for t in tasks if t != audio_task]
            audio_task = str(np.random.permutation(tasks)[0])

        stimuli = get_stimulus(audio_task)

        # ---- Merge pulses and stimulus ----
        n_pulse = len(subj_data)
        events = add_pulses_to_stim(stimuli, n_pulse, TR=TR)

        # ---- Cut extra pulses -----
        onset = np.floor(stimuli.dropna(subset=["onset"]).iloc[0].onset / TR)
        onset = int(np.max([trim_init, onset]))
        offset = np.ceil(stimuli.dropna(subset=["offset"]).iloc[-1].offset / TR)
        offset = int(offset)

        subj_data = subj_data[onset:offset]
        events = events.query("volume<@offset and volume>=@onset")
        events["volume"] = events["volume"] - onset
        assert len(subj_data) == len(events.query("condition=='Pulse'"))

        word_events = events.query("condition == 'word'")
        word_idx = word_events.word_index.values

        # ---- Get features ----
        # Load
        feature_task_files = [Path(str(f) % str(audio_task)) for f in feature_files]
        assert np.all(
            [f.is_file() for f in feature_task_files]
        ), f"!!!!!! NOT EXIXTS : {feature_task_files}"
        features = load_precomputed_features(
            audio_task, feature_task_files, idx=word_idx
        )
        assert np.all([len(feat) == len(word_events) for feat in features])

        if high_pass is not None:
            subj_data = signal.clean(
                subj_data,
                sessions=None,
                detrend=False,
                standardize=False,
                confounds=None,
                low_pass=None,
                high_pass=high_pass,
                t_r=1.5,
                ensure_finite=False,
            )

        valid = subj_data.std(0) > 0
        subj_data[:, valid] = scaler.fit_transform(subj_data[:, valid])
        Y.append(subj_data.copy())

        # Preprocess
        for lab, feat in zip(feature_files, features):
            feat = StandardScaler().fit_transform(feat)
            convolved = convolve_features(
                events, feat, model=convolve_model, n_delays=n_delays
            )
            convolved = StandardScaler().fit_transform(convolved)
            X[lab].append(convolved)

    # Check at least one task
    task_len = [len(i) for i in Y]
    if len(task_len) == 0:
        print("No task left")
        return

    print("Concatenating all the tasks")

    groups = np.repeat(np.arange(len(task_len)), task_len)
    X = {lab: np.concatenate(Xi, axis=0) for lab, Xi in X.items()}
    Y = np.concatenate(Y, axis=0)

    assert np.all([len(v) == len(Y) == len(groups) for _, v in X.items()])
    print(len(Y))

    if not cross_val_groups:
        groups = None

    for lab, feat in X.items():

        # Check : scale each task independently or not
        valid = Y.std(0) > 0
        # Y[:, valid] = scaler.fit_transform(Y[:, valid])

        if average_folds:
            r[lab] = np.zeros(Y.shape[1])
        else:
            r[lab] = np.zeros((Y.shape[1], n_folds))

        # CONFOUNDS
        if zero_out is not None and zero_out[lab] is not None:
            to_zero_out = list(zero_out[lab]) * n_delays
            to_zero_out = np.array(to_zero_out)
        else:
            to_zero_out = None

        if regress_out is not None and regress_out[lab] is not None:
            to_regress_out = list(regress_out[lab]) * n_delays
            to_regress_out = np.array(to_regress_out)
        else:
            to_regress_out = None

        if metric == "correlate":
            corr_function = correlate
        if metric == "t_correlate":
            corr_function = t_correlate
        if metric == "v2v":
            corr_function = v2v
        if metric == "v2v_per_voxel":
            corr_function = v2v_per_voxel
        if metric == "jr_2v2":
            corr_function = jr_2v2

        r[lab][valid] = encode_with_folds(
            feat,
            Y[:, valid],
            average_folds=average_folds,
            n_folds=n_folds,
            to_zero_out=to_zero_out,
            to_regress_out=to_regress_out,
            corr_function=corr_function,
            estimator=RidgeCV(
                np.logspace(-1, 8, 10),
                fit_intercept=fit_intercept,
                alpha_per_target=alpha_per_target,
            ),
            groups=groups,
        )
    return r
