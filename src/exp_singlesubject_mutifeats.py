from pathlib import Path

import numpy as np
from nilearn import signal
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from . import paths
from .encode import encode_with_betas, encode_with_folds, encode_with_permutation_imp
from .fir import convolve_features
from .get_bold import get_bold
from .get_features import load_precomputed_features
from .preprocess_stim import add_pulses_to_stim, get_stimulus
from .task_dataset import get_task_df


def encoding_multifeats(X, Y, encoding_type="base", groups=None):
    assert encoding_type in ["base", "betas", "perm_imp", "cvbetas"]
    valid = Y.std(0) > 0
    nv = Y.shape[1]
    nf = X.shape[1]
    if encoding_type == "base":
        r = np.zeros(nv)
        r[valid] = encode_with_folds(
            X,
            Y[:, valid],
            n_folds=5,
            average_folds=True,
            estimator=RidgeCV(
                np.logspace(-1, 8, 10),
                fit_intercept=True,
            ),
            to_regress_out=None,
            to_zero_out=None,
            groups=groups,
        )
    elif encoding_type == "betas":
        r = np.zeros((nv, nf))
        r[valid] = encode_with_betas(
            X,
            Y[:, valid],
            estimator=LinearRegression(),
        )

    elif encoding_type == "cvbetas":
        r = np.zeros((nv, nf + 1))  # not clean, in order to store R
        r[valid] = encode_with_betas(
            X,
            Y[:, valid],
            cv=KFold(5),
            estimator=LinearRegression(),
        )

    elif encoding_type == "perm_imp":
        r = np.zeros((nv, nf))
        r[valid] = encode_with_permutation_imp(
            X,
            Y[:, valid],
            average_folds=True,
            estimator=LinearRegression(),
        )

    return r


def run_exp_singlesub_multifeats(
    subject,
    feature_files=[],
    encoding_type="base",
    hemi="L",
    space="fsaverage6",
    TR=1.5,
    trim_init=2,
    convolve_model="fir",
    merge_func="sum",
    concatenate_tasks=True,
    high_pass=None,
    n_folds=5,
    other_task=False,
    n_delays=4,
    scaler=RobustScaler(quantile_range=(0.1, 99.9)),
):
    df_task = get_task_df()
    df_task = df_task.query("subject==@subject")

    r = {}

    # ---------- Build X and Y ----------
    errors = []
    X = {lab: [] for lab in feature_files}
    Y = []
    for _, row in df_task.iterrows():

        audio_onset = row.onset
        audio_task = row.audio_task

        try:

            bold_task = row.bold_task
            print(bold_task)

            print(f"Processing task {audio_task}")

            # Load bold responses
            gii_fname = f"{subject}_task-{bold_task}_*space-{space}_hemi-{hemi}_desc-clean.func.gii"
            subj_data = get_bold(
                gii_fname, subject, exclude=True, afni_dir=paths.afni_dir_nosmooth
            )
            if subj_data is None:
                continue

            # Trim bold w.r.t onset timing of the audio file => everything starts at 0 from now on
            subj_data = subj_data[audio_onset:, :]

            # Load stimulus
            stimuli = get_stimulus(audio_task)

            # Merge pulses and stimulus
            n_pulse = len(subj_data)
            events = add_pulses_to_stim(stimuli, n_pulse, TR=TR)

            # Cut extra pulses
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

            # Extract features from stimulus
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

            # Preprocess feats
            for lab, feat in zip(feature_files, features):
                feat = np.array(feat)
                feat = scaler.fit_transform(feat)
                convolved = convolve_features(
                    events,
                    feat,
                    model=convolve_model,
                    n_delays=n_delays,
                    merge_func=merge_func,
                )
                convolved = scaler.fit_transform(convolved)
                X[lab].append(convolved)

        except Exception as e:
            print(f"ERROR for task {audio_task}, {e}")
            errors.append((audio_task, e))

    # Check only one task
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

    # ---------- ENCODE ----------

    for lab, feat in X.items():
        r[lab] = encoding_multifeats(feat, Y, encoding_type=encoding_type)

    return r, errors
