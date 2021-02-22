from pathlib import Path

import numpy as np
from encoding.fmri import convolve_features
from nilearn import signal
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler

from . import paths
from .encode import encode_with_folds
from .get_bold import get_bold
from .get_features import load_precomputed_features
from .preprocess_stim import add_pulses_to_stim, get_stimulus
from .task_dataset import get_task_df


def run_exp_multitasks(
    subject,
    feature_files=[],
    hemi="L",
    space="fsaverage6",
    TR=1.5,
    trim_init=2,
    convolve_model="fir",
    concatenate_tasks=True,
    cross_val_groups=False,
    high_pass=None,
    n_folds=5,
    zero_out=None,
    regress_out=None,
    fit_intercept=False,
    other_task=False,
    average_folds=True,
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
                    events, feat, model=convolve_model, n_delays=n_delays
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

    if not cross_val_groups:
        groups = None

    for lab, feat in X.items():

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

        # Check : scale each task independently or not
        valid = Y.std(0) > 0

        r[lab][valid] = encode_with_folds(
            feat,
            Y[:, valid],
            n_folds=n_folds,
            average_folds=average_folds,
            estimator=RidgeCV(
                np.logspace(-1, 8, 10),
                fit_intercept=fit_intercept,
            ),
            to_regress_out=to_regress_out,
            to_zero_out=to_zero_out,
            groups=groups,
        )
    return r, errors
