from pathlib import Path

import numpy as np
from nilearn import signal
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler

from . import paths
from .encode import encode_with_folds
from .fir import convolve_features
from .get_bold import get_bold
from .get_features import load_precomputed_features
from .preprocess_stim import add_pulses_to_stim, get_stimulus
from .task_dataset import get_task_df


def run_exp_single_tasks(
    subject,
    audio_task,
    feature_files=[],  # ["wordpos", "seqlen", "gpt2.0"],
    hemis=["L", "R"],
    smooth=False,
    space="fsaverage6",
    TR=1.5,
    trim_init=2,
    use_torch=False,
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

    df_task = get_task_df(keep_milk=True)
    df_task = df_task.query("subject==@subject and audio_task==@audio_task")
    print(df_task)
    row = df_task.iloc[0]
    audio_onset = row.onset
    audio_task = row.audio_task
    bold_task = row.bold_task

    r = {}
    features = None
    for hemi in hemis:
        # Load bold responses
        gii_fname = (
            f"{subject}_task-{bold_task}_*space-{space}_hemi-{hemi}_desc-clean.func.gii"
        )
        if smooth:
            afni_dir = paths.afni_dir
        else:
            afni_dir = paths.afni_dir_nosmooth
        subj_data = get_bold(gii_fname, subject, exclude=True, afni_dir=afni_dir)
        if subj_data is None:
            return

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
        events["volume"] -= onset
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

        r[hemi] = {}
        for lab, feat in zip(feature_files, features):
            feat = scaler.fit_transform(feat)
            convolved = convolve_features(
                events, feat, model=convolve_model, n_delays=n_delays
            )
            convolved = scaler.fit_transform(convolved)

            # Encode
            bold = subj_data.copy()
            valid = bold.std(0) > 0
            bold[:, valid] = scaler.fit_transform(bold[:, valid])

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

            if average_folds:
                r[hemi][lab] = np.zeros(bold.shape[1])
            else:
                r[hemi][lab] = np.zeros((bold.shape[1], n_folds))
            r[hemi][lab][valid] = encode_with_folds(
                convolved,
                bold[:, valid],
                n_folds=n_folds,
                average_folds=average_folds,
                estimator=RidgeCV(
                    np.logspace(-1, 8, 10),
                    fit_intercept=fit_intercept,
                ),
                to_regress_out=to_regress_out,
                to_zero_out=to_zero_out,
                groups=None,
            )

    return r
