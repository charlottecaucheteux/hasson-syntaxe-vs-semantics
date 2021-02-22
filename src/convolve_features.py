import numpy as np
from nistats.hemodynamic_models import compute_regressor


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


def convolve_features(
    events,
    features,
    model="fir",
    pca=False,
    feature_pca=None,
    n_delays=4,
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
