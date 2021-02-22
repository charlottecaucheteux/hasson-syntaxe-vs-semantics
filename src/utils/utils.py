import numpy as np
from scipy.stats import pearsonr


def r_scorer(y_pred, y_true):
    n, nc, nt = y_pred.shape
    r = np.zeros((nc, nt))
    for i in range(nc):
        for j in range(nt):
            true = y_true[:, i, j]
            pred = y_pred[:, i, j]
            r[i, j] = pearsonr(true, pred)[0]
    return r


def r_metric(y_pred, y_true):
    _, nc = y_pred.shape
    r = np.zeros(y_pred.shape[1])
    for i in range(nc):
        true = y_true[:, i]
        pred = y_pred[:, i]
        r[i] = pearsonr(true, pred)[0]
    return r


def shuffle_c(x):
    if x.shape[1] == 0:
        return x
    else:
        cols = [np.random.permutation(col) for col in x.T]
        cols = np.stack(cols).T
        return cols


def correlate(X, Y):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    SX2 = (X ** 2).sum(0) ** 0.5
    SY2 = (Y ** 2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    return SXY / (SX2 * SY2)
