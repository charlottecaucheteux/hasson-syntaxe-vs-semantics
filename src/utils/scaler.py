import numpy
import numpy as np
import torch
from scipy.stats import pearsonr


class Scaler:
    """Compatible with numpy and torch"""

    def __init__(self, mode="mean"):
        self.mode = mode
        if mode != "mean":
            m, M = mode
            assert 0 <= m < 1.0
            assert 0 < M <= 1.0
            assert m < M

    def fit(self, X):

        if isinstance(X, torch.Tensor):
            base = torch
        else:
            base = numpy
        if self.mode == "mean":
            mask = base.ones(len(X), dtype=bool)
        else:
            m, M = self.mode
            prctile = base.argsort(X, 0)
            mask = (prctile >= int(len(X) * m)) & (prctile <= int(len(X) * M))
        self.mean_ = X[mask].mean(0)
        self.std_ = X[mask].std(0)
        return self

    def transform(self, X):
        X = X - self.mean_
        X /= self.std_
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = X * self.std_
        X += self.mean_
        return X


def scale(X):
    X = np.asarray(X)
    shape = X.shape
    if len(shape) == 1:
        X = X[:, None]
    X = X - np.nanmean(X, 0, keepdims=True)
    std = np.nanstd(X, 0, keepdims=True)
    non_zero = np.where(std > 0)
    X[:, non_zero] /= std[non_zero]
    X[np.isnan(X)] = 0
    X[~np.isfinite(X)] = 0

    return X.reshape(shape)

