import time

# %load src/encode
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold

# from torch_ridge import RidgeCV as
# from torch_ridge import RidgeCV as
from torch_ridge import RidgeCV as TorchRidgeCV


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


def encode_with_folds_old(
    X,
    bold,
    corr_function=correlate,
    events=None,
    restrict=None,
    test_restrict=None,
    independent_alphas=False,
    estimator=RidgeCV(np.logspace(-1, 8, 20)),
    n_folds=5,
    average_folds=True,
    groups=None,
    to_regress_out=None,
    to_zero_out=None,
):
    model = clone(estimator)

    if groups is None:
        cv = KFold(n_folds, shuffle=False)
    else:
        n_folds = np.min([n_folds, len(np.unique(groups))])
        cv = GroupKFold(n_folds)

    R = []
    start = time.time()
    for train, test in cv.split(X, groups=groups):
        print(".", end="")

        if to_regress_out is not None:
            idx = np.where(to_regress_out)[0]
            model.fit(X[train][:, idx], bold[train])
            Y = bold.copy()
            Y -= model.predict(X[:, idx])
            print(f"REGRESSING OUT {len(idx)} of Y")
        else:
            Y = bold.copy()

        model.fit(X[train], Y[train])
        X_test = X[test].copy()
        if to_zero_out is not None:
            assert len(to_zero_out) == X_test.shape[1]
            idx = np.where(to_zero_out)[0]
            X_test[:, idx] = 0
            print(f"ZEROING OUT {len(idx)} of X", X_test[:, idx].sum())
        pred = model.predict(X_test)
        R.append(correlate(pred, Y[test]))

        end = time.time()
        print(f"{(end-start):.2f}")
        start = end
    R = np.stack(R)
    if average_folds:
        return R.mean(0)
    return R.T


def encode_with_folds(
    X,
    bold,
    corr_function=correlate,
    events=None,
    restrict=None,
    test_restrict=None,
    use_torch=False,
    independent_alphas=False,
    estimator=RidgeCV(np.logspace(-1, 8, 20)),
    n_folds=5,
    average_folds=True,
    groups=None,
    to_regress_out=None,
    to_zero_out=None,
):
    assert use_torch == False
    model = clone(estimator)

    if groups is None:
        cv = KFold(n_folds, shuffle=False)
    else:
        n_folds = np.min([n_folds, len(np.unique(groups))])
        cv = GroupKFold(n_folds)

    R = []
    start = time.time()
    for train, test in cv.split(X, groups=groups):
        print(".", end="")

        if to_regress_out is not None:
            idx = np.where(to_regress_out)[0]
            model.fit(X[train][:, idx], bold[train])
            Y = bold.copy()
            Y -= model.predict(X[:, idx])
            print(f"REGRESSING OUT {len(idx)} of Y")

            idx = np.where(~to_regress_out)[0]
            model.fit(X[train][:, idx], Y[train])
            X_test = X[test][:, idx].copy()

        else:
            Y = bold.copy()
            model.fit(X[train], Y[train])
            X_test = X[test].copy()

        if to_zero_out is not None:
            assert len(to_zero_out) == X_test.shape[1]
            idx = np.where(to_zero_out)[0]
            X_test[:, idx] = 0
            print(f"ZEROING OUT {len(idx)} of X", X_test[:, idx].sum())

        pred = model.predict(X_test)
        R.append(correlate(pred, Y[test]))

        end = time.time()
        print(f"{(end-start):.2f}")
        start = end
    R = np.stack(R)
    if average_folds:
        return R.mean(0)
    return R.T
