import time

# %load src/encode
import numpy as np
from numpy.random import permutation
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import make_scorer, pairwise_distances
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

# from torch_ridge import RidgeCV as
# from torch_ridge import RidgeCV as
from torch_ridge import RidgeCV as TorchRidgeCV
from tqdm import tqdm


def v2v(true, pred, metric="cosine"):
    assert len(true) == len(pred)
    if len(true) <= 1:
        acc = 1
    else:
        ns = len(true)
        first = permutation(ns)  # first group of TR
        second = permutation(ns)  # second group of TR
        i = 0
        while (first == second).any() and i < 10:  # check that distinct TRs in pairs
            print("invalid", len(first))
            first[first == second] = np.random.choice((first == second).sum())
            i += 1

        correct = 0
        for i, j in zip(first, second):
            r = pairwise_distances(
                true[[i, j]], pred[[i, j]], metric
            )  # compute the 4 distances
            diag = np.diag(r).sum()  # distances of corresponding TR
            cross = r.sum() - diag  # distance of cross TR
            correct += 1 * (diag < cross)  # comparison
        acc = correct / ns
    return np.array(acc)[None]


def jr_2v2(true, pred, metric="cosine"):
    """Tsonova et al 2019 https://arxiv.org/pdf/2009.08424.pdf"""
    assert len(true) == len(pred)
    ns = len(true)
    first = permutation(ns)  # first group of TR
    second = permutation(ns)  # second group of TR
    while (first == second).any():  # check that distinct TRs in pairs
        first[first == second] = np.random.choice((first == second).sum())

    r = pairwise_distances(true, pred)
    s1 = r[first, first] + r[second, second]
    s2 = r[first, second] + r[second, first]

    acc = np.mean(1.0 * (s1 < s2))
    return acc[None]


def v2v_per_voxel(true, pred):
    assert len(true) == len(pred)
    if len(true) <= 2:
        print("invalid")
        return np.ones(true.shape[-1])
    ns = len(true)
    first = permutation(ns)
    second = permutation(ns)
    i = 0
    while (first == second).any() and i < 10:
        print("invalid", len(first))
        first[first == second] = np.random.choice((first == second).sum())
        i += 1

    correct = np.zeros(true.shape[-1])
    for i, j in zip(first, second):
        r = np.abs(true[[i, j]][None] - pred[[i, j]][:, None])
        diag = r[0, 0] + r[1, 1]
        correct += 1 * (diag < r.sum((0, 1)) - diag)
    acc = correct / ns
    return acc


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


def t_correlate(X, Y):
    return correlate(X.T, Y.T)


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
        corr = correlate(pred, Y[test])
        R.append(corr)

        end = time.time()
        print(f"time : {(end-start):.2f}, {corr:2f}")
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
        R.append(corr_function(pred, Y[test]))

        end = time.time()
        print(f"{(end-start):.2f}")
        start = end
    R = np.stack(R)
    if average_folds:
        return R.mean(0)
    return R.T


def encode_with_permutation_imp(
    X,
    bold,
    corr_function=correlate,
    average_folds=True,
    estimator=RidgeCV(np.logspace(-1, 8, 20)),
    cv=KFold(5, shuffle=False),
    groups=None,
):
    model = clone(estimator)
    R = []
    start = time.time()

    for train, test in tqdm(cv.split(X, groups=groups)):

        Y = bold.copy()
        r = np.zeros((X.shape[1], Y.shape[1]))
        model.fit(X[train], Y[train])
        X_test = X[test].copy()

        for vox in range(Y.shape[1]):
            r[:, vox] = permutation_importance(
                model,
                X_test,
                Y[test, vox],
                n_repeats=10,
                random_state=0,
                scoring=make_scorer(lambda x, y: corr_function(x, y).mean()),
            ).importances_mean
        R.append(r)

    R = np.stack(R)
    if average_folds:
        return R.mean(0)
    return R.T


def encode_with_betas(
    X,
    bold,
    estimator=LinearRegression(),
    cv=None,
):
    model = clone(estimator)
    X = StandardScaler().fit_transform(X)

    if cv is not None:
        Y = bold.copy()
        R = np.zeros(Y.shape[1])
        betas = np.zeros((Y.shape[1], X.shape[1]))
        for train, test in cv.split(X):

            model.fit(X[train], Y[train])
            pred = model.predict(X[test])

            # score
            R += correlate(pred, Y[test])

            # coeff
            betas += model.coef_

        R /= cv.get_n_splits()
        betas /= cv.get_n_splits()

        return np.concatenate([betas, R[:, None]], axis=1)

    else:

        model.fit(X, bold)

    return model.coef_
