import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

BLOCK = 1000
K = 5
MAX_TRAIN = 400_000
SEED = 42


def load_dataset(path="outputs/recovery_dataset.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def condition_matrix(d):
    """Koşullar: yangın öncesi NBR, yükseklik, eğim, kuzey-güney eğim, dNBR."""
    return np.column_stack([
        d["baseline_vigor"], d["elevation"], d["slope_deg"], d["ns_gradient"], d["dnbr_value"],
    ]).astype("float32")


class SpatialFolds:
    def __init__(self, x, y, block=BLOCK, k=K, seed=SEED):
        self.k = k
        self.block_id = (np.floor(x / block).astype(np.int64) * 10000
                         + np.floor(y / block).astype(np.int64))
        self.ub = np.unique(self.block_id)
        rng = np.random.default_rng(seed)
        self.perm_fold = rng.permutation(len(self.ub)) % k
        self.fold = self.perm_fold[np.searchsorted(self.ub, self.block_id)]
        self.train_masks = [self._train_mask(i) for i in range(k)]

    def _train_mask(self, i):
        tb = self.ub[self.perm_fold == i]
        tbx, tby = tb // 10000, tb % 10000
        excl = np.unique(np.concatenate([
            (tbx + dx) * 10000 + (tby + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        ]))
        return ~np.isin(self.block_id, excl)   # test + komşu bloklar eğitimden çıkar


def oof_predict(folds, X, target, seed=SEED, **model_kw):
    rng = np.random.default_rng(seed)
    params = dict(max_iter=200, learning_rate=0.1, max_depth=6, random_state=seed)
    params.update(model_kw)
    pred = np.full(len(target), np.nan)
    for i in range(folds.k):
        idx = np.flatnonzero(folds.train_masks[i])
        if len(idx) > MAX_TRAIN:
            idx = rng.choice(idx, MAX_TRAIN, replace=False)
        m = HistGradientBoostingRegressor(**params).fit(X[idx], target[idx])
        test = folds.fold == i
        pred[test] = m.predict(X[test])
    return pred


def percentile_from_quantiles(y, qpred, qs):
    """y'nin, tahmin edilen koşullu kantiller arasındaki konumu (0-1 arası persentil)."""
    qpred = np.sort(qpred, axis=1)
    qs = np.asarray(qs, dtype="float64")
    idx = (qpred < y[:, None]).sum(axis=1)
    p = np.empty(len(y))
    lo_m, hi_m = idx == 0, idx == len(qs)
    p[lo_m] = qs[0] / 2
    p[hi_m] = (1 + qs[-1]) / 2
    r = np.flatnonzero(~(lo_m | hi_m))
    i = idx[r]
    lo, hi = qpred[r, i - 1], qpred[r, i]
    w = (y[r] - lo) / np.maximum(hi - lo, 1e-9)
    p[r] = qs[i - 1] + w * (qs[i] - qs[i - 1])
    return p