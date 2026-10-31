"""Het (koşullu toparlanma sapması) için ortak araçlar: veri seti, tamponlu
mekansal katlar, out-of-fold tahmin ve tanılama yardımcıları."""
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

from wildfire.config import RECOVERY_DATASET

BLOCK = 1000
K = 5
MAX_TRAIN = 400_000
SEED = 42

# het_reliability.py / het_scales.py: veri setindeki ana tarihle aynı yazdan ek tarihler
EXTRA_DATES = ["2026-07-24", "2026-08-03"]


def load_dataset(path=RECOVERY_DATASET):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def main_date(d):
    """Veri setindeki hedef (2026) görüntünün tarihi; eski npz dosyalarında yoksa varsayılan."""
    return str(d["nbr_2026_date"]) if "nbr_2026_date" in d else "2026-08-20"


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


def oof_with_baseline(folds, X, y, make_model):
    """Herhangi bir model için OOF tahmin ve 'eğitim ortalaması' referans tahmini.

    Beceri skoru = 1 - SSE(model) / SSE(referans); referans, her katta yalnızca
    o katın eğitim pikselleri ile hesaplanan ortalamadır.
    """
    rng = np.random.default_rng(SEED)
    pred = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    for i in range(folds.k):
        idx = np.flatnonzero(folds.train_masks[i])
        if len(idx) > MAX_TRAIN:
            idx = rng.choice(idx, MAX_TRAIN, replace=False)
        m = make_model().fit(X[idx], y[idx])
        test = folds.fold == i
        pred[test] = m.predict(X[test])
        base[test] = y[idx].mean()
    return pred, base


def spread_ratio(v, cond, nb=10):
    """v'nin, cond'un nb kantil grubundaki standart sapmalarının max/min oranı (1,0 ideal)."""
    edges = np.quantile(cond, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, cond, side="right") - 1, 0, nb - 1)
    s = np.array([v[idx == b].std() for b in range(nb)])
    return float(s.max() / s.min())


def smooth_grid(g, size, min_coverage=0.2):
    """NaN'ları yok sayan kayan ortalama; kapsamı min_coverage altındaki hücreler NaN olur."""
    ok = np.isfinite(g)
    num = ndimage.uniform_filter(np.where(ok, g, 0).astype("float32"), size=size, mode="constant")
    den = ndimage.uniform_filter(ok.astype("float32"), size=size, mode="constant")
    return np.where(den > min_coverage, num / np.maximum(den, 1e-6), np.nan).astype("float32")
