"""Base classifiers, early fusion and late-fusion strategies.

All models are fitted on training subjects only. Stacking and the learned
quality weighting are fitted on *inner* out-of-fold predictions, produced by
grouped K-fold over the training subjects (no subject in both inner train and
inner validation), so the held-out test subject is never seen.

Strategies (``predict`` keys)
-----------------------------
early        one classifier on the concatenated features of the modality set;
             missing values are imputed with training medians.
late_avg     unweighted mean of the per-modality class probabilities over the
             modalities that are present in the window.
stack        logistic regression on the per-modality log-probabilities
             (inner out-of-fold); a missing modality contributes the training
             class prior.
qw_rule      weighted mean with ``w_m = q_m`` if ``q_m >= 0.5`` else 0
             (falls back to ``late_avg`` if all weights are 0).
qw_learned   weighted mean with ``w_m = exp(a_m + b_m * q_m)``, ``b_m >= 0``
             (weight can only grow with quality), ``a_m, b_m`` fitted by
             minimising class-balanced log-loss on the inner out-of-fold
             probabilities, with a small L2 penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import MODALITIES, feature_columns

STRATEGIES = ["early", "late_avg", "stack", "qw_rule", "qw_learned"]
QW_RULE_MIN_Q = 0.5
QW_L2 = 1e-2
INNER_K = 5
EPS = 1e-4


def make_base(kind: str, seed: int = 0):
    """Fixed hyper-parameters (as in ahr-stress-trigger's deep-baseline comparison); nothing tuned.

    Forests are single-threaded; parallelism is across LOSO folds (see :func:`fusion.evaluate.loso_run`).
    """
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    if kind == "lr":
        return make_pipeline(imp, StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    if kind == "rf":
        return make_pipeline(
            imp, RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=seed)
        )
    raise KeyError(kind)


def present(df: pd.DataFrame, mod: str) -> np.ndarray:
    """A modality is present in a window if at least one of its features is finite."""
    return df[feature_columns([mod])].notna().any(axis=1).to_numpy()


def _balanced_weights(y: np.ndarray, classes) -> np.ndarray:
    w = np.zeros(len(y))
    for c in classes:
        w[y == c] = len(y) / (len(classes) * max((y == c).sum(), 1))
    return w


def stack_inputs(P: dict[str, np.ndarray], avail: dict[str, np.ndarray], prior: np.ndarray, mods) -> np.ndarray:
    cols = []
    for m in mods:
        p = np.where(avail[m][:, None], P[m], prior[None, :])
        cols.append(np.log(np.clip(p, EPS, 1.0)))
    return np.hstack(cols)


def weighted_mean(P: dict[str, np.ndarray], W: dict[str, np.ndarray], mods) -> np.ndarray:
    num = sum(W[m][:, None] * P[m] for m in mods)
    den = sum(W[m] for m in mods)[:, None]
    return num / np.where(den > 0, den, 1.0)


def fit_qw(P: dict, Q: dict, avail: dict, y: np.ndarray, classes, mods, l2: float = QW_L2) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``a_m, b_m`` of ``w_m = exp(a_m + b_m q_m)`` (b_m >= 0) on out-of-fold probabilities."""
    M = len(mods)
    yi = np.searchsorted(classes, y)
    sw = _balanced_weights(y, classes)
    Pm = np.stack([P[m] for m in mods])  # (M, n, K)
    Qm = np.stack([Q[m] for m in mods])  # (M, n)
    Am = np.stack([avail[m] for m in mods]).astype(float)
    n = len(y)

    def loss(theta):
        a, b = theta[:M], theta[M:]
        w = np.exp(a[:, None] + b[:, None] * Qm) * Am  # (M, n)
        den = w.sum(0) + 1e-12
        pc = (w[:, :, None] * Pm).sum(0) / den[:, None]
        py = np.clip(pc[np.arange(n), yi], EPS, 1.0)
        return float(-(sw * np.log(py)).sum() / sw.sum() + l2 * (theta**2).sum())

    x0 = np.r_[np.zeros(M), np.ones(M)]
    bounds = [(-5, 5)] * M + [(0, 10)] * M
    r = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    return r.x[:M], r.x[M:]


@dataclass
class FoldModel:
    """Everything fitted on the training subjects of one LOSO fold."""

    kind: str
    classes: np.ndarray
    mods: list[str]
    sets: dict[str, list[str]]
    base: dict = field(default_factory=dict)
    early: dict = field(default_factory=dict)
    stack: dict = field(default_factory=dict)
    qw: dict = field(default_factory=dict)
    prior: np.ndarray | None = None

    def base_proba(self, df: pd.DataFrame) -> tuple[dict, dict, dict]:
        P, A, Qd = {}, {}, {}
        for m in self.mods:
            P[m] = self.base[m].predict_proba(df[feature_columns([m])].to_numpy(float))
            A[m] = present(df, m)
            Qd[m] = df[f"q__{m}"].fillna(0.0).to_numpy(float) * A[m]
        return P, A, Qd

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predicted labels: ``{"single:<mod>": ..., "<set>:<strategy>": ...}``."""
        P, A, Qd = self.base_proba(df)
        out = {f"single:{m}": self.classes[np.argmax(P[m], 1)] for m in self.mods}
        for s, mods in self.sets.items():
            probs = {}
            probs["early"] = self.early[s].predict_proba(df[feature_columns(mods)].to_numpy(float))
            Wp = {m: A[m].astype(float) for m in mods}
            avg = weighted_mean(P, Wp, mods)
            probs["late_avg"] = avg
            probs["stack"] = self.stack[s].predict_proba(stack_inputs(P, A, self.prior, mods))
            Wr = {m: np.where(Qd[m] >= QW_RULE_MIN_Q, Qd[m], 0.0) for m in mods}
            qr = weighted_mean(P, Wr, mods)
            none = sum(Wr[m] for m in mods) == 0
            qr[none] = avg[none]
            probs["qw_rule"] = qr
            a, b = self.qw[s]
            Wl = {m: np.exp(a[i] + b[i] * Qd[m]) * A[m] for i, m in enumerate(mods)}
            probs["qw_learned"] = weighted_mean(P, Wl, mods)
            for k, v in probs.items():
                out[f"{s}:{k}"] = self.classes[np.argmax(v, 1)]
        return out


def fit_fold(train: pd.DataFrame, classes, sets: dict[str, list[str]], kind: str, inner_k: int = INNER_K, seed: int = 0) -> FoldModel:
    """Fit base, early-fusion, stacking and learned-QW models on the training subjects only."""
    classes = np.asarray(classes)
    mods = sorted({m for v in sets.values() for m in v}, key=list(MODALITIES).index)
    y = train["label"].to_numpy()
    groups = train["subject"].to_numpy()
    fm = FoldModel(kind, classes, mods, sets)
    fm.prior = np.array([(y == c).mean() for c in classes])
    for m in mods:
        fm.base[m] = make_base(kind, seed).fit(train[feature_columns([m])].to_numpy(float), y)
    for s, ms in sets.items():
        fm.early[s] = make_base(kind, seed).fit(train[feature_columns(ms)].to_numpy(float), y)

    # inner out-of-fold probabilities (grouped by subject)
    n, K = len(train), len(classes)
    P_oof = {m: np.zeros((n, K)) for m in mods}
    for tr, va in GroupKFold(n_splits=min(inner_k, len(np.unique(groups)))).split(train, y, groups):
        assert not set(groups[tr]) & set(groups[va])
        for m in mods:
            cols = feature_columns([m])
            mdl = make_base(kind, seed).fit(train.iloc[tr][cols].to_numpy(float), y[tr])
            p = mdl.predict_proba(train.iloc[va][cols].to_numpy(float))
            # guard against an inner fold missing a class
            full = np.zeros((len(va), K))
            full[:, np.searchsorted(classes, mdl.classes_)] = p
            P_oof[m][va] = full
    A = {m: present(train, m) for m in mods}
    Qd = {m: train[f"q__{m}"].fillna(0.0).to_numpy(float) * A[m] for m in mods}
    for s, ms in sets.items():
        fm.stack[s] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(stack_inputs(P_oof, A, fm.prior, ms), y)
        fm.qw[s] = fit_qw(P_oof, Qd, A, y, classes, ms)
    fm.inner_subjects = sorted(set(groups))  # for tests: subjects seen by stacking
    return fm
