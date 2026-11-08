import numpy as np
import pandas as pd
import pytest

import fusion.models as models
from fusion.dataset import CLASSES, drop_modality
from fusion.evaluate import loso_run, paired_test, subject_metrics
from fusion.features import feature_columns
from fusion.models import fit_fold, fit_qw, weighted_mean

SETS = {"wrist": ["bvp", "eda_wrist"], "all": ["ecg", "bvp", "eda_wrist"]}


def test_loso_never_mixes_subjects(table, monkeypatch):
    seen = []
    orig = models.fit_fold

    def spy(train, *a, **k):
        seen.append(set(train["subject"]))
        return orig(train, *a, **k)

    monkeypatch.setattr("fusion.evaluate.fit_fold", spy)
    pred = loso_run(table, "binary", SETS, "lr", n_jobs=1)
    subjects = set(table["subject"])
    assert len(seen) == len(subjects)
    for train_subjects, (s, g) in zip(seen, pred.groupby("subject", sort=False)):
        assert s not in train_subjects and train_subjects == subjects - {s}
    # every test window is predicted exactly once per model and variant
    counts = pred.groupby(["model", "variant", "subject"]).size()
    expected = table[table["label"].isin([1, 2])].groupby("subject").size()
    for (_, _, s), c in counts.items():
        assert c == expected[s]


def test_stacking_and_qw_never_see_test_subject(table, monkeypatch):
    test = "S0"
    train = table[(table["subject"] != test) & table["label"].isin([1, 2])]
    seen_inner = []
    orig = models.GroupKFold.split

    def spy(self, X, y=None, groups=None):
        for tr, va in orig(self, X, y, groups):
            seen_inner.append((set(np.asarray(groups)[tr]), set(np.asarray(groups)[va])))
            yield tr, va

    monkeypatch.setattr(models.GroupKFold, "split", spy)
    fm = fit_fold(train, [1, 2], SETS, "lr")
    assert test not in fm.inner_subjects
    assert seen_inner
    for tr, va in seen_inner:
        assert test not in tr | va and not tr & va


def test_predict_handles_missing_modality(table):
    train = table[(table["subject"] != "S0") & table["label"].isin([1, 2])]
    test = table[(table["subject"] == "S0") & table["label"].isin([1, 2])].reset_index(drop=True)
    fm = fit_fold(train, [1, 2], SETS, "lr")
    p = fm.predict(drop_modality(test, "eda_wrist"))
    for k, v in p.items():
        assert len(v) == len(test) and set(v) <= {1, 2}


def test_qw_rule_ignores_low_quality_modality():
    P = {"a": np.array([[0.9, 0.1]]), "b": np.array([[0.1, 0.9]])}
    W = {"a": np.array([1.0]), "b": np.array([0.0])}
    assert np.allclose(weighted_mean(P, W, ["a", "b"]), [[0.9, 0.1]])


def test_quality_weighting_end_to_end(table):
    """A modality corrupted in the test set and flagged by its quality is down-weighted by qw_rule."""
    t = table[table["label"].isin([1, 2])].copy()
    train, test = t[t["subject"] != "S0"], t[t["subject"] == "S0"].reset_index(drop=True)
    sets = {"s": ["ecg", "bvp", "eda_wrist"]}
    fm = fit_fold(train, [1, 2], sets, "lr")
    bad = test.copy()
    # make two modalities say the opposite of the truth, and flag them as low quality
    for m in ("ecg", "bvp"):
        cols = feature_columns([m])
        flip = np.where((bad["label"] == 2).to_numpy(), -3.0, 4.5)
        bad[cols] = np.repeat(flip[:, None], len(cols), axis=1)
        bad[f"q__{m}"] = 0.1
    p = fm.predict(bad)
    acc = {k: np.mean(v == bad["label"].to_numpy()) for k, v in p.items()}
    assert acc["s:qw_rule"] > acc["s:late_avg"] + 0.3


def test_fit_qw_slopes_non_negative():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(1, 3, n)
    good = np.c_[(y == 1) * 0.8 + 0.1, (y == 2) * 0.8 + 0.1]
    noise = rng.dirichlet([1, 1], n)
    q = rng.random(n)
    # modality "b" is informative only when its quality is high
    Pb = np.where(q[:, None] > 0.5, good, noise)
    a, b = fit_qw({"a": noise, "b": Pb}, {"a": np.ones(n), "b": q}, {"a": np.ones(n, bool), "b": np.ones(n, bool)},
                  y, np.array([1, 2]), ["a", "b"])
    assert (b >= 0).all() and b[1] > 0.5


def test_paired_test():
    a = pd.Series([0.9, 0.8, 0.95, 0.7, 0.85, 0.9], index=list("abcdef"))
    b = a - 0.1
    r = paired_test(a, b)
    assert r["wins_a"] == 6 and r["wins_b"] == 0 and r["p_wilcoxon"] < 0.05


def test_subject_metrics(table):
    pred = loso_run(table, "three_class", {"all": ["ecg", "bvp"]}, "lr", n_jobs=1)
    m = subject_metrics(pred, "three_class")
    assert m["macro_f1"].between(0, 1).all() and m["subject"].nunique() == 6
    assert CLASSES["three_class"] == [1, 2, 3]


@pytest.mark.parametrize("kind", ["lr", "rf"])
def test_base_models_learn_separable_data(table, kind):
    pred = loso_run(table, "binary", {"all": ["ecg", "eda_wrist"]}, kind, n_jobs=1)
    m = subject_metrics(pred, "binary")
    assert m[m["model"] == "all:early"]["macro_f1"].mean() > 0.75
