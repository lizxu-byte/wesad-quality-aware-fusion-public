"""LOSO evaluation over clean and degraded test variants, metrics and paired statistics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from ahr.windows import balanced_accuracy, loso_splits, macro_f1
from joblib import Parallel, delayed
from scipy import stats

from .dataset import CLASSES
from .models import fit_fold

__all__ = ["loso_run", "subject_metrics", "summarise", "paired_test", "macro_f1", "balanced_accuracy"]


def loso_run(
    data: pd.DataFrame,
    task: str,
    sets: dict[str, list[str]],
    kind: str,
    test_variants: Callable[[str, pd.DataFrame], dict[str, pd.DataFrame]] | None = None,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Window-level predictions for every LOSO fold and every test variant.

    data           calibrated window table (all subjects, evaluated windows only)
    test_variants  ``f(subject, clean_test_df) -> {variant_name: test_df}``; the clean
                   test set is always evaluated as variant ``"clean"``. Variants must
                   keep the row order of the clean test set.

    Returns a long frame: subject, variant, row, label, model, pred.
    """
    classes = CLASSES[task]
    d = data[data["label"].isin(classes)].reset_index(drop=True)
    subj = d["subject"].to_numpy()

    def one_fold(s, tr, te):
        train, test = d.iloc[tr], d.iloc[te].reset_index(drop=True)
        assert s not in set(train["subject"])
        fm = fit_fold(train, classes, sets, kind, seed=seed)
        variants = {"clean": test}
        if test_variants is not None:
            variants.update(test_variants(s, test))
        big = pd.concat([v.assign(_variant=k, _row=np.arange(len(v))) for k, v in variants.items()], ignore_index=True)
        preds = fm.predict(big)
        return pd.concat(
            [pd.DataFrame({"subject": s, "variant": big["_variant"], "row": big["_row"], "label": big["label"], "model": m, "pred": p})
             for m, p in preds.items()],
            ignore_index=True,
        )

    folds = list(loso_splits(subj))
    out = Parallel(n_jobs=n_jobs)(delayed(one_fold)(s, tr, te) for s, tr, te in folds)
    if log:
        log(f"  {task} {kind}: {len(folds)} folds done")
    res = pd.concat(out, ignore_index=True)
    res["task"], res["kind"] = task, kind
    return res


def subject_metrics(pred: pd.DataFrame, task: str, by=("kind", "model", "variant", "subject")) -> pd.DataFrame:
    classes = CLASSES[task]
    rows = []
    for key, g in pred.groupby(list(by), sort=False):
        rows.append({**dict(zip(by, key)), "n": len(g), "macro_f1": macro_f1(g["label"], g["pred"], classes),
                     "balanced_accuracy": balanced_accuracy(g["label"], g["pred"], classes)})
    out = pd.DataFrame(rows)
    out["task"] = task
    return out


def summarise(m: pd.DataFrame, by=("task", "kind", "model", "variant")) -> pd.DataFrame:
    g = m.groupby(list(by), sort=False)
    return g.agg(n_subjects=("subject", "nunique"), macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                 bacc_mean=("balanced_accuracy", "mean"), bacc_sd=("balanced_accuracy", "std")).reset_index()


def paired_test(a: pd.Series, b: pd.Series) -> dict:
    """Two-sided Wilcoxon signed-rank test on per-subject scores (aligned by index), plus win counts."""
    a, b = a.align(b, join="inner")
    d = (a - b).to_numpy(float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    return {"n": len(d), "mean_diff": float(np.mean(d)) if len(d) else np.nan, "wins_a": int((d > 0).sum()),
            "wins_b": int((d < 0).sum()), "ties": int((d == 0).sum()), "p_wilcoxon": p}
