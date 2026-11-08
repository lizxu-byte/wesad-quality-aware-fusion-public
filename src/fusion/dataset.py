"""Window table: features + quality per modality, labels, per-subject calibration.

Windows and calibration reuse :mod:`ahr.windows` from ``ahr-stress-trigger``
unchanged: 60 s windows with a 30 s step, a window is kept when >= 95 % of
it lies inside one condition, and the *calibration interval* is the first
5 min of each subject's baseline. Windows overlapping the calibration interval
are never evaluated; windows lying fully inside it provide the subject's
calibration means. Every feature is calibrated by subtracting the subject's
calibration mean (quality indices are not calibrated). Calibration means use
only windows whose quality for that modality is >= 0.5 (all calibration
windows if fewer than two qualify), and they never use data recorded after
the calibration interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ahr import windows as ahr_windows

from . import artefacts
from .features import MODALITIES, Params, extract, feature_columns
from .io import LABEL_FS, Recording

CLASSES = {"binary": [1, 2], "three_class": [1, 2, 3]}
WCFG = ahr_windows.WindowConfig(win_s=60.0, step_s=30.0, min_purity=0.95, calib_s=300.0)
CALIB_MIN_Q = 0.5


def subject_windows(rec: Recording, cfg: ahr_windows.WindowConfig = WCFG) -> pd.DataFrame:
    """Candidate windows of one subject with ``evaluate`` (labels 1-3, usable) and ``calib`` flags."""
    calib = ahr_windows.calibration_interval(rec.labels, LABEL_FS, cfg.calib_s)
    w = ahr_windows.make_windows(rec.labels, LABEL_FS, rec.duration_s, calib, cfg)
    w["calib"] = (w["t0"] >= calib[0] - 1e-9) & (w["t1"] <= calib[1] + 1e-9)
    w["evaluate"] = w["usable"] & w["label"].isin([1, 2, 3])
    w["calib_start"], w["calib_end"] = calib
    w["subject"] = rec.subject
    return w[w["evaluate"] | w["calib"]].reset_index(drop=True)


def segment(x: np.ndarray, fs: float, t0: float, t1: float, pad_s: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Padded raw segment and the (lo, hi) indices of the core ``[t0, t1)`` inside it."""
    n = int(round((t1 - t0) * fs))
    c0 = int(round(t0 * fs))
    a = max(0, c0 - int(round(pad_s * fs)))
    b = min(len(x), c0 + n + int(round(pad_s * fs)))
    return np.asarray(x[a:b], float), (c0 - a, c0 - a + n)


def window_features(
    rec: Recording,
    t0: float,
    t1: float,
    mods=None,
    p: Params | None = None,
    scenario: str | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Features (``<mod>__<feat>``) and quality (``q__<mod>``) of one window, optionally corrupted.

    Corruption touches only the 60 s core; the padding stays clean.
    """
    p = p or Params()
    mods = list(MODALITIES) if mods is None else mods
    segs = {}
    for m in mods:
        spec = MODALITIES[m]
        segs[spec.channel] = segment(rec.signals[spec.channel], rec.fs[spec.channel], t0, t1, spec.pad_s)
    temp_seg = None
    if p.eda_temp_rule and "eda_wrist" in mods:
        # Kleckner rule 3 needs wrist temperature aligned with wrist EDA (both 4 Hz, same padding)
        temp_seg = segment(rec.signals["wrist_temp"], rec.fs["wrist_temp"], t0, t1, MODALITIES["eda_wrist"].pad_s)
    if scenario is not None:
        rng = rng or np.random.default_rng(0)
        chans = [c for c in artefacts.SCENARIOS[scenario] if c in segs]
        cores = {c: segs[c][0][segs[c][1][0] : segs[c][1][1]] for c in chans}
        bad = artefacts.corrupt_core(scenario, cores, rec.fs, rng)
        for c in chans:
            x, (lo, hi) = segs[c]
            x = x.copy()
            x[lo:hi] = bad[c]
            segs[c] = (x, (lo, hi))
        if temp_seg is not None and "wrist_temp" in artefacts.SCENARIOS[scenario]:
            # same 60 s core, so the corrupted core can be reused
            if "wrist_temp" not in bad:
                bad = artefacts.corrupt_core(scenario, {**cores, "wrist_temp": temp_seg[0][temp_seg[1][0] : temp_seg[1][1]]}, rec.fs, rng)
            x, (lo, hi) = temp_seg
            x = x.copy()
            x[lo:hi] = bad["wrist_temp"]
            temp_seg = (x, (lo, hi))
    out = {}
    for m in mods:
        spec = MODALITIES[m]
        x, core = segs[spec.channel]
        aux = {"temp": temp_seg[0]} if (m == "eda_wrist" and temp_seg is not None) else None
        f, q = extract(m, x, rec.fs[spec.channel], core, p, aux)
        for k, v in f.items():
            out[f"{m}__{k}"] = v
        out[f"q__{m}"] = q
    return out


def build_subject_table(rec: Recording, p: Params | None = None) -> pd.DataFrame:
    """All evaluated and calibration windows of one subject with raw features and quality."""
    w = subject_windows(rec)
    rows = [window_features(rec, t0, t1, p=p) for t0, t1 in zip(w["t0"], w["t1"])]
    return pd.concat([w, pd.DataFrame(rows)], axis=1)


def calibration_means(table: pd.DataFrame, mods=None) -> pd.Series:
    """Per-feature calibration means of one subject (quality-filtered, see module docstring)."""
    mods = list(MODALITIES) if mods is None else mods
    cal = table[table["calib"]]
    out = {}
    for m in mods:
        good = cal[cal[f"q__{m}"] >= CALIB_MIN_Q]
        use = good if len(good) >= 2 else cal
        for c in feature_columns([m]):
            out[c] = float(np.nanmean(use[c])) if use[c].notna().any() else np.nan
    return pd.Series(out)


def calibrate(table: pd.DataFrame, means: pd.Series) -> pd.DataFrame:
    """Subtract calibration means; calibrated columns keep the raw names."""
    out = table.copy()
    cols = [c for c in means.index if c in out]
    out[cols] = out[cols].to_numpy(float) - means[cols].to_numpy(float)
    return out


def drop_modality(table: pd.DataFrame, mod: str, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Simulate a missing modality: features NaN and quality 0 (in ``rows`` or everywhere)."""
    out = table.copy()
    idx = out.index if rows is None else out.index[rows]
    out.loc[idx, feature_columns([mod])] = np.nan
    out.loc[idx, f"q__{mod}"] = 0.0
    return out
