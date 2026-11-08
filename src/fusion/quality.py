"""Per-window signal-quality indices (QI) in [0, 1] for every modality.

All indices are computed from the (possibly corrupted) raw signal of one
window, so they can be recomputed after test-time artefact injection.

PPG (wrist BVP)
    Share of the seven 30 s sub-windows (5 s step) inside the 60 s window
    that pass the ``ahr-stress-trigger`` quality rule: beat-template
    correlation >= 0.8 (Orphanidou et al., 2015), spectral entropy <= 0.75 and
    valid inter-beat intervals covering >= 50 % of the sub-window. Uses
    :func:`ahr.sqi.window_quality` unchanged.
ECG (chest)
    bSQI-style agreement of two independent R-peak detectors (idea: Li, Mark &
    Clifford, 2008): ``n_matched / (n_a + n_b - n_matched)`` with a 150 ms
    matching tolerance. Detector A is :func:`ahr.ecg.detect_r_peaks`
    (Pan-Tompkins style), detector B the two-moving-average QRS detector of
    Elgendi (2013), implemented with :func:`ahr.ppg.elgendi_peaks`.
EDA (chest and wrist)
    Rules of Kleckner et al. (2018): EDA outside 0.05-60 uS, change faster
    than 10 uS/s, and data within 5 s of an invalid sample are invalid. Their
    third rule (skin temperature outside 30-40 degC) is available but off by
    default (see README: two WESAD participants have wrist temperatures of
    29-31 degC throughout). Added here, not from Kleckner: flat line (identical
    consecutive samples for >= 10 s, e.g. a frozen sensor).
    QI = 1 - share of invalid samples.
Respiration (chest)
    Dominance of the breathing peak: power within +-0.05 Hz of the largest
    spectral peak in 0.1-0.7 Hz divided by the power in 0.05-1 Hz. A regular
    breathing pattern gives values near 1, noise or irregular breathing
    spreads the power. (Ad-hoc rule, no published threshold.)
Temperature, ACC, EMG
    Generic rules: missing samples, values outside the physical/plausible
    range or at the ADC limit (clipping), flat line, and (temperature)
    implausibly fast changes. QI = 1 - share of flagged samples.

Every QI is multiplied by ``1 - share of missing (NaN) samples`` in the window.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def flat_mask(x: np.ndarray, fs: float, min_s: float) -> np.ndarray:
    """True for samples inside runs of identical consecutive values lasting >= ``min_s`` seconds.

    For multi-channel ``x`` (n, c) a sample is flat only if *all* channels are unchanged.
    """
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return np.zeros(n, bool)
    same = np.diff(x, axis=0) == 0
    if same.ndim == 2:
        same = same.all(axis=1)
    # run-length encode "same" transitions
    out = np.zeros(n, bool)
    min_len = int(np.ceil(min_s * fs))
    d = np.diff(np.r_[0, same.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # transition indices
    for s, e in zip(starts, ends):
        # transitions s..e-1 equal -> samples s..e are identical (e - s + 1 samples)
        if e - s + 1 >= min_len:
            out[s : e + 1] = True
    return out


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Extend every True sample by ``radius`` samples on both sides."""
    mask = np.asarray(mask, bool)
    if radius <= 0 or not mask.any():
        return mask
    k = np.ones(2 * radius + 1)
    return np.convolve(mask.astype(float), k, mode="same") > 0


def resample_mask(mask: np.ndarray, n_out: int) -> np.ndarray:
    """Nearest-neighbour resampling of a boolean mask to ``n_out`` samples."""
    mask = np.asarray(mask, bool)
    if len(mask) == n_out:
        return mask
    idx = np.minimum((np.arange(n_out) * len(mask) / max(n_out, 1)).astype(int), len(mask) - 1)
    return mask[idx]


# ----------------------------------------------------------------------------
# beat agreement (ECG)
# ----------------------------------------------------------------------------


def match_beats(a: np.ndarray, b: np.ndarray, tol: int) -> int:
    """Number of one-to-one matches between sorted beat indices ``a`` and ``b`` within ``tol`` samples."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    i = j = m = 0
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tol:
            m += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return m


def beat_agreement(a: np.ndarray, b: np.ndarray, fs: float, tol_s: float = 0.15) -> float:
    """bSQI-style agreement ``n_matched / (n_a + n_b - n_matched)`` (0 when no beats at all)."""
    m = match_beats(a, b, int(round(tol_s * fs)))
    denom = len(a) + len(b) - m
    return float(m / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------------
# EDA (Kleckner et al., 2018 + flat line)
# ----------------------------------------------------------------------------


def eda_invalid_mask(
    eda: np.ndarray,
    fs: float,
    min_us: float = 0.05,
    max_us: float = 60.0,
    max_slope_us_per_s: float = 10.0,
    radius_s: float = 5.0,
    temp: np.ndarray | None = None,
    temp_range: tuple[float, float] = (30.0, 40.0),
    extra_invalid: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of invalid EDA samples (Kleckner rules 1, 2, optional 3, and 4).

    ``temp`` (same length as ``eda``) enables rule 3. ``extra_invalid`` (e.g.
    a flat-line mask) is combined before the transition radius (rule 4).
    """
    eda = np.asarray(eda, float)
    bad = ~np.isfinite(eda) | (eda < min_us) | (eda > max_us)
    slope = np.abs(np.diff(eda, prepend=eda[:1])) * fs
    bad |= slope > max_slope_us_per_s
    if temp is not None:
        t = np.asarray(temp, float)
        bad |= (t < temp_range[0]) | (t > temp_range[1])
    if extra_invalid is not None:
        bad |= extra_invalid
    return dilate(bad, int(round(radius_s * fs)))


# ----------------------------------------------------------------------------
# respiration
# ----------------------------------------------------------------------------


def resp_peak_dominance(x: np.ndarray, fs: float, band=(0.1, 0.7), total=(0.05, 1.0), half_width_hz: float = 0.05) -> tuple[float, float]:
    """``(dominance, peak_frequency_hz)`` of the breathing spectral peak.

    Periodogram (Hann window, zero-padded) of the linearly detrended signal.
    """
    x = np.asarray(x, float)
    if len(x) < 4 * fs or not np.all(np.isfinite(x)) or np.std(x) == 0:
        return 0.0, float("nan")
    x = signal.detrend(x)
    nfft = max(4096, int(2 ** np.ceil(np.log2(len(x)))))
    f, p = signal.periodogram(x, fs=fs, window="hann", nfft=nfft)
    in_band = (f >= band[0]) & (f <= band[1])
    tot = (f >= total[0]) & (f <= total[1])
    if p[tot].sum() <= 0:
        return 0.0, float("nan")
    fpk = float(f[in_band][np.argmax(p[in_band])])
    near = tot & (np.abs(f - fpk) <= half_width_hz)
    return float(p[near].sum() / p[tot].sum()), fpk


# ----------------------------------------------------------------------------
# generic
# ----------------------------------------------------------------------------


def generic_invalid_mask(
    x: np.ndarray,
    fs: float,
    valid_range: tuple[float, float] | None = None,
    clip_level: float | None = None,
    flat_s: float | None = None,
    max_rate_per_s: float | None = None,
) -> np.ndarray:
    """Union of: NaN, outside ``valid_range``, ``|x| >= clip_level`` (any channel), flat line, fast change."""
    x = np.asarray(x, float)
    x2 = x if x.ndim == 2 else x[:, None]
    bad = ~np.isfinite(x2).all(axis=1)
    if valid_range is not None:
        bad |= ((x2 < valid_range[0]) | (x2 > valid_range[1])).any(axis=1)
    if clip_level is not None:
        bad |= (np.abs(x2) >= clip_level).any(axis=1)
    if flat_s is not None:
        bad |= flat_mask(x2, fs, flat_s)
    if max_rate_per_s is not None and len(x2) > 1:
        rate = np.abs(np.diff(x2, axis=0, prepend=x2[:1])) * fs
        bad |= (rate > max_rate_per_s).any(axis=1)
    return bad
