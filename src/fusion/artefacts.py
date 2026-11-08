"""Simulated artefacts, applied at test time to the raw samples of a window's 60 s core.

These are *simplified* models of real failure modes; they are not real-world
artefacts and the README says so.

bvp_motion
    Motion-like noise on the wrist BVP: Gaussian noise band-limited to
    0.5-3 Hz (the range of arm movement and gait, overlapping the heart-rate
    band) with an SD of 1-3 times the clean BVP SD, over a contiguous 50-100 %
    of the window.
eda_flat
    Frozen sensor: the wrist EDA is held at its current value for a contiguous
    30-80 % of the window.
eda_step
    Electrode-pressure step: from a random time in the first 70 % of the
    window, the wrist EDA is shifted up by 1-5 uS for the rest of the window.
wrist_gap
    Device dropout: all wrist channels (BVP, EDA, TEMP, ACC) are missing (NaN)
    for a contiguous 30-70 % of the window.
drop_<modality>
    The modality is missing for the whole window (features NaN, quality 0);
    handled in :mod:`fusion.dataset`, not here.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

SCENARIOS = {
    "bvp_motion": ["wrist_bvp"],
    "eda_flat": ["wrist_eda"],
    "eda_step": ["wrist_eda"],
    "wrist_gap": ["wrist_bvp", "wrist_eda", "wrist_temp", "wrist_acc"],
}


def _span(n: int, rng: np.random.Generator, frac_lo: float, frac_hi: float) -> tuple[int, int]:
    length = int(round(rng.uniform(frac_lo, frac_hi) * n))
    start = int(rng.integers(0, n - length + 1))
    return start, start + length


def bvp_motion(x: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
    x = np.array(x, float, copy=True)
    n = len(x)
    s, e = _span(n, rng, 0.5, 1.0)
    sos = signal.butter(2, [0.5, 3.0], btype="bandpass", fs=fs, output="sos")
    noise = signal.sosfiltfilt(sos, rng.standard_normal(n + int(4 * fs)))[int(2 * fs) : int(2 * fs) + n]
    noise *= rng.uniform(1.0, 3.0) * np.nanstd(x) / max(np.std(noise), 1e-12)
    x[s:e] += noise[s:e]
    return x


def eda_flat(x: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
    x = np.array(x, float, copy=True)
    s, e = _span(len(x), rng, 0.3, 0.8)
    x[s:e] = x[s]
    return x


def eda_step(x: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
    x = np.array(x, float, copy=True)
    s = int(rng.integers(0, int(0.7 * len(x)) + 1))
    x[s:] += rng.uniform(1.0, 5.0)
    return x


def gap(x: np.ndarray, fs: float, rng: np.random.Generator, span: tuple[float, float]) -> np.ndarray:
    """NaN over the fractional span ``span`` (shared across channels of one device)."""
    x = np.array(x, float, copy=True)
    n = len(x)
    x[int(round(span[0] * n)) : int(round(span[1] * n))] = np.nan
    return x


def corrupt_core(scenario: str, cores: dict[str, np.ndarray], fs: dict[str, float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return corrupted copies of the core segments named in ``SCENARIOS[scenario]``."""
    out = dict(cores)
    if scenario == "bvp_motion":
        out["wrist_bvp"] = bvp_motion(cores["wrist_bvp"], fs["wrist_bvp"], rng)
    elif scenario == "eda_flat":
        out["wrist_eda"] = eda_flat(cores["wrist_eda"], fs["wrist_eda"], rng)
    elif scenario == "eda_step":
        out["wrist_eda"] = eda_step(cores["wrist_eda"], fs["wrist_eda"], rng)
    elif scenario == "wrist_gap":
        length = rng.uniform(0.3, 0.7)
        start = rng.uniform(0, 1 - length)
        for ch in SCENARIOS["wrist_gap"]:
            out[ch] = gap(cores[ch], fs[ch], rng, (start, start + length))
    else:
        raise KeyError(scenario)
    return out
