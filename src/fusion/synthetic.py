"""Synthetic multi-device recordings for tests (no WESAD data needed).

Signals are simple but have the structure the pipeline relies on: an ECG
with sharp QRS-like spikes at known beat times, a PPG pulse train (from
``ahr.synthetic``), EDA with a tonic level and SCRs, sinusoidal breathing,
EMG noise, slowly varying temperatures and accelerometers with gravity.
"Stress" (label 2) raises heart rate, tonic EDA and SCR rate and speeds up
breathing, so that a classifier can separate it from baseline.
"""

from __future__ import annotations

import numpy as np
from ahr.ppg import moving_average
from ahr.synthetic import pulse_train

from .io import FS, LABEL_FS, Recording

DEFAULT_PROTOCOL = [(0, 60.0), (1, 600.0), (0, 60.0), (2, 300.0), (0, 60.0), (3, 240.0), (0, 60.0)]
EFFECT = {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.2}


def _labels(protocol) -> np.ndarray:
    return np.concatenate([np.full(int(round(d * LABEL_FS)), lab, np.int8) for lab, d in protocol])


def _level(protocol, fs: float, n: int) -> np.ndarray:
    """Stress effect (0..1) per sample at rate ``fs``."""
    lab = _labels(protocol)
    idx = np.minimum((np.arange(n) / fs * LABEL_FS).astype(int), len(lab) - 1)
    return np.vectorize(EFFECT.get)(lab[idx]).astype(float)


def _ecg_from_beats(beats_s: np.ndarray, n: int, fs: float, rng) -> np.ndarray:
    t = np.arange(n) / fs
    x = 0.02 * rng.standard_normal(n)
    w = 0.012  # QRS half-width (s)
    for b in beats_s:
        lo, hi = int((b - 0.1) * fs), int((b + 0.1) * fs)
        lo, hi = max(lo, 0), min(hi, n)
        x[lo:hi] += 1.0 * np.exp(-0.5 * ((t[lo:hi] - b) / w) ** 2)
        tl, th = int((b + 0.15) * fs), int((b + 0.4) * fs)
        tl, th = max(tl, 0), min(th, n)
        x[tl:th] += 0.2 * np.exp(-0.5 * ((t[tl:th] - b - 0.27) / 0.05) ** 2)  # T wave
    return x


def make_recording(subject: str = "S0", seed: int = 0, protocol=DEFAULT_PROTOCOL, chest_fs: float = 700.0) -> Recording:
    rng = np.random.default_rng(seed)
    dur = sum(d for _, d in protocol)
    fs = dict(FS)
    for k in fs:
        if k.startswith("chest_"):
            fs[k] = chest_fs
    n = {k: int(round(dur * f)) for k, f in fs.items()}
    hr_rest = rng.uniform(60, 75)

    # heart beats: integrate an HR that rises with stress
    t_hr = np.arange(0, dur, 0.1)
    hr = hr_rest + 20.0 * _level(protocol, 10.0, len(t_hr)) + 2 * np.sin(2 * np.pi * 0.25 * t_hr)
    phase = np.cumsum(hr / 60.0 * 0.1)
    beats = np.interp(np.arange(1, phase[-1]), phase, t_hr)
    ecg = _ecg_from_beats(beats, n["chest_ecg"], fs["chest_ecg"], rng)

    # PPG: concatenate constant-rate pulse trains per protocol segment (ahr.synthetic)
    parts = []
    for i, (lab, d) in enumerate(protocol):
        _, x, _ = pulse_train(hr_rest + 20.0 * EFFECT[lab], d, fs["wrist_bvp"], noise_sd=0.02, seed=seed * 100 + i)
        parts.append(x)
    bvp = -np.concatenate(parts)[: n["wrist_bvp"]] * 100.0  # E4 polarity: inverted (pipeline flips it back)
    bvp = np.r_[bvp, np.zeros(n["wrist_bvp"] - len(bvp))]

    def eda(fs_e: float, n_e: int, base: float) -> np.ndarray:
        lev = _level(protocol, fs_e, n_e)
        t = np.arange(n_e) / fs_e
        tonic = base + 1.5 * base * moving_average(lev, int(30 * fs_e))
        x = tonic.copy()
        rate = 0.5 / 60 + 6.0 / 60 * lev  # SCRs per s
        on = np.flatnonzero(rng.random(n_e) < rate / fs_e)
        for o in on:
            k = t[o:] - t[o]
            x[o:] += 0.1 * (np.exp(-k / 4.0) - np.exp(-k / 0.75))
        return x + 0.001 * rng.standard_normal(n_e)

    resp_rate = 0.25 + 0.12 * _level(protocol, fs["chest_resp"], n["chest_resp"])
    resp_phase = 2 * np.pi * np.cumsum(resp_rate) / fs["chest_resp"]
    resp = 5 * np.sin(resp_phase) + 0.2 * rng.standard_normal(n["chest_resp"])

    def acc(n_a: int, fs_a: float, act: float) -> np.ndarray:
        t = np.arange(n_a) / fs_a
        g = np.column_stack([np.zeros(n_a), np.zeros(n_a), np.ones(n_a)])
        dyn = act * np.column_stack([np.sin(2 * np.pi * 1.8 * t), np.cos(2 * np.pi * 1.8 * t), np.zeros(n_a)])
        return g + dyn + 0.005 * rng.standard_normal((n_a, 3))

    def temp(n_t: int, fs_t: float, base: float) -> np.ndarray:
        lev = _level(protocol, fs_t, n_t)
        return base - 0.5 * lev + 0.01 * rng.standard_normal(n_t)

    signals = {
        "chest_ecg": ecg,
        "chest_eda": eda(fs["chest_eda"], n["chest_eda"], rng.uniform(2, 6)),
        "chest_emg": 0.01 * rng.standard_normal(n["chest_emg"]),
        "chest_resp": resp,
        "chest_temp": temp(n["chest_temp"], fs["chest_temp"], 34.0),
        "chest_acc": acc(n["chest_acc"], fs["chest_acc"], 0.02),
        "wrist_bvp": bvp,
        "wrist_eda": eda(fs["wrist_eda"], n["wrist_eda"], rng.uniform(0.3, 2)),
        "wrist_temp": temp(n["wrist_temp"], fs["wrist_temp"], 33.0),
        "wrist_acc": np.round(acc(n["wrist_acc"], fs["wrist_acc"], 0.05) * 64) / 64,  # E4 resolution 1/64 g
    }
    return Recording(subject, signals, _labels(protocol), fs)
