"""WESAD loader for *both* devices (chest RespiBAN and wrist Empatica E4).

Builds on :mod:`ahr.io_wesad` (pickle reader, label constants, subject list),
which loads only the wrist signals and the chest ECG. Here every channel is
kept, because the fusion analysis needs all of them.

Verified against ``wesad_readme.pdf`` (shipped in the official archive) and
against the value ranges in the synchronised ``SX.pkl`` files:

=========  ======  ======  ==================================================
device     signal  fs      unit in ``SX.pkl``
=========  ======  ======  ==================================================
chest      ECG     700 Hz  mV (already converted from raw ADC counts)
chest      EDA     700 Hz  microsiemens
chest      EMG     700 Hz  mV
chest      Resp    700 Hz  % (piezo-electric belt, relative units)
chest      Temp    700 Hz  degrees Celsius
chest      ACC     700 Hz  g, 3 axes
wrist      BVP      64 Hz  arbitrary PPG units
wrist      EDA       4 Hz  microsiemens
wrist      TEMP      4 Hz  degrees Celsius
wrist      ACC      32 Hz  raw counts of 1/64 g, 3 axes (converted to g here);
                           the E4 range is +-2 g (counts -128..127)
labels             700 Hz  0 transient, 1 baseline, 2 stress, 3 amusement,
                           4 meditation, 5-7 ignore
=========  ======  ======  ==================================================

In the pickles both devices start at the same instant (manually synchronised
by the dataset authors with a double-tap gesture), so samples are aligned by
time ``t = i / fs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ahr.io_wesad import ACC_COUNTS_PER_G, LABEL_FS, LABEL_NAMES, SUBJECT_IDS, read_pickle

__all__ = ["LABEL_FS", "LABEL_NAMES", "SUBJECT_IDS", "FS", "Recording", "cache_subject", "load_subject", "available_subjects"]

CHEST_FS = 700.0
# channel name -> sampling rate (Hz)
FS = {
    "chest_ecg": CHEST_FS,
    "chest_eda": CHEST_FS,
    "chest_emg": CHEST_FS,
    "chest_resp": CHEST_FS,
    "chest_temp": CHEST_FS,
    "chest_acc": CHEST_FS,
    "wrist_bvp": 64.0,
    "wrist_eda": 4.0,
    "wrist_temp": 4.0,
    "wrist_acc": 32.0,
}
_PKL_KEYS = {
    "chest_ecg": ("chest", "ECG"),
    "chest_eda": ("chest", "EDA"),
    "chest_emg": ("chest", "EMG"),
    "chest_resp": ("chest", "Resp"),
    "chest_temp": ("chest", "Temp"),
    "chest_acc": ("chest", "ACC"),
    "wrist_bvp": ("wrist", "BVP"),
    "wrist_eda": ("wrist", "EDA"),
    "wrist_temp": ("wrist", "TEMP"),
    "wrist_acc": ("wrist", "ACC"),
}


@dataclass
class Recording:
    """All signals of one subject. ``signals[name]`` is (n,) or (n, 3), float32/float64."""

    subject: str
    signals: dict[str, np.ndarray]
    labels: np.ndarray  # int8 at LABEL_FS
    fs: dict[str, float] = field(default_factory=lambda: dict(FS))

    @property
    def duration_s(self) -> float:
        return min(len(self.labels) / LABEL_FS, *(len(x) / self.fs[k] for k, x in self.signals.items()))


def _squeeze(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.ravel() if x.ndim == 2 and x.shape[1] == 1 else x


def from_pickle_dict(subject: str, raw: dict) -> Recording:
    sig = {}
    for name, (dev, key) in _PKL_KEYS.items():
        x = _squeeze(np.asarray(raw["signal"][dev][key], dtype=np.float32))
        if name == "wrist_acc":
            x = x / np.float32(ACC_COUNTS_PER_G)
        sig[name] = x
    return Recording(subject, sig, np.asarray(raw["label"]).astype(np.int8).ravel())


def cache_subject(pkl_path: str | Path, cache_path: str | Path) -> Path:
    """Write all channels + labels of one ``SX.pkl`` to an uncompressed ``.npz`` (float32)."""
    rec = from_pickle_dict(Path(pkl_path).stem, read_pickle(pkl_path))
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, label=rec.labels, **rec.signals)
    return cache_path


def cache_file(data_dir: str | Path, subject: str) -> Path:
    return Path(data_dir) / "cache" / f"{subject}_all.npz"


def load_subject(data_dir: str | Path, subject: str) -> Recording:
    """Load one subject from ``<data_dir>/cache/<subject>_all.npz`` or ``<data_dir>/<subject>/<subject>.pkl``."""
    cache = cache_file(data_dir, subject)
    if cache.exists():
        with np.load(cache) as z:
            return Recording(subject, {k: z[k] for k in FS}, z["label"])
    pkl = Path(data_dir) / subject / f"{subject}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Neither {cache} nor {pkl} exists; run scripts/download_wesad.py")
    return from_pickle_dict(subject, read_pickle(pkl))


def available_subjects(data_dir: str | Path) -> list[str]:
    d = Path(data_dir)
    return [s for s in SUBJECT_IDS if cache_file(d, s).exists() or (d / s / f"{s}.pkl").exists()]
