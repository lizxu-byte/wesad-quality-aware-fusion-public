"""Test-time degradation: which windows are corrupted, recomputed features, test variants.

For each test subject, scenario, corruption level and seed, ``level`` of the
windows *of each class* are drawn at random (stratified), so that every task
sees the same fraction. Artefact scenarios recompute the affected modalities'
features and quality from the corrupted raw signal (:mod:`fusion.artefacts`);
``drop_*`` scenarios remove a modality (features NaN, quality 0).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import artefacts, io
from .dataset import drop_modality, window_features
from .features import MODALITIES, feature_columns

ART_SCENARIOS = ["bvp_motion", "eda_flat", "eda_step", "wrist_gap"]
DROP_SCENARIOS = {"drop_bvp": "bvp", "drop_eda_wrist": "eda_wrist"}
SCENARIOS = ART_SCENARIOS + list(DROP_SCENARIOS)
LEVELS = [0.1, 0.3, 0.5]


def select_rows(ev_s: pd.DataFrame, level: float, rng) -> np.ndarray:
    """Stratified by label: ``level`` of each class's windows."""
    chosen = []
    for _, g in ev_s.groupby("label"):
        k = int(round(level * len(g)))
        chosen.extend(rng.choice(g["wid"].to_numpy(), size=k, replace=False))
    return np.asarray(chosen)


def scenario_mods(scenario: str) -> list[str]:
    """Modalities whose raw channel is touched by an artefact scenario."""
    chans = set(artefacts.SCENARIOS[scenario])
    return [m for m, s in MODALITIES.items() if s.channel in chans]


def corrupt_subject(data_dir: Path, s: str, ev_s: pd.DataFrame, seeds, levels) -> tuple[list, list]:
    return corrupt_recording(io.load_subject(data_dir, s), ev_s, seeds, levels)


def corrupt_recording(rec: io.Recording, ev_s: pd.DataFrame, seeds, levels) -> tuple[list, list]:
    """Selections and recomputed raw features for every scenario x level x seed of one subject.

    ``ev_s``: the subject's evaluated windows (raw, uncalibrated) with a unique ``wid`` column.
    """
    s = rec.subject
    si = zlib.crc32(s.encode())  # stable per-subject seed component
    feats, sels = [], []
    for sci, sc in enumerate(SCENARIOS):
        for lev in levels:
            for seed in seeds:
                rng = np.random.default_rng([seed, int(lev * 100), sci, si])
                wids = select_rows(ev_s, lev, rng)
                sels += [{"subject": s, "scenario": sc, "level": lev, "seed": seed, "wid": w} for w in wids]
                if sc in DROP_SCENARIOS:
                    continue
                by_wid = ev_s.set_index("wid")
                for w in wids:
                    r = by_wid.loc[w]
                    f = window_features(rec, r.t0, r.t1, mods=scenario_mods(sc), scenario=sc, rng=rng)
                    feats.append({"subject": s, "scenario": sc, "level": lev, "seed": seed, "wid": w, **f})
    return feats, sels


class Variants:
    """Picklable test-variant factory used inside the LOSO folds."""

    def __init__(self, corr: pd.DataFrame, sels: pd.DataFrame, seeds, levels, full_drops=True):
        self.corr = {k: g.set_index("wid") for k, g in corr.groupby(["subject", "scenario", "level", "seed"])}
        self.sels = {k: set(g["wid"]) for k, g in sels.groupby(["subject", "scenario", "level", "seed"])}
        self.seeds, self.levels, self.full_drops = seeds, levels, full_drops

    def __call__(self, s: str, test: pd.DataFrame) -> dict[str, pd.DataFrame]:
        out = {}
        if self.full_drops:
            for m in MODALITIES:
                out[f"drop:{m}"] = drop_modality(test, m)
        for sc in SCENARIOS:
            for lev in self.levels:
                for seed in self.seeds:
                    key = (s, sc, lev, seed)
                    sel = self.sels.get(key, set())
                    mask = test["wid"].isin(sel).to_numpy()
                    name = f"{sc}|{lev}|{seed}"
                    if sc in DROP_SCENARIOS:
                        out[name] = drop_modality(test, DROP_SCENARIOS[sc], rows=mask)
                        continue
                    v = test.copy()
                    if mask.any():
                        c = self.corr[key]
                        mods = scenario_mods(sc)  # only the modalities this scenario touches
                        cols = feature_columns(mods) + [f"q__{x}" for x in mods]
                        v.loc[mask, cols] = c.loc[test.loc[mask, "wid"], cols].to_numpy(float)
                    out[name] = v
        return out
