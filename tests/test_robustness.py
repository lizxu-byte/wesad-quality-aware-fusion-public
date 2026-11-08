import numpy as np
import pandas as pd

from fusion.dataset import build_subject_table, calibrate, calibration_means
from fusion.features import MODALITIES, feature_columns
from fusion.robustness import SCENARIOS, Variants, corrupt_recording, scenario_mods, select_rows


def _setup(rec, levels=(0.5,)):
    raw = build_subject_table(rec)
    raw["wid"] = raw["subject"] + ":" + raw["t0"].astype(str)
    means = calibration_means(raw)
    ev_raw = raw[raw["evaluate"]].reset_index(drop=True)
    feats, sels = corrupt_recording(rec, ev_raw, [0], list(levels))
    corr = calibrate(pd.DataFrame(feats), means)
    ev = calibrate(raw, means)
    ev = ev[ev["evaluate"]].reset_index(drop=True)
    return ev, corr, pd.DataFrame(sels)


def test_select_rows_stratified():
    ev = pd.DataFrame({"wid": [f"w{i}" for i in range(30)], "label": [1] * 20 + [2] * 10})
    sel = select_rows(ev, 0.3, np.random.default_rng(0))
    lab = ev.set_index("wid").loc[sel, "label"]
    assert (lab == 1).sum() == 6 and (lab == 2).sum() == 3 and len(set(sel)) == 9


def test_variants_change_only_the_corrupted_modalities(rec):
    ev, corr, sels = _setup(rec)
    V = Variants(corr, sels, [0], [0.5], full_drops=False)
    test = ev[ev["label"].isin([1, 2])].reset_index(drop=True)
    out = V(rec.subject, test)
    for sc in SCENARIOS:
        v = out[f"{sc}|0.5|0"]
        assert len(v) == len(test) and (v["wid"] == test["wid"]).all()
        touched = scenario_mods(sc) if not sc.startswith("drop_") else [sc[5:]]
        allowed = set(feature_columns(touched)) | {f"q__{m}" for m in touched}
        for m in MODALITIES:
            for c in feature_columns([m]) + [f"q__{m}"]:
                same = np.allclose(v[c].to_numpy(float), test[c].to_numpy(float), equal_nan=True)
                if c not in allowed:
                    assert same, (sc, c)
        # the corrupted modality's quality drops on the selected rows (on average)
        q = f"q__{touched[0]}"
        changed = ~np.isclose(v[q], test[q])
        assert changed.any(), sc
        assert v.loc[changed, q].mean() < test.loc[changed, q].mean(), sc


def test_corruption_only_in_test_windows(rec):
    """Calibration windows are never corrupted (they are not evaluated)."""
    ev, corr, sels = _setup(rec)
    raw = build_subject_table(rec)
    calib_t0 = set((rec.subject + ":" + raw.loc[raw["calib"], "t0"].astype(str)))
    assert not calib_t0 & set(sels["wid"])
