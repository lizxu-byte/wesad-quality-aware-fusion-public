import numpy as np
import pandas as pd

from fusion.dataset import WCFG, build_subject_table, calibrate, calibration_means, subject_windows, window_features
from fusion.io import LABEL_FS, Recording


def test_windows_length_step_purity(rec):
    w = subject_windows(rec)
    assert np.allclose(w["t1"] - w["t0"], 60.0)
    # all candidate windows lie on the 30 s grid
    assert np.allclose((w["t0"] / 30.0) % 1, 0)
    ev = w[w["evaluate"]]
    assert (ev["label_purity"] >= 0.95).all()
    assert set(ev["label"]) <= {1, 2, 3}
    # each evaluated window really is >= 95 % inside its label
    for _, r in ev.iterrows():
        seg = rec.labels[int(r.t0 * LABEL_FS) : int(r.t1 * LABEL_FS)]
        assert np.mean(seg == r.label) >= 0.95


def test_calibration_interval_is_first_5_min_of_baseline_and_excluded(rec):
    w = subject_windows(rec)
    c0, c1 = w["calib_start"].iloc[0], w["calib_end"].iloc[0]
    first_baseline = np.flatnonzero(rec.labels == 1)[0] / LABEL_FS
    assert np.isclose(c0, first_baseline)
    assert np.isclose(c1 - c0, WCFG.calib_s)
    cal = w[w["calib"]]
    assert len(cal) >= 2 and ((cal["t0"] >= c0 - 1e-9) & (cal["t1"] <= c1 + 1e-9)).all()
    ev = w[w["evaluate"]]
    # no evaluated window overlaps the calibration interval
    assert not ((ev["t0"] < c1) & (ev["t1"] > c0)).any()


def _with_future_changed(rec: Recording, after_s: float) -> Recording:
    sig = {}
    for k, x in rec.signals.items():
        y = np.array(x, float, copy=True)
        y[int(after_s * rec.fs[k]) :] = y[int(after_s * rec.fs[k]) :] * 3.0 + 7.0
        sig[k] = y
    return Recording(rec.subject, sig, rec.labels, rec.fs)


def test_calibration_uses_no_future_data(rec):
    tab = build_subject_table(rec)
    c1 = tab["calib_end"].iloc[0]
    # padding of the slowest filter (EDA, 30 s) is the only look-ahead allowed past the interval
    changed = _with_future_changed(rec, c1 + 30.0)
    tab2 = build_subject_table(changed)
    m1, m2 = calibration_means(tab), calibration_means(tab2)
    pd.testing.assert_series_equal(m1, m2)
    # but evaluated windows after the change differ
    late = tab["t0"] > c1 + 60
    assert not np.allclose(tab.loc[late, "eda_wrist__scl"], tab2.loc[late, "eda_wrist__scl"])


def test_calibrate_subtracts_means(rec):
    tab = build_subject_table(rec)
    m = calibration_means(tab)
    cal = calibrate(tab, m)
    assert np.allclose(cal["ecg__hr"], tab["ecg__hr"] - m["ecg__hr"], equal_nan=True)
    assert np.allclose(cal["q__ecg"], tab["q__ecg"])  # quality is not calibrated


def test_corruption_is_window_local(rec):
    w = subject_windows(rec)
    ev = w[w["evaluate"]].reset_index(drop=True)
    a, b = ev.iloc[3], ev.iloc[4]  # overlapping neighbours
    clean_b = window_features(rec, b.t0, b.t1)
    _ = window_features(rec, a.t0, a.t1, scenario="bvp_motion", rng=np.random.default_rng(0))
    again_b = window_features(rec, b.t0, b.t1)
    assert clean_b == again_b  # the recording itself is never modified
    corrupted_a = window_features(rec, a.t0, a.t1, scenario="bvp_motion", rng=np.random.default_rng(0))
    clean_a = window_features(rec, a.t0, a.t1)
    assert corrupted_a["q__bvp"] < clean_a["q__bvp"]
    assert corrupted_a["eda_wrist__scl"] == clean_a["eda_wrist__scl"]  # other modalities untouched
