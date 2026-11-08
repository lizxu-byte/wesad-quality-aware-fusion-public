import numpy as np

from fusion import quality as Q
from fusion.dataset import subject_windows, window_features
from fusion.features import Params, extract


def _baseline_window(rec, k=3):
    w = subject_windows(rec)
    return w[w["evaluate"] & (w["label"] == 1)].iloc[k]


def test_clean_synthetic_quality_is_high(rec):
    w = _baseline_window(rec)
    f = window_features(rec, w.t0, w.t1)
    for m, q in f.items():
        if m.startswith("q__"):
            assert q > 0.9, (m, q)


def test_ppg_quality_drops_with_motion_noise(rec):
    w = _baseline_window(rec)
    clean = window_features(rec, w.t0, w.t1, mods=["bvp"])
    qs = [window_features(rec, w.t0, w.t1, mods=["bvp"], scenario="bvp_motion", rng=np.random.default_rng(i))["q__bvp"]
          for i in range(5)]
    assert np.mean(qs) < clean["q__bvp"] - 0.3


def test_eda_quality_flat_line_and_step(rec):
    w = _baseline_window(rec)
    clean = window_features(rec, w.t0, w.t1, mods=["eda_wrist"])["q__eda_wrist"]
    flat = window_features(rec, w.t0, w.t1, mods=["eda_wrist"], scenario="eda_flat", rng=np.random.default_rng(0))["q__eda_wrist"]
    assert clean == 1.0 and flat < 0.8
    # a 5 uS step at 4 Hz is a 20 uS/s change -> Kleckner rule 2 + 5 s radius
    seg = np.full(4 * 60, 1.0) + 0.001 * np.arange(240)
    seg[120:] += 5.0
    bad = Q.eda_invalid_mask(seg, 4.0)
    assert bad[120] and bad[120 - 20] and bad[120 + 19] and not bad[0] and not bad[-1]


def test_eda_kleckner_range_and_temperature_rule():
    x = np.r_[np.full(40, 0.01), np.full(200, 1.0)]  # contact loss at the start
    bad = Q.eda_invalid_mask(x, 4.0)
    assert bad[:40].all() and bad[40 + 19] and not bad[40 + 21]
    temp = np.full(240, 25.0)
    assert Q.eda_invalid_mask(np.full(240, 1.0), 4.0, temp=temp).all()
    assert not Q.eda_invalid_mask(np.full(240, 1.0) + 0.001 * np.arange(240), 4.0).any()


def test_gap_lowers_wrist_quality(rec):
    w = _baseline_window(rec)
    f = window_features(rec, w.t0, w.t1, scenario="wrist_gap", rng=np.random.default_rng(3))
    for m in ("bvp", "eda_wrist", "temp_wrist", "acc_wrist"):
        assert f[f"q__{m}"] <= 0.7
    for m in ("ecg", "eda_chest", "resp"):
        assert f[f"q__{m}"] > 0.9


def test_ecg_bsqi_drops_with_noise(rec):
    w = _baseline_window(rec)
    fs = rec.fs["chest_ecg"]
    lo, hi = int(w.t0 * fs), int(w.t1 * fs)
    x = np.asarray(rec.signals["chest_ecg"][lo - int(5 * fs) : hi + int(5 * fs)], float)
    core = (int(5 * fs), int(5 * fs) + hi - lo)
    _, q_clean = extract("ecg", x, fs, core, Params())
    rng = np.random.default_rng(0)
    noisy = x + 1.0 * rng.standard_normal(len(x))
    _, q_noisy = extract("ecg", noisy, fs, core, Params())
    assert q_clean > 0.95 and q_noisy < q_clean - 0.3


def test_beat_agreement():
    a = np.arange(0, 7000, 700)
    assert Q.beat_agreement(a, a + 30, 700.0) == 1.0  # 43 ms offset < 150 ms
    assert Q.beat_agreement(a, a + 350, 700.0) == 0.0
    assert np.isclose(Q.beat_agreement(a, a[::2], 700.0), 5 / 10)
    assert Q.beat_agreement(np.array([]), np.array([]), 700.0) == 0.0


def test_resp_dominance():
    fs = 10.0
    t = np.arange(0, 60, 1 / fs)
    regular = np.sin(2 * np.pi * 0.25 * t)
    noise = np.random.default_rng(0).standard_normal(len(t))
    d1, f1 = Q.resp_peak_dominance(regular, fs)
    d2, _ = Q.resp_peak_dominance(noise, fs)
    assert d1 > 0.8 and abs(f1 - 0.25) < 0.01 and d2 < 0.4


def test_generic_rules():
    x = np.r_[np.linspace(30, 31, 100), np.full(200, 31.0), [50.0]]
    bad = Q.generic_invalid_mask(x, 1.0, valid_range=(20, 42), flat_s=120)
    assert bad[150] and bad[-1] and not bad[10]
    acc = np.column_stack([np.full(10, 2.0), np.zeros(10), np.zeros(10)])
    assert Q.generic_invalid_mask(acc, 32.0, clip_level=127 / 64).all()


def test_flat_mask_multichannel():
    x = np.zeros((100, 3))
    x[:, 0] = np.arange(100)
    assert not Q.flat_mask(x, 10.0, 2.0).any()
    x[:, 0] = 0
    assert Q.flat_mask(x, 10.0, 2.0).all()
