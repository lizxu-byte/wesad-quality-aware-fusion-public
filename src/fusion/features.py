"""Window-local features and quality index for every modality.

Each modality is processed from the raw samples of one 60 s window plus a short
padding on both sides (for filter settling). Features and the quality index
are computed on the 60 s *core* only. Processing is window-local on purpose:
artefacts injected into one test window then change only that window, and
clean and corrupted windows go through exactly the same code.

Missing samples (NaN, e.g. a simulated Bluetooth dropout) are linearly
interpolated before filtering; if more than half of the core is missing, all
features of that modality are NaN and its quality is 0.

Modalities (name: device, signals -> features)
-------------------------------------------------
ecg         chest ECG 700 Hz            hr, rmssd, sdnn
bvp         wrist BVP 64 Hz             hr, rmssd, sdnn
eda_chest   chest EDA 700 Hz -> 8 Hz    scl, scl_slope, scr_rate, scr_amp
eda_wrist   wrist EDA 4 Hz              scl, scl_slope, scr_rate, scr_amp
resp        chest Resp 700 Hz -> 10 Hz  rate, amp
emg         chest EMG 700 Hz            rms, rms_sd
temp_chest  chest Temp 700 Hz -> 1 Hz   mean, slope
temp_wrist  wrist TEMP 4 Hz -> 1 Hz     mean, slope
acc_chest   chest ACC 700 Hz -> 32 Hz   act, act_sd
acc_wrist   wrist ACC 32 Hz             act, act_sd
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ahr import ecg as ahr_ecg
from ahr import motion as ahr_motion
from ahr import ppg as ahr_ppg
from ahr import sqi as ahr_sqi
from scipy import signal

from . import quality as Q


@dataclass(frozen=True)
class Modality:
    name: str
    device: str  # "chest" | "wrist"
    channel: str  # key in Recording.signals
    pad_s: float
    features: tuple[str, ...]
    label: str  # human-readable


MODALITIES: dict[str, Modality] = {
    m.name: m
    for m in [
        Modality("ecg", "chest", "chest_ecg", 5.0, ("hr", "rmssd", "sdnn"), "ECG (chest)"),
        Modality("bvp", "wrist", "wrist_bvp", 5.0, ("hr", "rmssd", "sdnn"), "PPG/BVP (wrist)"),
        Modality("eda_chest", "chest", "chest_eda", 30.0, ("scl", "scl_slope", "scr_rate", "scr_amp"), "EDA (chest)"),
        Modality("eda_wrist", "wrist", "wrist_eda", 30.0, ("scl", "scl_slope", "scr_rate", "scr_amp"), "EDA (wrist)"),
        Modality("resp", "chest", "chest_resp", 10.0, ("rate", "amp"), "Respiration (chest)"),
        Modality("emg", "chest", "chest_emg", 1.0, ("rms", "rms_sd"), "EMG (chest)"),
        Modality("temp_chest", "chest", "chest_temp", 0.0, ("mean", "slope"), "Temperature (chest)"),
        Modality("temp_wrist", "wrist", "wrist_temp", 0.0, ("mean", "slope"), "Temperature (wrist)"),
        Modality("acc_chest", "chest", "chest_acc", 5.0, ("act", "act_sd"), "ACC (chest)"),
        Modality("acc_wrist", "wrist", "wrist_acc", 5.0, ("act", "act_sd"), "ACC (wrist)"),
    ]
}
WRIST = [m for m, s in MODALITIES.items() if s.device == "wrist"]
CHEST = [m for m, s in MODALITIES.items() if s.device == "chest"]


def feature_columns(mods) -> list[str]:
    return [f"{m}__{f}" for m in mods for f in MODALITIES[m].features]


@dataclass
class Params:
    """All processing constants in one place (documented in the README)."""

    win_s: float = 60.0
    max_missing: float = 0.5
    # PPG (ahr-stress-trigger defaults, offline)
    invert_bvp: bool = True
    ppg_sub_win_s: float = 30.0
    ppg_sub_step_s: float = 5.0
    ppg_min_template_corr: float = 0.8
    ppg_max_spectral_entropy: float = 0.75
    hrv_min_coverage: float = 0.5
    # ECG second detector (Elgendi 2013)
    qrs_band: tuple[float, float] = (8.0, 20.0)
    qrs_w1_s: float = 0.097
    qrs_w2_s: float = 0.611
    qrs_beta: float = 0.08
    bsqi_tol_s: float = 0.15
    ecg_clip_mv: float = 1.499  # RespiBAN ECG/EMG ADC range is +-1.5 mV after conversion
    # EDA
    eda_lp_hz: float = 1.0
    eda_tonic_hz: float = 0.05
    scr_min_amp_us: float = 0.01
    scr_min_dist_s: float = 1.0
    eda_chest_fs: float = 8.0
    eda_flat_s: float = 10.0
    eda_temp_rule: bool = False  # Kleckner rule 3, off by default (see README)
    # respiration
    resp_fs: float = 10.0
    resp_band: tuple[float, float] = (0.1, 0.7)
    resp_flat_s: float = 2.0
    # EMG
    emg_hp_hz: float = 20.0
    emg_flat_s: float = 1.0
    # temperature
    temp_range: tuple[float, float] = (20.0, 42.0)
    temp_max_rate: float = 0.5  # degC per s on the 1 Hz series
    temp_flat_s: float = 120.0
    # ACC
    acc_fs: float = 32.0
    acc_flat_s: float = 30.0  # E4 ACC (1/64 g) is bit-identical for up to ~10 s when the arm rests
    wrist_acc_clip_g: float = 127.0 / 64.0  # E4 range: counts -128..127 of 1/64 g
    chest_acc_clip_g: float = 6.0
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs per channel (nearest value at the edges)."""
    x = np.array(x, dtype=float, copy=True)
    x2 = x if x.ndim == 2 else x[:, None]
    idx = np.arange(len(x2))
    for k in range(x2.shape[1]):
        bad = ~np.isfinite(x2[:, k])
        if bad.all():
            x2[:, k] = 0.0
        elif bad.any():
            x2[bad, k] = np.interp(idx[bad], idx[~bad], x2[~bad, k])
    return x


def _resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    from fractions import Fraction

    if fs_in == fs_out:
        return np.asarray(x, float)
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(np.asarray(x, float), fr.numerator, fr.denominator, axis=0)


def _block_mean(x: np.ndarray, k: int) -> np.ndarray:
    n = len(x) // k
    return np.asarray(x[: n * k], float).reshape(n, k, *x.shape[1:]).mean(axis=1)


def _hrv_core(peaks: np.ndarray, fs: float, core_end_s: float, p: Params) -> tuple[float, float, float]:
    """HR (bpm), RMSSD, SDNN (ms) over the 60 s core ending at ``core_end_s``."""
    if len(peaks) < 3:
        return np.nan, np.nan, np.nan
    bt, ibi = ahr_ppg.ibi_from_peaks(peaks, fs)
    valid = ahr_ppg.clean_ibi(ibi)
    r = ahr_ppg.windowed_hrv(bt, ibi, valid, np.array([core_end_s]), p.win_s, p.hrv_min_coverage).iloc[0]
    return float(r["mean_hr"]), float(r["rmssd"]), float(r["sdnn"])


def _slope_per_min(y: np.ndarray, fs: float) -> float:
    if len(y) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(y)) / fs, y, 1)[0] * 60.0)


def nan_result(mod: str) -> tuple[dict, float]:
    return {f: np.nan for f in MODALITIES[mod].features}, 0.0


# ----------------------------------------------------------------------------
# per-modality processing: (padded segment, core slice) -> (features, quality)
# ----------------------------------------------------------------------------


def ecg_peaks_two_detectors(x: np.ndarray, fs: float, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """R peaks from ``ahr.ecg.detect_r_peaks`` (A) and an Elgendi (2013) two-moving-average detector (B)."""
    a = ahr_ecg.detect_r_peaks(x, fs)
    sos = signal.butter(3, list(p.qrs_band), btype="bandpass", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, x)
    # Elgendi squares the band-passed signal; ahr's implementation squares the positive part,
    # so feed |bp| to obtain bp**2 and locate the peak at the largest absolute deflection.
    b = ahr_ppg.elgendi_peaks(np.abs(bp), fs, w1_s=p.qrs_w1_s, w2_s=p.qrs_w2_s, beta=p.qrs_beta, min_distance_s=0.25)
    return a, b


def process_ecg(x, fs, core, p):
    lo, hi = core
    a, b = ecg_peaks_two_detectors(x, fs, p)
    ac, bc = a[(a >= lo) & (a < hi)], b[(b >= lo) & (b < hi)]
    hr, rmssd, sdnn = _hrv_core(a, fs, hi / fs, p)
    q = Q.beat_agreement(ac, bc, fs, p.bsqi_tol_s)
    clip = np.mean(np.abs(x[lo:hi]) >= p.ecg_clip_mv)
    return {"hr": hr, "rmssd": rmssd, "sdnn": sdnn}, q * (1.0 - clip)


def bvp_quality_and_peaks(x, fs, core, p):
    lo, hi = core
    s = -x if p.invert_bvp else x
    bp = ahr_ppg.bandpass(s, fs, causal=False)
    peaks = ahr_ppg.elgendi_peaks(bp, fs)
    t_end = np.arange(lo / fs + p.ppg_sub_win_s, hi / fs + 1e-9, p.ppg_sub_step_s)
    sq = ahr_sqi.window_quality(bp, peaks, fs, t_end, p.ppg_sub_win_s, p.ppg_min_template_corr, p.ppg_max_spectral_entropy)
    bt, ibi = ahr_ppg.ibi_from_peaks(peaks, fs)
    cov = ahr_ppg.windowed_hrv(bt, ibi, ahr_ppg.clean_ibi(ibi), t_end, p.ppg_sub_win_s, p.hrv_min_coverage)
    ok = sq["quality_ok"].to_numpy() & cov["mean_hr"].notna().to_numpy()
    return peaks, float(ok.mean()) if len(ok) else 0.0


def process_bvp(x, fs, core, p):
    peaks, q = bvp_quality_and_peaks(x, fs, core, p)
    hr, rmssd, sdnn = _hrv_core(peaks, fs, core[1] / fs, p)
    return {"hr": hr, "rmssd": rmssd, "sdnn": sdnn}, q


def process_eda(x, fs, core, p, flat_native=None, temp=None):
    """EDA at ``fs`` (already resampled). ``flat_native``: flat-line mask on the core at this rate."""
    lo, hi = core
    nyq = fs / 2
    sos_lp = signal.butter(2, min(p.eda_lp_hz, 0.9 * nyq), btype="lowpass", fs=fs, output="sos")
    sm = signal.sosfiltfilt(sos_lp, x)
    sos_t = signal.butter(2, p.eda_tonic_hz, btype="lowpass", fs=fs, output="sos")
    tonic = signal.sosfiltfilt(sos_t, sm)
    phasic = (sm - tonic)[lo:hi]
    pk, prop = signal.find_peaks(phasic, prominence=p.scr_min_amp_us, distance=max(1, int(p.scr_min_dist_s * fs)))
    dur_min = (hi - lo) / fs / 60.0
    feats = {
        "scl": float(np.mean(tonic[lo:hi])),
        "scl_slope": _slope_per_min(tonic[lo:hi], fs),
        "scr_rate": len(pk) / dur_min,
        "scr_amp": float(np.mean(prop["prominences"])) if len(pk) else 0.0,
    }
    # quality on the core samples (raw, not smoothed) at this rate; rule 4 radius may reach into the padding
    bad = Q.eda_invalid_mask(
        x,
        fs,
        temp=temp if p.eda_temp_rule else None,
        extra_invalid=None if flat_native is None else _embed(flat_native, len(x), core),
    )
    return feats, float(1.0 - bad[lo:hi].mean())


def _embed(core_mask: np.ndarray, n: int, core: tuple[int, int]) -> np.ndarray:
    out = np.zeros(n, bool)
    out[core[0] : core[0] + len(core_mask)] = core_mask[: core[1] - core[0]]
    return out


def process_resp(x, fs, core, p):
    lo, hi = core
    sos = signal.butter(2, list(p.resp_band), btype="bandpass", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, x)[lo:hi]
    dom, fpk = Q.resp_peak_dominance(x[lo:hi], fs, band=p.resp_band)
    bad = Q.generic_invalid_mask(x[lo:hi], fs, flat_s=p.resp_flat_s)
    return {"rate": fpk * 60.0, "amp": float(2 * np.sqrt(2) * np.std(bp))}, dom * (1.0 - bad.mean())


def process_emg(x, fs, core, p):
    lo, hi = core
    sos = signal.butter(4, p.emg_hp_hz, btype="highpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)[lo:hi]
    k = int(fs)
    rms = np.sqrt(_block_mean(y**2, k))
    bad = Q.generic_invalid_mask(x[lo:hi], fs, clip_level=p.ecg_clip_mv, flat_s=p.emg_flat_s)
    return {"rms": float(np.mean(rms)), "rms_sd": float(np.std(rms))}, float(1.0 - bad.mean())


def process_temp(x1hz, p):
    bad = Q.generic_invalid_mask(x1hz, 1.0, valid_range=p.temp_range, max_rate_per_s=p.temp_max_rate)
    return {"mean": float(np.mean(x1hz)), "slope": _slope_per_min(x1hz, 1.0)}, bad


def process_acc(x, fs, core, p, clip_g):
    lo, hi = core
    dyn = ahr_motion.remove_gravity(x, fs, "highpass", 0.5)
    a = ahr_motion.activity_magnitude(dyn)[lo:hi]
    return {"act": float(np.mean(a)), "act_sd": float(np.std(a))}


# ----------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------


def extract(mod: str, seg: np.ndarray, fs: float, core: tuple[int, int], p: Params, aux: dict | None = None) -> tuple[dict, float]:
    """Features and quality of modality ``mod`` from a padded raw segment.

    seg    raw samples of the channel (padded), may contain NaN
    core   (lo, hi) sample indices of the 60 s core inside ``seg``
    aux    optional extra signals, e.g. ``{"temp": wrist temperature segment at EDA rate}``
    """
    lo, hi = core
    seg = np.asarray(seg, float)
    if hi <= lo or len(seg) < hi:
        return nan_result(mod)
    core_raw = seg[lo:hi]
    fin = np.isfinite(core_raw) if core_raw.ndim == 1 else np.isfinite(core_raw).all(axis=1)
    miss = float(np.mean(~fin))
    if miss > p.max_missing:
        return nan_result(mod)
    x = _fill_nan(seg) if miss > 0 or not np.isfinite(seg).all() else seg
    keep = 1.0 - miss

    if mod == "ecg":
        f, q = process_ecg(x, fs, core, p)
    elif mod == "bvp":
        f, q = process_bvp(x, fs, core, p)
    elif mod in ("eda_chest", "eda_wrist"):
        flat = Q.flat_mask(core_raw, fs, p.eda_flat_s)  # identical raw samples at the native rate
        if mod == "eda_chest":
            xr = _resample(x, fs, p.eda_chest_fs)
            r = p.eda_chest_fs / fs
            core_r = (int(round(lo * r)), int(round(hi * r)))
            f, q = process_eda(xr, p.eda_chest_fs, core_r, p, Q.resample_mask(flat, core_r[1] - core_r[0]))
        else:
            temp = None if aux is None else aux.get("temp")
            f, q = process_eda(x, fs, core, p, flat, temp)
    elif mod == "resp":
        r = p.resp_fs / fs
        xr = _resample(x, fs, p.resp_fs)
        core_r = (int(round(lo * r)), int(round(hi * r)))
        f, q = process_resp(xr, p.resp_fs, core_r, p)
        # flat line on the native signal
        q *= 1.0 - Q.flat_mask(core_raw, fs, p.resp_flat_s).mean()
    elif mod == "emg":
        f, q = process_emg(x, fs, core, p)
    elif mod in ("temp_chest", "temp_wrist"):
        k = int(round(fs))
        x1 = _block_mean(x[lo:hi], k)
        f, bad = process_temp(x1, p)
        bad = bad | Q.resample_mask(Q.flat_mask(core_raw, fs, p.temp_flat_s), len(x1))
        q = float(1.0 - bad.mean())
    elif mod in ("acc_chest", "acc_wrist"):
        clip_g = p.chest_acc_clip_g if mod == "acc_chest" else p.wrist_acc_clip_g
        # on the gap-filled core: missing samples are accounted for once, by ``keep`` below
        bad = Q.generic_invalid_mask(x[lo:hi], fs, clip_level=clip_g, flat_s=p.acc_flat_s)
        if fs != p.acc_fs:
            r = p.acc_fs / fs
            x = _resample(x, fs, p.acc_fs)
            core = (int(round(lo * r)), int(round(hi * r)))
            fs = p.acc_fs
        f = process_acc(x, fs, core, p, clip_g)
        q = float(1.0 - bad.mean())
    else:
        raise KeyError(mod)
    q = float(np.clip(q * keep, 0.0, 1.0))
    if not np.isfinite(q):
        q = 0.0
    return f, q
