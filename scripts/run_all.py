"""End-to-end analysis on WESAD: features, quality validation, fusion, robustness.

Usage
-----
    python scripts/download_wesad.py      # once (official source, ~2.2 GB)
    python scripts/run_all.py             # -> results/  (~30-40 min on 4 CPU cores)
    python scripts/run_all.py --quick     # 4 subjects, 1 seed, LR only: smoke test

Stages (derived per-window arrays are cached in ``data/derived/``, git-ignored)
1. window tables (features + quality per modality) for all 15 subjects
2. quality-index validation (PPG vs ECG heart rate, ECG semi-synthetic noise, artefact response)
3. simulated test-time corruption of 10 / 30 / 50 % of each test subject's windows
4. LOSO: single-modality, early fusion, late fusion (average, stacking, quality-weighted)
   for wrist / chest / both, on clean data, with missing modalities and with corruption
5. leave-one-modality-out, forward selection, quality-only confound check
6. natural degradation (windows with low wrist-PPG quality), paired Wilcoxon tests

Only aggregated tables, per-subject summary rows and figures are written to ``results/``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import signal, stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ahr.windows import balanced_accuracy, loso_splits, macro_f1  # noqa: E402
from fusion import io  # noqa: E402
from fusion.dataset import CLASSES, build_subject_table, calibrate, calibration_means, segment  # noqa: E402
from fusion.evaluate import loso_run, paired_test, subject_metrics, summarise  # noqa: E402
from fusion.features import CHEST, MODALITIES, WRIST, Params, extract, feature_columns  # noqa: E402
from fusion.models import STRATEGIES, fit_fold, make_base  # noqa: E402
from fusion.robustness import ART_SCENARIOS, DROP_SCENARIOS, LEVELS, SCENARIOS, Variants, corrupt_subject, scenario_mods  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

SETS = {"wrist": WRIST, "chest": CHEST, "all": list(MODALITIES)}
STRAT_LABEL = {"early": "Early fusion (imputation)", "late_avg": "Late: average", "stack": "Late: stacking",
               "qw_rule": "Late: quality-weighted (rule)", "qw_learned": "Late: quality-weighted (learned)"}
SCEN_LABEL = {"bvp_motion": "BVP motion noise", "eda_flat": "Wrist EDA flat line", "eda_step": "Wrist EDA step",
              "wrist_gap": "Wrist dropout gap", "drop_bvp": "BVP missing", "drop_eda_wrist": "Wrist EDA missing"}

# reference categorical palette (dataviz skill, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
C_STRAT = dict(zip(STRATEGIES, C))
M_STRAT = dict(zip(STRATEGIES, ["o", "s", "^", "D", "v"]))
C_DEV = {"chest": C[0], "wrist": C[1]}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------
# 1. window tables
# ----------------------------------------------------------------------------


def build_tables(data_dir: Path, cache: Path, subjects: list[str], rebuild: bool) -> pd.DataFrame:
    if cache.exists() and not rebuild:
        tab = pd.read_pickle(cache)
        if sorted(tab["subject"].unique()) == sorted(subjects):
            return tab
    tabs = Parallel(n_jobs=4)(delayed(lambda s: build_subject_table(io.load_subject(data_dir, s)))(s) for s in subjects)
    tab = pd.concat(tabs, ignore_index=True)
    tab["wid"] = tab["subject"] + ":" + tab["t0"].round(1).astype(str)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tab.to_pickle(cache)
    return tab


# ----------------------------------------------------------------------------
# 3. corruption (raw features recomputed from corrupted signals)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# 2. quality validation
# ----------------------------------------------------------------------------


def auc_good_vs_bad(q: np.ndarray, bad: np.ndarray) -> float:
    good = ~bad
    if good.sum() == 0 or bad.sum() == 0:
        return float("nan")
    return float(roc_auc_score(good.astype(int), q))


def ecg_noise_validation(data_dir: Path, raw_ev: pd.DataFrame, n_windows: int, seed: int = 0) -> pd.DataFrame:
    """Semi-synthetic: add EMG-like (20-150 Hz), in-band electrode-motion (5-25 Hz) and baseline-wander
    (0.5-2 Hz) noise, in equal parts, to real chest-ECG windows with high clean bSQI."""
    rng = np.random.default_rng(seed)
    pick = raw_ev[raw_ev["q__ecg"] >= 0.9].sample(n=min(n_windows, len(raw_ev)), random_state=seed)
    p = Params()
    rows = []
    for s, g in pick.groupby("subject"):
        rec = io.load_subject(data_dir, s)
        fs = rec.fs["chest_ecg"]
        sos_e = signal.butter(2, [20, 150], btype="bandpass", fs=fs, output="sos")  # EMG-like
        sos_i = signal.butter(2, [5, 25], btype="bandpass", fs=fs, output="sos")  # electrode motion, inside the QRS band
        sos_m = signal.butter(2, [0.5, 2.0], btype="bandpass", fs=fs, output="sos")  # baseline wander
        for _, r in g.iterrows():
            x, core = segment(rec.signals["chest_ecg"], fs, r.t0, r.t1, MODALITIES["ecg"].pad_s)
            f0, q0 = extract("ecg", x, fs, core, p)
            sd = np.std(x[core[0] : core[1]])
            for lvl in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
                comps = [signal.sosfiltfilt(so, rng.standard_normal(len(x))) for so in (sos_e, sos_i, sos_m)]
                noise = lvl * sd * sum(c / c.std() for c in comps) / np.sqrt(3)
                f, q = extract("ecg", x + noise, fs, core, p)
                rows.append({"subject": s, "wid": r.wid, "noise_sd_rel": lvl, "q_ecg": q, "hr_clean": f0["hr"], "hr": f["hr"],
                             "abs_err": abs(f["hr"] - f0["hr"]) if np.isfinite(f["hr"]) else np.nan})
    return pd.DataFrame(rows)


def quality_validation(raw_ev: pd.DataFrame, ecg_noise: pd.DataFrame, corr_raw: pd.DataFrame, out: Path) -> dict:
    res = {}
    lab = raw_ev["label"].map(io.LABEL_NAMES)
    # --- PPG: wrist HR vs chest-ECG HR, ECG reference restricted to high bSQI
    ref = raw_ev[(raw_ev["q__ecg"] >= 0.9) & raw_ev["ecg__hr"].notna()].copy()
    ref["err"] = (ref["bvp__hr"] - ref["ecg__hr"]).abs()
    ref["bad"] = ref["err"].isna() | (ref["err"] > 5.0)
    res["ppg_auc"] = auc_good_vs_bad(ref["q__bvp"].to_numpy(), ref["bad"].to_numpy())
    has = ref["err"].notna()
    res["ppg_spearman_q_err"] = float(stats.spearmanr(ref.loc[has, "q__bvp"], ref.loc[has, "err"]).statistic)
    res["ppg_n"] = int(len(ref))
    bins = pd.cut(ref["q__bvp"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"])
    by_bin = ref.groupby(bins, observed=False).agg(n=("err", "size"), hr_available=("err", lambda e: e.notna().mean()),
                                                   mae_bpm=("err", "mean"), median_err_bpm=("err", "median"),
                                                   share_err_gt5_or_missing=("bad", "mean")).reset_index().rename(columns={"q__bvp": "q_bvp_bin"})
    by_bin.to_csv(out / "quality_validation_ppg_bins.csv", index=False, float_format="%.3f")

    cond = []
    for name, g in raw_ev.groupby(lab):
        r = ref[ref["label"] == g["label"].iloc[0]]
        hi = r[r["q__bvp"] >= 0.5]
        cond.append({"condition": name, "n_windows": len(g), **{f"mean_q_{m}": g[f"q__{m}"].mean() for m in MODALITIES},
                     "share_q_bvp_ge_0.5": (g["q__bvp"] >= 0.5).mean(), "share_bvp_hr_available": g["bvp__hr"].notna().mean(),
                     "ppg_mae_bpm_q_ge_0.5": hi["err"].mean(), "ppg_mae_bpm_all": r["err"].mean()})
    pd.DataFrame(cond).to_csv(out / "quality_by_condition.csv", index=False, float_format="%.3f")

    # --- ECG: chest HR vs wrist HR on windows with excellent PPG quality
    r2 = raw_ev[(raw_ev["q__bvp"] >= 6 / 7 - 1e-9) & raw_ev["bvp__hr"].notna()].copy()
    r2["err"] = (r2["ecg__hr"] - r2["bvp__hr"]).abs()
    r2["bad"] = r2["err"].isna() | (r2["err"] > 5.0)
    res["ecg_vs_ppg_n"], res["ecg_vs_ppg_n_bad"] = int(len(r2)), int(r2["bad"].sum())
    res["ecg_vs_ppg_auc"] = auc_good_vs_bad(r2["q__ecg"].to_numpy(), r2["bad"].to_numpy())
    res["ecg_q_min_real"] = float(raw_ev["q__ecg"].min())
    # --- ECG semi-synthetic noise
    e = ecg_noise.copy()
    e["bad"] = e["abs_err"].isna() | (e["abs_err"] > 5.0)
    res["ecg_noise_auc"] = auc_good_vs_bad(e["q_ecg"].to_numpy(), e["bad"].to_numpy())
    res["ecg_noise_n"] = int(len(e))
    e.groupby("noise_sd_rel").agg(n=("q_ecg", "size"), mean_q_ecg=("q_ecg", "mean"), mae_bpm=("abs_err", "mean"),
                                  share_err_gt5_or_missing=("bad", "mean")).reset_index().to_csv(
        out / "quality_validation_ecg_noise.csv", index=False, float_format="%.3f")
    # --- PPG with injected motion noise: does q_bvp still rank HR error?
    if len(corr_raw):
        cm = corr_raw[corr_raw["scenario"] == "bvp_motion"].merge(raw_ev[["wid", "ecg__hr", "q__ecg"]], on="wid")
        cm = cm[(cm["q__ecg"] >= 0.9) & cm["ecg__hr"].notna()]
        both = pd.concat([ref[["q__bvp", "bvp__hr", "ecg__hr"]], cm[["q__bvp", "bvp__hr", "ecg__hr"]]])
        err = (both["bvp__hr"] - both["ecg__hr"]).abs()
        res["ppg_auc_with_injected_motion"] = auc_good_vs_bad(both["q__bvp"].to_numpy(), (err.isna() | (err > 5)).to_numpy())
        # artefact response of each quality index (same windows, clean vs corrupted)
        resp = []
        clean = raw_ev.set_index("wid")
        for sc in ART_SCENARIOS:
            g = corr_raw[corr_raw["scenario"] == sc]
            for m in scenario_mods(sc):
                resp.append({"scenario": sc, "modality": m, "n": len(g), "mean_q_clean": clean.loc[g["wid"], f"q__{m}"].mean(),
                             "mean_q_corrupted": g[f"q__{m}"].mean(),
                             "share_q_below_0.5_clean": (clean.loc[g["wid"], f"q__{m}"] < 0.5).mean(),
                             "share_q_below_0.5_corrupted": (g[f"q__{m}"] < 0.5).mean(),
                             "share_features_missing_corrupted": g[feature_columns([m])].isna().all(axis=1).mean()})
        pd.DataFrame(resp).to_csv(out / "quality_artefact_response.csv", index=False, float_format="%.3f")
    with open(out / "quality_validation.json", "w") as f:
        json.dump(res, f, indent=2)

    # --- figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    ax = axes[0]
    data = [ref.loc[(bins == b) & has, "err"].clip(upper=40).to_numpy() for b in bins.cat.categories]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=TEXT, lw=1.5))
    for b in bp["boxes"]:
        b.set(facecolor=C[0], alpha=0.35, edgecolor=C[0])
    ax.set_xticks(range(1, len(data) + 1), bins.cat.categories)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.12)
    for i, d in enumerate(data, 1):
        ax.text(i, top * 1.1, f"n={len(d)}", ha="center", va="top", fontsize=7, color=MUTED)
    ax.set_xlabel("PPG quality index (wrist BVP)")
    ax.set_ylabel("|wrist HR − chest-ECG HR| (bpm)")
    ax.set_title(f"PPG quality vs HR error (AUC {res['ppg_auc']:.2f})", fontsize=9, color=TEXT, loc="left")
    style(ax)
    ax = axes[1]
    cd = pd.DataFrame(cond).set_index("condition").loc[["baseline", "stress", "amusement"]]
    x = np.arange(len(cd))
    ax.bar(x, cd["share_q_bvp_ge_0.5"], color=C[1], width=0.55)
    for i, v in enumerate(cd["share_q_bvp_ge_0.5"]):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(x, cd.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of windows with PPG quality ≥ 0.5")
    ax.set_title("Wrist PPG usable per condition", fontsize=9, color=TEXT, loc="left")
    style(ax)
    ax = axes[2]
    g = e.groupby("noise_sd_rel")
    ax.plot(g["q_ecg"].mean().index, g["q_ecg"].mean(), marker="o", color=C[0], lw=2, label="mean ECG quality (bSQI)")
    ax.plot(g["bad"].mean().index, g["bad"].mean(), marker="s", color=C[1], lw=2, label="share HR error > 5 bpm")
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xticks([0, 0.25, 0.5, 1, 2, 4], ["0", "0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("added noise SD (× clean ECG SD)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"ECG semi-synthetic noise (AUC {res['ecg_noise_auc']:.2f})", fontsize=9, color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.savefig(out / "figures" / "quality_validation.png", dpi=150)
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------
# 5. selection
# ----------------------------------------------------------------------------


def _early_fold(d, cols, kind, tr, te, classes):
    m = make_base(kind).fit(d.iloc[tr][cols].to_numpy(float), d.iloc[tr]["label"].to_numpy())
    p = m.predict(d.iloc[te][cols].to_numpy(float))
    y = d.iloc[te]["label"].to_numpy()
    return {"subject": d.iloc[te]["subject"].iloc[0], "macro_f1": macro_f1(y, p, classes), "balanced_accuracy": balanced_accuracy(y, p, classes)}


def loso_early(ev: pd.DataFrame, task: str, cols: list[str], kind: str) -> pd.DataFrame:
    classes = CLASSES[task]
    d = ev[ev["label"].isin(classes)].reset_index(drop=True)
    folds = list(loso_splits(d["subject"].to_numpy()))
    return pd.DataFrame(Parallel(n_jobs=4)(delayed(_early_fold)(d, cols, kind, tr, te, classes) for _, tr, te in folds))


def forward_selection(ev, task, kind) -> pd.DataFrame:
    chosen, rows, remaining = [], [], list(MODALITIES)
    while remaining:
        scores = {m: loso_early(ev, task, feature_columns(chosen + [m]), kind)["macro_f1"] for m in remaining}
        best = max(scores, key=lambda m: scores[m].mean())
        chosen.append(best)
        remaining.remove(best)
        rows.append({"task": task, "kind": kind, "step": len(chosen), "added": best, "device": MODALITIES[best].device,
                     "modalities": "+".join(chosen), "macro_f1_mean": scores[best].mean(), "macro_f1_sd": scores[best].std(),
                     "runner_up": max((m for m in scores if m != best), key=lambda m: scores[m].mean(), default=""),
                     "runner_up_f1": max((scores[m].mean() for m in scores if m != best), default=np.nan)})
        log(f"  forward {task} {kind} step {len(chosen)}: +{best} -> {scores[best].mean():.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/WESAD"))
    ap.add_argument("--derived", type=Path, default=Path("data/derived"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--quick", action="store_true", help="4 subjects, 1 seed, LR only (smoke test)")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    subjects = io.available_subjects(args.data_dir)
    seeds, kinds = [0, 1, 2], ["rf", "lr"]
    if args.quick:
        subjects, seeds, kinds = subjects[:4], [0], ["lr"]
        args.out, args.derived = args.out / "quick", args.derived / "quick"
    if len(subjects) < 3:
        raise SystemExit("WESAD cache not found; run scripts/download_wesad.py")
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    args.derived.mkdir(parents=True, exist_ok=True)
    log(f"{len(subjects)} subjects: {', '.join(subjects)}")

    # ---- 1. tables
    raw = build_tables(args.data_dir, args.derived / "windows.pkl", subjects, args.rebuild)
    means = {s: calibration_means(g) for s, g in raw.groupby("subject")}
    cal = pd.concat([calibrate(g, means[s]) for s, g in raw.groupby("subject")], ignore_index=True)
    raw_ev = raw[raw["evaluate"]].reset_index(drop=True)
    ev = cal[cal["evaluate"]].reset_index(drop=True)
    log(f"tables: {len(raw)} windows ({len(ev)} evaluated, {int(raw['calib'].sum())} calibration)")
    ev.groupby(["subject", ev["label"].map(io.LABEL_NAMES)]).size().unstack().assign(
        calibration_windows=raw[raw["calib"]].groupby("subject").size()).to_csv(out / "windows_per_subject.csv")

    # ---- 3. corruption
    corr_cache = args.derived / "corrupted.pkl"
    if corr_cache.exists() and not args.rebuild:
        corr_raw, sels = pd.read_pickle(corr_cache)
    else:
        res = Parallel(n_jobs=4)(delayed(corrupt_subject)(args.data_dir, s, raw_ev[raw_ev["subject"] == s], seeds, LEVELS)
                                 for s in subjects)
        corr_raw = pd.DataFrame([r for f, _ in res for r in f])
        sels = pd.DataFrame([r for _, s in res for r in s])
        pd.to_pickle((corr_raw, sels), corr_cache)
    log(f"corruption: {len(corr_raw)} corrupted windows recomputed")
    corr = pd.concat([calibrate(g, means[s]) for s, g in corr_raw.groupby("subject")], ignore_index=True)

    # ---- 2. quality validation
    ecg_cache = args.derived / "ecg_noise.pkl"
    if ecg_cache.exists() and not args.rebuild:
        ecg_noise = pd.read_pickle(ecg_cache)
    else:
        ecg_noise = ecg_noise_validation(args.data_dir, raw_ev, 60 if args.quick else 300)
        ecg_noise.to_pickle(ecg_cache)
    qv = quality_validation(raw_ev, ecg_noise, corr_raw, out)
    log(f"quality validation: {qv}")

    # ---- 4. main LOSO runs
    variants = Variants(corr, sels, seeds, LEVELS)
    metrics, qw_weights = [], []
    preds_clean = []
    for task in CLASSES:
        for kind in kinds:
            pc = args.derived / f"preds_{task}_{kind}.pkl"
            if pc.exists() and not args.rebuild:
                pr = pd.read_pickle(pc)
            else:
                t = time.time()
                pr = loso_run(ev, task, SETS, kind, test_variants=variants, log=log)
                pr.to_pickle(pc)
                log(f"LOSO {task} {kind}: {time.time() - t:.0f} s")
            metrics.append(subject_metrics(pr, task))
            preds_clean.append(pr[pr["variant"] == "clean"])
    m = pd.concat(metrics, ignore_index=True)
    m["set"] = m["model"].str.split(":").str[0]
    m["strategy"] = m["model"].str.split(":").str[1]
    clean = m[m["variant"] == "clean"]
    summ = summarise(clean)
    summ.to_csv(out / "summary_clean.csv", index=False, float_format="%.4f")
    clean[["task", "kind", "model", "subject", "n", "macro_f1", "balanced_accuracy"]].to_csv(
        out / "per_subject_clean.csv", index=False, float_format="%.4f")

    # missing modality at test time (100 %)
    drops = m[m["variant"].str.startswith("drop:") & ~m["set"].eq("single")]
    dsum = summarise(drops, by=("task", "kind", "set", "strategy", "variant")).merge(
        summarise(clean[~clean["set"].eq("single")], by=("task", "kind", "set", "strategy"))[
            ["task", "kind", "set", "strategy", "macro_f1_mean"]].rename(columns={"macro_f1_mean": "clean_macro_f1"}),
        on=["task", "kind", "set", "strategy"])
    dsum["delta_macro_f1"] = dsum["macro_f1_mean"] - dsum["clean_macro_f1"]
    dsum["dropped"] = dsum["variant"].str[5:]
    dsum = dsum[[r.dropped in SETS[r.set] for r in dsum.itertuples()]]
    dsum.drop(columns="variant").to_csv(out / "missing_modality.csv", index=False, float_format="%.4f")

    # corruption curves (mean over seeds per subject first)
    cv = m[m["variant"].str.contains(r"\|") & ~m["set"].eq("single")].copy()
    parts = cv["variant"].str.split("|", expand=True)
    cv["scenario"], cv["level"] = parts[0], parts[1].astype(float)
    cvs = cv.groupby(["task", "kind", "set", "strategy", "scenario", "level", "subject"])[["macro_f1", "balanced_accuracy"]].mean().reset_index()
    base = clean[~clean["set"].eq("single")]
    for sc in SCENARIOS:
        cvs = pd.concat([cvs, base.assign(scenario=sc, level=0.0)[cvs.columns]], ignore_index=True)
    curves = summarise(cvs, by=("task", "kind", "set", "strategy", "scenario", "level")).sort_values(
        ["task", "kind", "set", "scenario", "strategy", "level"])
    curves.to_csv(out / "corruption_curves.csv", index=False, float_format="%.4f")
    cvs.to_csv(out / "corruption_per_subject.csv", index=False, float_format="%.4f")

    # ---- 6a. natural degradation: windows with low wrist-PPG quality
    nat = []
    for pr in preds_clean:
        task, kind = pr["task"].iloc[0], pr["kind"].iloc[0]
        d = ev[ev["label"].isin(CLASSES[task])].reset_index(drop=True)
        d["row"] = d.groupby("subject").cumcount()
        x = pr.merge(d[["subject", "row", "q__bvp"]], on=["subject", "row"])
        x = x[~x["model"].str.startswith("single:") | x["model"].isin(["single:bvp", "single:eda_wrist", "single:ecg"])]
        x["stratum"] = np.where(x["q__bvp"] < 0.5, "PPG quality < 0.5", "PPG quality >= 0.5")
        for (model, stratum), g in x.groupby(["model", "stratum"]):
            row = {"task": task, "kind": kind, "model": model, "stratum": stratum, "n_windows": len(g),
                   "pooled_macro_f1": macro_f1(g["label"], g["pred"], CLASSES[task]),
                   "pooled_balanced_accuracy": balanced_accuracy(g["label"], g["pred"], CLASSES[task])}
            for c in CLASSES[task]:
                gc = g[g["label"] == c]
                row[f"n_{io.LABEL_NAMES[c]}"] = len(gc)
                row[f"recall_{io.LABEL_NAMES[c]}"] = (gc["pred"] == c).mean() if len(gc) else np.nan
            ps = [balanced_accuracy(h["label"], h["pred"], CLASSES[task]) for _, h in g.groupby("subject")
                  if h["label"].nunique() == len(CLASSES[task])]
            row["n_subjects_all_classes"] = len(ps)
            row["per_subject_bacc_mean"] = np.mean(ps) if ps else np.nan
            nat.append(row)
    nat = pd.DataFrame(nat)
    nat.to_csv(out / "natural_degradation.csv", index=False, float_format="%.4f")

    # ---- 5. selection, LOMO, quality-only confound check
    sel_rows, lomo_rows, qonly = [], [], []
    sel_cache = args.derived / "selection.pkl"
    if sel_cache.exists() and not args.rebuild:
        fwd, lomo, qonly = pd.read_pickle(sel_cache)
    else:
        for kind in kinds:
            sel_rows.append(forward_selection(ev, "binary", kind))
            for task in CLASSES:
                full = loso_early(ev, task, feature_columns(list(MODALITIES)), kind).set_index("subject")
                for mo in MODALITIES:
                    r = loso_early(ev, task, feature_columns([x for x in MODALITIES if x != mo]), kind).set_index("subject")
                    t = paired_test(full["macro_f1"], r["macro_f1"])
                    lomo_rows.append({"task": task, "kind": kind, "left_out": mo, "device": MODALITIES[mo].device,
                                      "macro_f1_all": full["macro_f1"].mean(), "macro_f1_without": r["macro_f1"].mean(),
                                      "delta": full["macro_f1"].mean() - r["macro_f1"].mean(), **{f"wilcoxon_{k}": v for k, v in t.items()}})
                q = loso_early(ev, task, [f"q__{x}" for x in MODALITIES], kind)
                qonly.append({"task": task, "kind": kind, "features": "quality indices only",
                              "macro_f1_mean": q["macro_f1"].mean(), "macro_f1_sd": q["macro_f1"].std(),
                              "bacc_mean": q["balanced_accuracy"].mean()})
                for mo in ("bvp", "resp"):
                    q = loso_early(ev, task, [f"q__{mo}"], kind)
                    qonly.append({"task": task, "kind": kind, "features": f"q__{mo} only", "macro_f1_mean": q["macro_f1"].mean(),
                                  "macro_f1_sd": q["macro_f1"].std(), "bacc_mean": q["balanced_accuracy"].mean()})
            log(f"selection {kind} done")
        fwd, lomo, qonly = pd.concat(sel_rows), pd.DataFrame(lomo_rows), pd.DataFrame(qonly)
        pd.to_pickle((fwd, lomo, qonly), sel_cache)
    fwd.to_csv(out / "forward_selection.csv", index=False, float_format="%.4f")
    lomo.to_csv(out / "leave_one_modality_out.csv", index=False, float_format="%.4f")
    qonly.to_csv(out / "quality_only_confound.csv", index=False, float_format="%.4f")

    # ---- 6b. paired tests
    tests = []

    def add(family, task, kind, desc, a: pd.Series, b: pd.Series):
        tests.append({"family": family, "task": task, "kind": kind, "comparison": desc, **paired_test(a, b)})

    def ser(df, **kw):
        q = df
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.set_index("subject")["macro_f1"]

    for task in CLASSES:
        for kind in kinds:
            for st in ("wrist", "chest", "all"):
                for s2 in ("late_avg", "stack", "qw_rule", "qw_learned"):
                    add("clean", task, kind, f"{st}: {s2} - early", ser(clean, task=task, kind=kind, model=f"{st}:{s2}"),
                        ser(clean, task=task, kind=kind, model=f"{st}:early"))
                add("clean", task, kind, f"{st}: qw_rule - late_avg", ser(clean, task=task, kind=kind, model=f"{st}:qw_rule"),
                    ser(clean, task=task, kind=kind, model=f"{st}:late_avg"))
            for st in ("wrist", "all"):
                for sc in SCENARIOS:
                    c50 = cvs[(cvs.task == task) & (cvs.kind == kind) & (cvs.set == st) & (cvs.scenario == sc) & (cvs.level == 0.5)]
                    c0 = cvs[(cvs.task == task) & (cvs.kind == kind) & (cvs.set == st) & (cvs.scenario == sc) & (cvs.level == 0.0)]
                    add("corrupted_50pct_qw_vs_avg", task, kind, f"{st} / {sc}: qw_rule - late_avg", ser(c50, strategy="qw_rule"),
                        ser(c50, strategy="late_avg"))
                    for s2 in ("qw_rule", "qw_learned", "late_avg", "stack"):
                        add("corrupted_50pct", task, kind, f"{st} / {sc}: {s2} - early", ser(c50, strategy=s2), ser(c50, strategy="early"))
                        # graceful degradation: loss from clean to 50 % (positive = smaller loss for s2)
                        loss_s2 = ser(c0, strategy=s2) - ser(c50, strategy=s2)
                        loss_e = ser(c0, strategy="early") - ser(c50, strategy="early")
                        add("degradation_loss", task, kind, f"{st} / {sc}: loss(early) - loss({s2})", loss_e, loss_s2)
    tests = pd.DataFrame(tests)
    # Holm correction within each family x task x kind
    tests["p_holm"] = np.nan
    for _, g in tests.groupby(["family", "task", "kind"]):
        order = g["p_wilcoxon"].sort_values().index
        k = len(order)
        adj = np.minimum(1, np.maximum.accumulate([(k - i) * tests.loc[ix, "p_wilcoxon"] for i, ix in enumerate(order)]))
        tests.loc[order, "p_holm"] = adj
    tests.to_csv(out / "wilcoxon.csv", index=False, float_format="%.4g")

    # ---- learned quality weights (logistic-regression base models; cheap to refit per fold)
    wrows = []
    for task in CLASSES:
        d = ev[ev["label"].isin(CLASSES[task])].reset_index(drop=True)
        for subj, tr, _ in loso_splits(d["subject"].to_numpy()):
            fm = fit_fold(d.iloc[tr], CLASSES[task], {"wrist": WRIST, "all": list(MODALITIES)}, "lr")
            for st, (a, b) in fm.qw.items():
                w1 = np.exp(a + b)  # weight at quality 1
                for i, mo in enumerate(fm.sets[st]):
                    wrows.append({"task": task, "set": st, "fold": subj, "modality": mo, "a": a[i], "b": b[i],
                                  "weight_share_at_q1": w1[i] / w1.sum(), "weight_ratio_q0_to_q1": float(np.exp(-b[i]))})
    qww = pd.DataFrame(wrows).groupby(["task", "set", "modality"])[["a", "b", "weight_share_at_q1", "weight_ratio_q0_to_q1"]].agg(["mean", "std"])
    qww.columns = ["_".join(c) for c in qww.columns]
    qww.reset_index().to_csv(out / "qw_learned_weights_lr.csv", index=False, float_format="%.3f")

    # ---- figures
    fig_contribution(clean, lomo, fwd, out, kinds[0])
    for task in CLASSES:
        fig_curves(curves, task, kinds[0], out)
    fig_natural(nat, kinds[0], out)

    # ---- summary
    def mean_of(task, kind, model):
        r = summ[(summ.task == task) & (summ.kind == kind) & (summ.model == model)]
        return {"macro_f1": round(float(r["macro_f1_mean"].iloc[0]), 3), "macro_f1_sd": round(float(r["macro_f1_sd"].iloc[0]), 3),
                "bacc": round(float(r["bacc_mean"].iloc[0]), 3)}

    k0 = kinds[0]
    summary = {
        "subjects": len(subjects), "windows_evaluated": int(len(ev)), "kinds": kinds, "seeds": seeds,
        "quality_validation": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in qv.items()},
        "clean_binary": {mod: mean_of("binary", k0, mod) for mod in summ[(summ.task == "binary") & (summ.kind == k0)]["model"]},
        "clean_three_class": {mod: mean_of("three_class", k0, mod) for mod in summ[(summ.task == "three_class") & (summ.kind == k0)]["model"]},
        "runtime_min": round((time.time() - t_start) / 60, 1),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"done in {summary['runtime_min']} min -> {out}")


def fig_contribution(clean, lomo, fwd, out, kind):
    sm = clean[(clean.task == "binary") & (clean.kind == kind) & (clean.set == "single")].groupby("strategy")["macro_f1"].agg(["mean", "std"])
    lr = clean[(clean.task == "binary") & (clean.kind == "lr") & (clean.set == "single")].groupby("strategy")["macro_f1"].mean()
    sm = sm.sort_values("mean")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1, 1.3]})
    ax = axes[0]
    y = np.arange(len(sm))
    ax.barh(y, sm["mean"], xerr=sm["std"], color=[C_DEV[MODALITIES[m].device] for m in sm.index], height=0.6,
            error_kw=dict(ecolor=MUTED, lw=1, capsize=0))
    if kind != "lr" and len(lr):
        ax.scatter(lr.reindex(sm.index), y, marker="|", s=120, color=TEXT, zorder=3, label="logistic regression")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_yticks(y, [MODALITIES[m].label for m in sm.index], fontsize=8)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel(f"macro-F1 (mean ± SD over {clean['subject'].nunique()} subjects)")
    ax.set_title(f"Single modality, stress vs baseline ({'RF' if kind == 'rf' else 'LR'})", fontsize=9, loc="left", color=TEXT)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    lo = lomo[(lomo.task == "binary") & (lomo.kind == kind)].sort_values("delta")
    y = np.arange(len(lo))
    ax.barh(y, lo["delta"], color=[C_DEV[d] for d in lo["device"]], height=0.6)
    ax.set_yticks(y, [MODALITIES[m].label for m in lo["left_out"]], fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("macro-F1 lost when the modality is left out")
    ax.set_title("Leave-one-modality-out (early fusion, all sensors)", fontsize=9, loc="left", color=TEXT)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax = axes[2]
    f = fwd[fwd.kind == kind]
    ax.plot(f["step"], f["macro_f1_mean"], color=C[2], lw=2, marker="o", ms=5)
    ax.set_xticks(f["step"], [f"+{MODALITIES[a].label}" for a in f["added"]], rotation=45, ha="right", fontsize=7)
    for t, d in zip(ax.get_xticklabels(), f["device"]):
        t.set_color(C_DEV[d])
    ax.set_xlabel("greedy forward selection (modality added at each step)")
    ax.set_ylabel("macro-F1 (mean over subjects)")
    lo_, hi_ = f["macro_f1_mean"].min(), f["macro_f1_mean"].max()
    ax.set_ylim(lo_ - 0.03, min(1.005, hi_ + 0.02))
    ax.set_title("Which sensors are worth wearing (label colour = device)", fontsize=9, loc="left", color=TEXT)
    style(ax)
    for t, d in zip(ax.get_xticklabels(), f["device"]):  # after style(), which resets tick colours
        t.set_color(C_DEV[d])
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_DEV[d]) for d in ("chest", "wrist")]
    axes[0].legend(handles + axes[0].get_legend_handles_labels()[0], ["chest (RespiBAN)", "wrist (E4)"] + axes[0].get_legend_handles_labels()[1],
                   frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "figures" / "modality_contribution.png", dpi=150)
    plt.close(fig)


def fig_curves(curves, task, kind, out):
    c = curves[(curves.task == task) & (curves.kind == kind)]
    sets = ["wrist", "all"]
    fig, axes = plt.subplots(len(sets), len(SCENARIOS), figsize=(16, 6), sharex=True, sharey="row", constrained_layout=True)
    for i, st in enumerate(sets):
        for j, sc in enumerate(SCENARIOS):
            ax = axes[i, j]
            for strat in STRATEGIES:
                g = c[(c.set == st) & (c.scenario == sc) & (c.strategy == strat)].sort_values("level")
                ax.plot(g["level"] * 100, g["macro_f1_mean"], color=C_STRAT[strat], marker=M_STRAT[strat], ms=5, lw=2,
                        label=STRAT_LABEL[strat])
            if i == 0:
                ax.set_title(SCEN_LABEL[sc], fontsize=9, color=TEXT)
            if j == 0:
                ax.set_ylabel(f"{'wrist only' if st == 'wrist' else 'chest + wrist'}\nmacro-F1", fontsize=9)
            if i == len(sets) - 1:
                ax.set_xlabel("% of test windows corrupted", fontsize=8)
            ax.set_xticks([0, 10, 30, 50])
            style(ax)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=5, frameon=False, fontsize=8.5)
    fig.savefig(out / "figures" / f"corruption_curves_{task}.png", dpi=150)
    plt.close(fig)


def fig_natural(nat, kind, out):
    n = nat[(nat.task == "binary") & (nat.kind == kind)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, constrained_layout=True)
    for ax, st in zip(axes, ["wrist", "all"]):
        x = np.arange(len(STRATEGIES))
        for k, (stratum, col) in enumerate([("PPG quality >= 0.5", C[0]), ("PPG quality < 0.5", C[1])]):
            vals = [n[(n.model == f"{st}:{s}") & (n.stratum == stratum)]["pooled_balanced_accuracy"].mean() for s in STRATEGIES]
            ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=col, label=stratum)
        ax.set_xticks(x, [STRAT_LABEL[s].replace("Late: ", "").replace(" (", "\n(") for s in STRATEGIES], fontsize=7.5)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f"{'Wrist only' if st == 'wrist' else 'Chest + wrist'}: windows split by wrist-PPG quality", fontsize=9,
                     loc="left", color=TEXT)
        style(ax)
    axes[0].set_ylabel("balanced accuracy (pooled windows)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "figures" / "natural_degradation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
