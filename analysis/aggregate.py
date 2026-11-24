"""Aggregate per-text features into the effect-size tables used in the paper.

Reads the per-text feature CSVs written by features/extract_features.py and writes, per
(dataset, model, feature):
  - effects.csv    : AUC / Cliff's delta vs human, moments, quantiles
  - human_stats.csv: human reference moments and quantiles
  - hist.csv       : fixed-bin histograms per (dataset, feature, group)
  - sentiment.csv  : sentiment share statistics per (dataset, source)

Runs on a CPU node. Memory footprint: one dataset at a time, float32.
"""

import os
import sys
import glob
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# set from the command line in __main__
ROOT = None   # folder written by features/extract_features.py
OUT = None    # output folder for the four tables

MAX_H = 200_000   # human subsample cap for AUC computation
MAX_POOL = 300_000  # pooled-AI subsample cap for AUC computation
N_BINS = 60
SEED = 1337

FAMILY = {
    "gpt-4.1-nano-2025-04-14": "GPT",
    "Llama-3.1-8B-Instruct": "Llama",
    "Meta-Llama-3.1-70B-Instruct-AWQ-INT4": "Llama",
    "Llama-3.2-3B-Instruct": "Llama",
    "Meta-Llama-3.3-70B-Instruct-AWQ-INT4": "Llama",
    "Phi-3-mini-128k-instruct": "Phi",
    "Phi-3-small-128k-instruct": "Phi",
    "Phi-3-medium-128k-instruct": "Phi",
    "Phi-3.5-mini-instruct": "Phi",
    "Phi-4-mini-instruct": "Phi",
    "phi-4": "Phi",
    "Phi-4": "Phi",
    "Ministral-8B-Instruct-2410": "Mistral",
    "Mistral-Nemo-Instruct-2407": "Mistral",
    "Qwen2-7B-Instruct": "Qwen",
    "Qwen2-72B-Instruct-AWQ": "Qwen",
    "Qwen2.5-3B-Instruct": "Qwen",
    "Qwen2.5-7B-Instruct": "Qwen",
    "Qwen2.5-14B-Instruct": "Qwen",
    "Qwen2.5-72B-Instruct-AWQ": "Qwen",
    "Falcon3-3B-Instruct": "Falcon",
    "Falcon3-7B-Instruct": "Falcon",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_features(path):
    df = pd.read_csv(path, engine="c")
    # drop unnamed trailing columns from stray trailing commas
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df.astype(np.float32)


def quantiles(a):
    qs = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(q05=qs[0], q25=qs[1], median=qs[2], q75=qs[3], q95=qs[4])


def moments(a):
    return dict(mean=float(np.mean(a)), std=float(np.std(a)), **quantiles(a))


def auc_vs_human(h, a, rng):
    """P(AI > human) + 0.5 P(AI == human) via Mann-Whitney U."""
    if len(h) > MAX_H:
        h = rng.choice(h, MAX_H, replace=False)
    if len(a) > MAX_POOL:
        a = rng.choice(a, MAX_POOL, replace=False)
    try:
        u, p = mannwhitneyu(a, h, alternative="two-sided")
        auc = float(u) / (len(a) * len(h))
        return auc, float(p)
    except ValueError:
        return np.nan, np.nan


def main():
    rng = np.random.default_rng(SEED)
    human_files = sorted(glob.glob(os.path.join(ROOT, "data_human", "*_features.csv")))
    datasets = [os.path.basename(f).replace("_human_features.csv", "") for f in human_files]
    log(f"datasets: {datasets}")

    eff_rows, hum_rows, hist_rows, sent_rows = [], [], [], []

    for ds, hf in zip(datasets, human_files):
        log(f"=== {ds}: loading human {hf}")
        H = load_features(hf)
        features = list(H.columns)

        # locate the AI dir for this dataset (hyphen/underscore variants)
        cand = [ds, ds.replace("-", "_"), ds.replace("_", "-")]
        ai_dir = None
        for c in cand:
            d = os.path.join(ROOT, "data_ai", c)
            if os.path.isdir(d):
                ai_dir = d
                break
        if ai_dir is None:
            log(f"WARNING: no AI dir for {ds}")
            continue

        ai_files = sorted(glob.glob(os.path.join(ai_dir, "*_features.csv")))
        log(f"{ds}: {len(ai_files)} AI files")

        # parse model names from file names
        ai_data = {}
        for f in ai_files:
            core = os.path.basename(f)[: -len("_features.csv")]
            model = None
            for c in cand:
                if core.startswith(c + "_"):
                    model = core[len(c) + 1:]
                    break
            if model is None:
                # fall back: strip up to first underscore
                model = core.split("_", 1)[1]
            ai_data[model] = load_features(f)

        # human stats + sentiment
        for feat in features:
            h = H[feat].dropna().values
            if len(h) == 0:
                continue
            hum_rows.append(dict(dataset=ds, feature=feat, n=len(h), **moments(h)))
        if "sentiment" in H.columns:
            s = H["sentiment"].dropna().values
            sent_rows.append(dict(
                dataset=ds, source="human", n=len(s),
                mean=float(np.mean(s)),
                share_pos=float(np.mean(s > 0.5)),
                share_neg=float(np.mean(s < -0.5)),
                share_zero=float(np.mean(np.abs(s) < 0.05)),
            ))

        # per-model effects + sentiment
        for model, A in ai_data.items():
            for feat in features:
                if feat not in A.columns:
                    continue
                h = H[feat].dropna().values
                a = A[feat].dropna().values
                if len(h) == 0 or len(a) == 0:
                    continue
                auc, p = auc_vs_human(h, a, rng)
                eff_rows.append(dict(
                    dataset=ds, model=model, family=FAMILY.get(model, "?"),
                    feature=feat, n=len(a), auc=auc, cliffs_delta=2 * auc - 1,
                    p_value=p, **moments(a),
                ))
            if "sentiment" in A.columns:
                s = A["sentiment"].dropna().values
                sent_rows.append(dict(
                    dataset=ds, source=model, n=len(s),
                    mean=float(np.mean(s)),
                    share_pos=float(np.mean(s > 0.5)),
                    share_neg=float(np.mean(s < -0.5)),
                    share_zero=float(np.mean(np.abs(s) < 0.05)),
                ))

        # pooled AI (all models) effects per feature
        for feat in features:
            h = H[feat].dropna().values
            pooled = np.concatenate([
                A[feat].dropna().values for A in ai_data.values() if feat in A.columns
            ]) if ai_data else np.array([])
            if len(h) == 0 or len(pooled) == 0:
                continue
            auc, p = auc_vs_human(h, pooled, rng)
            eff_rows.append(dict(
                dataset=ds, model="ALL_AI", family="ALL_AI", feature=feat,
                n=len(pooled), auc=auc, cliffs_delta=2 * auc - 1, p_value=p,
                **moments(pooled),
            ))

        # histograms: bins from robust range of human + pooled AI
        log(f"{ds}: histograms")
        for feat in features:
            h = H[feat].dropna().values
            pooled = np.concatenate([
                A[feat].dropna().values for A in ai_data.values() if feat in A.columns
            ]) if ai_data else np.array([])
            if len(h) == 0 or len(pooled) == 0:
                continue
            both = np.concatenate([
                rng.choice(h, min(len(h), 500_000), replace=False), pooled
            ])
            lo, hi = np.quantile(both, [0.005, 0.995])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                continue
            edges = np.linspace(lo, hi, N_BINS + 1)

            groups = {"human": h}
            for fam in ["GPT", "Llama", "Phi", "Mistral", "Qwen", "Falcon"]:
                vals = [A[feat].dropna().values for m, A in ai_data.items()
                        if FAMILY.get(m) == fam and feat in A.columns]
                if vals:
                    groups[fam] = np.concatenate(vals)
            groups["ALL_AI"] = pooled

            for gname, vals in groups.items():
                cnt, _ = np.histogram(np.clip(vals, lo, hi), bins=edges)
                hist_rows.append(dict(
                    dataset=ds, feature=feat, group=gname, n=len(vals),
                    lo=float(lo), hi=float(hi),
                    counts=";".join(str(int(c)) for c in cnt),
                ))

        del H, ai_data

    log("writing outputs")
    pd.DataFrame(eff_rows).to_csv(os.path.join(OUT, "effects.csv"), index=False)
    pd.DataFrame(hum_rows).to_csv(os.path.join(OUT, "human_stats.csv"), index=False)
    pd.DataFrame(hist_rows).to_csv(os.path.join(OUT, "hist.csv"), index=False)
    pd.DataFrame(sent_rows).to_csv(os.path.join(OUT, "sentiment.csv"), index=False)
    log("done")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate per-text features into effect-size tables.")
    ap.add_argument("--features", required=True, help="folder written by features/extract_features.py")
    ap.add_argument("--out", required=True, help="output folder (the released tables are in data/)")
    a = ap.parse_args()
    ROOT, OUT = a.features, a.out
    os.makedirs(OUT, exist_ok=True)
    main()
