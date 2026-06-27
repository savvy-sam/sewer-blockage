"""
preprocess.py
=============
Leakage-controlled preprocessing for the sewer-blockage dataset, implementing the
rules in Methodology files/Leakage_Control_and_External_Validation.docx plus
standard practice. Produces BOTH a tree-ready feature table and sliding-window
tensors for sequence models, from one shared, grouped split.

Leakage controls (from the methodology):
  * Run-level grouped split — every run goes wholly to one split; rows never shared.
  * Stratified by scenario so all classes appear in every split.
  * Two OOD test sets: held-out injection nodes AND a held-out rainfall band,
    reported separately from the in-distribution test.
  * Fit-on-train-only: scaler stats, class weights, imputation fills use TRAIN rows
    only, then applied to val/test/ood.
  * Knob/onset decorrelation: gt_severity, gt_setting, antecedent ctx_*, absolute
    time (t_min/timestamp) are EXCLUDED from features. Inputs are restricted to
    quantities a flow meter + ultrasonic + rain gauge could produce.
  * Audit tests: group-disjointness, forbidden-feature, deterministic-of-label,
    plus reported knob-correlation and optional shuffle-label / mechanism probes.

Standard practice added: dropout handled via the realism _missing flags + causal
forward-fill; causal feature engineering (depth-to-flow ratio, rate-of-change,
rolling stats, neighbour aggregates, spatial spread); per-feature standardisation;
class weights.

Usage:
  python preprocess.py --data data --out processed --window 60 --stride 10
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

EPS = 1e-6
RAIN_GAGES = ["rg5425", "rg5427"]
# never features (knob / schedule / target / metadata)
META = {"label", "label_id", "run_id", "scenario", "target_conduit", "t_min",
        "gt_severity", "split"}
FORBIDDEN = {"gt_severity", "gt_setting", "ctx_antecedent_dry_days",
             "ctx_antecedent_precip_index", "t_min", "timestamp",
             "label", "label_id", "run_id", "scenario", "target_conduit"}


# --------------------------------------------------------------------------- #
# Per-run feature construction (causal; consistent columns across runs)
# --------------------------------------------------------------------------- #
def build_run_features(df: pd.DataFrame, roll: int = 15, diff_lags=(1, 5)) -> pd.DataFrame:
    df = df.sort_values("t_min").reset_index(drop=True)
    tgt = df["target_conduit"].iloc[0]
    n = len(df)
    out = pd.DataFrame(index=df.index)

    def pick(base):  # prefer the measured (noisy) channel where it exists
        m = base + "_meas"
        if m in df.columns:
            return df[m]
        return df[base] if base in df.columns else pd.Series(np.nan, index=df.index)

    # primary conduit (the deployed flow-meter location)
    out["p_flow"] = pick(f"flow__{tgt}")
    out["p_depth"] = df.get(f"depth__{tgt}", pd.Series(np.nan, index=df.index))
    out["p_vel"] = pick(f"vel__{tgt}")
    out["p_ushear"] = pick(f"ushear__{tgt}")
    out["p_fill"] = df.get(f"fill__{tgt}", pd.Series(np.nan, index=df.index))
    out["p_froude"] = df.get(f"froude__{tgt}", pd.Series(np.nan, index=df.index))

    # primary node depth (ultrasonic sensor); recover node id from the _meas column
    node_meas = [c for c in df.columns if c.startswith("depth__node_") and c.endswith("_meas")]
    if node_meas:
        out["p_node_depth"] = df[node_meas[0]]
        prim_node_base = node_meas[0][:-5]
    else:
        ncols = [c for c in df.columns if c.startswith("depth__node_")]
        out["p_node_depth"] = df[ncols[0]] if ncols else pd.Series(np.nan, index=df.index)
        prim_node_base = ncols[0] if ncols else None

    # neighbour aggregates (spatial context, order-independent so it's run-consistent)
    nb_depth = [c for c in df.columns if c.startswith("depth__") and
                not c.startswith("depth__node_") and c != f"depth__{tgt}"]
    nb_flow = [c for c in df.columns if c.startswith("flow__") and c != f"flow__{tgt}"
               and not c.endswith(("_meas", "_missing"))]
    nb_vel = [c for c in df.columns if c.startswith("vel__") and c != f"vel__{tgt}"
              and not c.endswith(("_meas", "_missing"))]
    out["nb_depth_mean"] = df[nb_depth].mean(axis=1) if nb_depth else 0.0
    out["nb_depth_max"] = df[nb_depth].max(axis=1) if nb_depth else 0.0
    out["nb_flow_mean"] = df[nb_flow].mean(axis=1) if nb_flow else 0.0
    out["nb_vel_mean"] = df[nb_vel].mean(axis=1) if nb_vel else 0.0
    node_all = [c for c in df.columns if c.startswith("depth__node_")
                and not c.endswith(("_meas", "_missing"))]
    out["node_depth_spread"] = (df[node_all].max(axis=1) - df[node_all].min(axis=1)
                                if node_all else 0.0)

    # rainfall (gauge — allowed observable)
    for g in RAIN_GAGES:
        out[f"rain_{g}"] = df.get(f"rain__{g}", 0.0)
    out["rain_mean"] = out[[f"rain_{g}" for g in RAIN_GAGES]].mean(axis=1)

    # dropout / missing-data indicator flags (these ARE features)
    for base, flag in [(f"flow__{tgt}", "p_flow_missing"), (f"vel__{tgt}", "p_vel_missing"),
                       (f"ushear__{tgt}", "p_ushear_missing")]:
        out[flag] = df[base + "_missing"] if base + "_missing" in df.columns else 0
    out["p_node_depth_missing"] = (df[prim_node_base + "_missing"]
                                   if prim_node_base and prim_node_base + "_missing" in df.columns
                                   else 0)

    # causal forward-fill of the physical sensor channels (sensor holds last reading)
    sensor = ["p_flow", "p_vel", "p_ushear", "p_node_depth"]
    out[sensor] = out[sensor].ffill()

    # engineered features (all causal: use current row + its own past only)
    out["dtf_ratio"] = out["p_node_depth"] / (out["p_flow"].abs() + EPS)   # core feature
    for lag in diff_lags:
        out[f"d{lag}_node_depth"] = out["p_node_depth"].diff(lag)
        out[f"d{lag}_flow"] = out["p_flow"].diff(lag)
        out[f"d{lag}_ushear"] = out["p_ushear"].diff(lag)
    out["roll_mean_depth"] = out["p_node_depth"].rolling(roll, min_periods=1).mean()
    out["roll_std_depth"] = out["p_node_depth"].rolling(roll, min_periods=1).std()
    out["roll_mean_flow"] = out["p_flow"].rolling(roll, min_periods=1).mean()

    out = out.fillna(0.0)

    # metadata for splitting / audit (NOT features)
    out["label_id"] = df["label_id"].values
    out["label"] = df["label"].values
    out["run_id"] = df["run_id"].values
    out["scenario"] = df["scenario"].values
    out["target_conduit"] = tgt
    out["t_min"] = df["t_min"].values
    out["gt_severity"] = df["gt_severity"].values if "gt_severity" in df.columns else 0.0
    return out


def feature_columns(table: pd.DataFrame) -> list:
    return [c for c in table.columns if c not in META]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "runs", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {data_dir}/runs/")
    feats, meta_rows = [], []
    for f in files:
        df = pd.read_parquet(f)
        feats.append(build_run_features(df))
        rain_cols = [c for c in df.columns if c.startswith("rain__")]
        total_rain = float(df[rain_cols].to_numpy().sum()) if rain_cols else 0.0
        meta_rows.append(dict(run_id=df["run_id"].iloc[0], scenario=df["scenario"].iloc[0],
                              target_conduit=df["target_conduit"].iloc[0], total_rain=total_rain))
    return pd.concat(feats, ignore_index=True), pd.DataFrame(meta_rows)


# --------------------------------------------------------------------------- #
# Grouped, stratified split + two OOD hold-outs
# --------------------------------------------------------------------------- #
def make_splits(meta_runs: pd.DataFrame, seed=0, heldout_node_frac=0.2,
                n_rain_bins=4, val_frac=0.15, test_frac=0.15, min_per_split_class=1):
    rng = np.random.default_rng(seed)
    runs = meta_runs.copy()
    runs["split"] = ""

    # OOD #1: held-out injection nodes (spatial generalisation)
    nodes = runs["target_conduit"].unique().tolist()
    rng.shuffle(nodes)
    n_hold = max(1, int(round(len(nodes) * heldout_node_frac)))
    ood_nodes = set(nodes[:n_hold])
    runs.loc[runs.target_conduit.isin(ood_nodes), "split"] = "ood_node"

    # OOD #2: held-out top rainfall band (storm-severity generalisation), from the rest
    rem = runs[runs.split == ""]
    rainy = rem[rem.total_rain > 0]
    ood_rain_ids = set()
    if len(rainy) >= n_rain_bins:
        bins = pd.qcut(rainy["total_rain"].rank(method="first"), n_rain_bins, labels=False)
        top = bins.max()
        ood_rain_ids = set(rainy.loc[bins == top, "run_id"])
        runs.loc[runs.run_id.isin(ood_rain_ids), "split"] = "ood_rain"

    # remaining -> train/val/test, stratified by scenario, grouped by run
    rem2 = runs[runs.split == ""]
    for scn, grp in rem2.groupby("scenario"):
        ids = grp.run_id.tolist()
        rng.shuffle(ids)
        n = len(ids)
        nt, nv = int(round(n * test_frac)), int(round(n * val_frac))
        nt = min(nt, max(0, n - 1))                       # keep >=1 in train if possible
        test, val = set(ids[:nt]), set(ids[nt:nt + nv])
        train = set(ids[nt + nv:])
        for s, idset in [("test", test), ("val", val), ("train", train)]:
            runs.loc[runs.run_id.isin(idset), "split"] = s

    # --- class-coverage guarantee: ensure val & test each hold >= min runs that
    #     contain blockage and rainfall, pulling spares from train (OOD untouched) ---
    flag_cols = [c for c in ("has_blockage", "has_rainfall") if c in runs.columns]
    for tgt_split in ("val", "test"):
        for flag in flag_cols:
            while int(((runs.split == tgt_split) & runs[flag]).sum()) < min_per_split_class:
                pool = runs[(runs.split == "train") & runs[flag]]
                if pool.empty:
                    break                       # can't satisfy from train; caught by guard
                pick = rng.choice(pool["run_id"].to_numpy())
                runs.loc[runs.run_id == pick, "split"] = tgt_split

    return dict(zip(runs.run_id, runs.split)), runs


def assert_class_coverage(table, allow_incomplete=False):
    """Fail loudly if any in-distribution split (train/val/test) is missing a class
    that exists in the dataset. OOD sets are exempt (they are deliberate regimes)."""
    all_classes = set(table["label"].unique())
    problems = []
    for s in ("train", "val", "test"):
        present = set(table.loc[table.split == s, "label"].unique())
        missing = all_classes - present
        if table[table.split == s].empty:
            problems.append(f"split '{s}' is EMPTY")
        elif missing:
            problems.append(f"split '{s}' is missing class(es): {sorted(missing)}")
    if problems:
        msg = ("SPLIT COVERAGE FAILURE — not every class reaches every in-distribution "
               "split:\n  - " + "\n  - ".join(problems) +
               "\n  Fix: generate more runs (esp. blockage scenarios S3/S4/S6) so each "
               "split can hold each class. Re-run, or pass --allow-incomplete-splits to "
               "proceed anyway (results will be uninterpretable for the missing class).")
        if allow_incomplete:
            print("WARNING: " + msg)
        else:
            raise ValueError(msg)


# --------------------------------------------------------------------------- #
# Fit-on-train-only transforms
# --------------------------------------------------------------------------- #
def fit_scaler(table, feat_cols):
    tr = table[table.split == "train"]
    if len(tr) == 0:
        raise ValueError("no training rows - check the split / number of runs")
    mean = tr[feat_cols].mean()
    std = tr[feat_cols].std(ddof=0).replace(0, 1.0)
    return mean, std


def apply_scaler(table, feat_cols, mean, std):
    table = table.copy()
    table[feat_cols] = (table[feat_cols] - mean) / std
    return table


def class_weights(table):
    tr = table[table.split == "train"]
    vc = tr["label_id"].value_counts()
    n, k = len(tr), len(vc)
    return {int(c): float(n / (k * cnt)) for c, cnt in vc.items()}


# --------------------------------------------------------------------------- #
# Sliding windows (per run -> never cross a run / split boundary)
# --------------------------------------------------------------------------- #
def build_windows(table, feat_cols, window=60, stride=10):
    Xs, ys, gids, spl = [], [], [], []
    for rid, g in table.groupby("run_id"):
        g = g.sort_values("t_min")
        arr = g[feat_cols].to_numpy(np.float32)
        y = g["label_id"].to_numpy()
        s = g["split"].iloc[0]
        for st in range(0, len(g) - window + 1, stride):
            Xs.append(arr[st:st + window])
            ys.append(int(y[st + window - 1]))            # label at window end (causal)
            gids.append(rid); spl.append(s)
    if not Xs:
        return (np.empty((0, window, len(feat_cols)), np.float32),
                np.empty((0,), int), np.array([]), np.array([]))
    return np.stack(Xs), np.array(ys), np.array(gids), np.array(spl)


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
def run_audit(table, feat_cols):
    issues, knob = [], []
    # group-disjointness
    sets = {s: set(table[table.split == s].run_id) for s in table.split.unique()}
    ks = list(sets)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if sets[ks[i]] & sets[ks[j]]:
                issues.append(f"RUN OVERLAP between {ks[i]} and {ks[j]}")
    # forbidden features
    leaked = [c for c in feat_cols if c in FORBIDDEN]
    if leaked:
        issues.append(f"FORBIDDEN features present: {leaked}")
    # deterministic-of-label
    tr = table[table.split == "train"]
    for c in feat_cols:
        if tr[c].nunique() > 1:
            corr = np.corrcoef(tr[c], tr["label_id"])[0, 1]
            if np.isfinite(corr) and abs(corr) > 0.999:
                issues.append(f"feature {c} is ~deterministic of the label")
    # knob correlation (REPORTED, not failed: high is expected for genuine hydraulic
    # features; only a concern for features that shouldn't carry the signal)
    if "gt_severity" in table and tr["gt_severity"].nunique() > 1:
        for c in feat_cols:
            if tr[c].nunique() > 1:
                k = np.corrcoef(tr[c], tr["gt_severity"])[0, 1]
                if np.isfinite(k) and abs(k) > 0.6:
                    knob.append((c, round(float(abs(k)), 3)))
        knob.sort(key=lambda x: -x[1])
    return issues, knob


def shuffle_label_control(X, y, seed=0):
    """Optional: train a quick classifier on PERMUTED labels; performance must
    collapse to the class prior. Needs scikit-learn."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    yp = rng.permutation(y)
    Xf = X.reshape(len(X), -1) if X.ndim == 3 else X
    m = LogisticRegression(max_iter=200, multi_class="auto").fit(Xf, yp)
    return float(balanced_accuracy_score(yp, m.predict(Xf)))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="processed")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--roll", type=int, default=15)
    ap.add_argument("--heldout-node-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--min-split-class", type=int, default=1,
                    help="min runs containing each of blockage/rainfall in val & test")
    ap.add_argument("--allow-incomplete-splits", action="store_true",
                    help="downgrade the class-coverage failure to a warning")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    table, meta_runs = load(args.data)
    # per-run class presence (computed from the actual labels, not the scenario type)
    pres = (table.assign(_b=table.label == "blockage", _r=table.label == "rainfall")
            .groupby("run_id")[["_b", "_r"]].any().rename(
                columns={"_b": "has_blockage", "_r": "has_rainfall"}))
    meta_runs = meta_runs.merge(pres, on="run_id", how="left")

    split_map, runs_split = make_splits(meta_runs, args.seed, args.heldout_node_frac,
                                        val_frac=args.val_frac, test_frac=args.test_frac,
                                        min_per_split_class=args.min_split_class)
    table["split"] = table.run_id.map(split_map)
    assert_class_coverage(table, allow_incomplete=args.allow_incomplete_splits)
    feat_cols = feature_columns(table)

    mean, std = fit_scaler(table, feat_cols)              # TRAIN ONLY
    scaled = apply_scaler(table, feat_cols, mean, std)
    cw = class_weights(table)
    issues, knob = run_audit(scaled, feat_cols)

    # ---- save tabular (drop gt_severity so it can't be used downstream) ----
    keep = feat_cols + ["label_id", "label", "run_id", "scenario", "target_conduit",
                        "t_min", "split"]
    scaled[keep].to_parquet(os.path.join(args.out, "feature_table.parquet"), index=False)

    # ---- save windows per split ----
    X, y, gids, spl = build_windows(scaled, feat_cols, args.window, args.stride)
    for s in ["train", "val", "test", "ood_node", "ood_rain"]:
        m = spl == s
        np.savez_compressed(os.path.join(args.out, f"windows_{s}.npz"),
                            X=X[m], y=y[m], run_ids=gids[m])

    # ---- save artefacts ----
    json.dump({"features": feat_cols, "mean": mean[feat_cols].tolist(),
               "std": std[feat_cols].tolist(), "window": args.window,
               "stride": args.stride},
              open(os.path.join(args.out, "scaler.json"), "w"), indent=1)
    json.dump(cw, open(os.path.join(args.out, "class_weights.json"), "w"), indent=1)
    runs_split.to_csv(os.path.join(args.out, "split_manifest.csv"), index=False)

    # ---- report ----
    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"runs: {len(meta_runs)}   rows: {len(table)}   features: {len(feat_cols)}")
    print("\nrun-level split:")
    print(runs_split["split"].value_counts().to_string())
    print("\nrows per split:")
    print(table["split"].value_counts().to_string())
    print("\nclass balance per split (timesteps):")
    print(pd.crosstab(table["split"], table["label"]).to_string())
    print(f"\nwindows: total={len(X)}  " +
          "  ".join(f"{s}={(spl==s).sum()}" for s in
                    ['train', 'val', 'test', 'ood_node', 'ood_rain']))
    print("\nclass weights (train):", cw)
    print("\ntop knob (gt_severity) correlations [expected high for real hydraulic "
          "features; concern only for ones that shouldn't carry signal]:")
    print("  " + (", ".join(f"{c}:{v}" for c, v in knob[:8]) if knob else "none > 0.6"))
    print("\n" + "=" * 60)
    if issues:
        print("LEAKAGE AUDIT FAILURES:")
        for i in issues:
            print("  -", i)
    else:
        print("LEAKAGE AUDIT: all hard checks passed "
              "(disjoint splits, no forbidden features, none deterministic-of-label)")
    print("=" * 60)
    print(f"written -> {args.out}/ (feature_table.parquet, windows_*.npz, scaler.json, "
          "class_weights.json, split_manifest.csv)")


if __name__ == "__main__":
    main()
