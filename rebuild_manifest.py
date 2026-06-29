"""
rebuild_manifest.py
===================
Reconstruct manifest.csv from the run parquet files (e.g. if the original was
overwritten). Each parquet carries enough metadata to recover most fields.

NOTE: preprocessing/training do NOT need the manifest — preprocess.py and
class_summary.py read the parquet directly. This is only for restoring the record.

Two fields cannot be recovered (they were never written into the data):
  * seed  — the per-run random seed (so exact re-runs aren't reproducible from this)
  * the exact sensor-realism noise parameters (only their effects are stored;
    the dropout fraction per channel can be estimated from the _missing columns)

Usage:
  python rebuild_manifest.py --data data
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import pandas as pd


def rebuild_manifest(data_dir: str = "data") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "runs", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {data_dir}/runs/")
    rows = []
    for f in files:
        df = pd.read_parquet(f)
        sev = (df["gt_severity"].to_numpy(float)
               if "gt_severity" in df.columns else np.zeros(len(df)))
        active = sev > 0
        onset = int(np.argmax(active)) if active.any() else 0
        smax = float(sev.max())
        reach = int(np.argmax(sev >= smax)) if active.any() else 0
        ramp_min = max(0, reach - onset) if active.any() else 0
        lab = df["label"].to_numpy()
        conds = sorted({c.split("__", 1)[1] for c in df.columns
                        if c.startswith("flow__") and not c.endswith(("_meas", "_missing"))})
        # estimate per-channel dropout from _missing columns (realism effects)
        miss = {c[:-8]: round(float(df[c].mean()), 4)
                for c in df.columns if c.endswith("_missing")}
        rows.append(dict(
            run_id=df["run_id"].iloc[0],
            scenario=df["scenario"].iloc[0],
            target=df["target_conduit"].iloc[0],
            duration_h=round(len(df) / 60, 2),
            final_sev=round(smax, 4),
            onset_min=onset,
            ramp_min=ramp_min,
            ramp_type=("instant" if ramp_min <= 1 else "gradual") if active.any() else "instant",
            antecedent_dry_days=(int(df["ctx_antecedent_dry_days"].iloc[0])
                                 if "ctx_antecedent_dry_days" in df.columns else None),
            antecedent_precip_index=(float(df["ctx_antecedent_precip_index"].iloc[0])
                                     if "ctx_antecedent_precip_index" in df.columns else None),
            n_rows=len(df),
            n_normal=int((lab == "normal").sum()),
            n_rainfall=int((lab == "rainfall").sum()),
            n_blockage=int((lab == "blockage").sum()),
            sensor_conduits=";".join(conds),
            dropout_fraction_est=str(miss) if miss else "",
            seed=None,                                   # NOT recoverable from data
            parquet=os.path.relpath(f, data_dir),
        ))
    m = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    out = os.path.join(data_dir, "manifest.csv")
    m.to_csv(out, index=False)
    print(f"rebuilt manifest: {len(m)} runs -> {out}")
    print("  (note: 'seed' and exact realism noise params are not recoverable)")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder containing runs/*.parquet")
    rebuild_manifest(ap.parse_args().data)
