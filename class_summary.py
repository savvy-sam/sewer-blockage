"""
class_summary.py
================
Coverage / class-balance check for a generated dataset. The generator does NOT
rebalance classes (normal dominates, by design); this report makes the imbalance
visible and — more importantly — confirms there is enough ABSOLUTE minority-class
coverage (blockage / rainfall) across severities, injection locations, ramp types
and scenarios before training.

Self-sufficient: reads only data/runs/*.parquet (columns: label, scenario,
run_id, target_conduit, gt_severity). ramp_type is inferred from the severity
trajectory, so no manifest is required.

Colab cell:
    from class_summary import summarize
    summarize("data")            # prints the report, returns a dict, writes data/class_summary.csv
CLI:
    python class_summary.py --data data
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import pandas as pd

CLASSES = ["normal", "rainfall", "blockage"]
SEV_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]            # area-fraction blocked
SEV_LABELS = ["<20%", "20-40%", "40-60%", "60-80%", "80-90%+"]


def _count_events(mask: np.ndarray) -> int:
    """Number of maximal contiguous True segments (rising edges)."""
    a = mask.astype(int)
    return int((np.diff(np.concatenate([[0], a])) == 1).sum())


def _infer_ramp(sev: np.ndarray) -> str | None:
    active = sev > 0
    if not active.any():
        return None
    onset = int(np.argmax(active))
    smax = sev.max()
    reach = int(np.argmax(sev >= smax))
    return "instant" if (reach - onset) <= 1 else "gradual"


def summarize(data_dir: str = "data", verbose: bool = True) -> dict:
    files = sorted(glob.glob(os.path.join(data_dir, "runs", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet runs under {data_dir}/runs/")

    tot = {c: 0 for c in CLASSES}                 # total timesteps per class
    ev = {c: 0 for c in CLASSES}                  # total events per class
    runs_with = {c: 0 for c in CLASSES}           # runs containing >=1 of class
    per_run = []
    blk_locations = {}                            # conduit -> n blockage runs
    sev_finals, ramp_counts = [], {"instant": 0, "gradual": 0}
    by_scn_steps = {}                             # scenario -> {class: steps}
    rain_steps_per_run = []

    for f in files:
        df = pd.read_parquet(f, columns=["label", "scenario", "gt_severity",
                                         "target_conduit", "run_id"])
        labels = df["label"].to_numpy()
        scn = str(df["scenario"].iloc[0])
        sev = df["gt_severity"].to_numpy(dtype=float)
        target = str(df["target_conduit"].iloc[0])
        by_scn_steps.setdefault(scn, {c: 0 for c in CLASSES})
        rec = {"run_id": str(df["run_id"].iloc[0]), "scenario": scn}
        for c in CLASSES:
            m = labels == c
            n = int(m.sum())
            tot[c] += n
            ev[c] += _count_events(m)
            runs_with[c] += int(n > 0)
            by_scn_steps[scn][c] += n
            rec[c] = n
        per_run.append(rec)
        rain_steps_per_run.append(int((labels == "rainfall").sum()))
        if (labels == "blockage").any():
            blk_locations[target] = blk_locations.get(target, 0) + 1
            sev_finals.append(float(sev.max()))
            rt = _infer_ramp(sev)
            if rt:
                ramp_counts[rt] += 1

    grand = sum(tot.values())
    sev_arr = np.array(sev_finals) if sev_finals else np.array([])
    sev_hist = (pd.cut(sev_arr, SEV_BINS, labels=SEV_LABELS, right=False)
                .value_counts().reindex(SEV_LABELS).fillna(0).astype(int)
                if len(sev_arr) else pd.Series(0, index=SEV_LABELS))
    singletons = sorted([k for k, v in blk_locations.items() if v == 1])

    summary = {
        "n_runs": len(files),
        "timesteps_total": grand,
        "timesteps_per_class": tot,
        "pct_per_class": {c: (100 * tot[c] / grand if grand else 0) for c in CLASSES},
        "events_per_class": ev,
        "runs_with_class": runs_with,
        "blockage_locations": len(blk_locations),
        "blockage_location_singletons": len(singletons),
        "severity_hist": sev_hist.to_dict(),
        "ramp_counts": ramp_counts,
        "by_scenario_steps": by_scn_steps,
    }

    if verbose:
        _print_report(summary, sev_arr, rain_steps_per_run, singletons)

    # persist tidy tables
    pd.DataFrame([{"class": c, "timesteps": tot[c],
                   "pct": round(summary["pct_per_class"][c], 3),
                   "events": ev[c], "runs_with": runs_with[c]} for c in CLASSES]
                 ).to_csv(os.path.join(data_dir, "class_summary.csv"), index=False)
    pd.DataFrame(per_run).to_csv(os.path.join(data_dir, "class_counts_per_run.csv"),
                                 index=False)
    return summary


def _bar(p, width=28):
    return "█" * int(round(p / 100 * width))


def _print_report(s, sev_arr, rain_steps_per_run, singletons):
    L = "─" * 60
    print(L); print("DATASET COVERAGE / CLASS-BALANCE SUMMARY"); print(L)
    print(f"runs: {s['n_runs']:>6}     total timesteps: {s['timesteps_total']:,}\n")
    print("Per-class timesteps (imbalance is expected — normal dominates):")
    for c in CLASSES:
        n, p = s["timesteps_per_class"][c], s["pct_per_class"][c]
        print(f"  {c:<9} {n:>12,}  {p:6.2f}%  {_bar(p)}")
    nb = s["timesteps_per_class"]["blockage"]
    if nb:
        ratio = s["timesteps_per_class"]["normal"] / nb
        print(f"\n  normal : blockage  ≈  {ratio:,.0f} : 1")
    print("\nEvents (contiguous segments) and run coverage:")
    for c in CLASSES:
        print(f"  {c:<9} events={s['events_per_class'][c]:>5}   "
              f"runs containing={s['runs_with_class'][c]:>4}")
    print("\nBlockage coverage:")
    print(f"  distinct injection locations : {s['blockage_locations']} "
          f"({s['blockage_location_singletons']} used only once)")
    if len(sev_arr):
        print(f"  final severity  min/median/max : "
              f"{sev_arr.min():.2f} / {np.median(sev_arr):.2f} / {sev_arr.max():.2f}")
    print(f"  severity bins  : " +
          "  ".join(f"{k}:{v}" for k, v in s["severity_hist"].items()))
    print(f"  ramp type      : instant={s['ramp_counts']['instant']}  "
          f"gradual={s['ramp_counts']['gradual']}")
    if rain_steps_per_run:
        rs = np.array(rain_steps_per_run)
        print("\nRainfall coverage:")
        print(f"  runs with rainfall timesteps : {(rs > 0).sum()}/{len(rs)}   "
              f"median rainfall min/run (rainy runs): "
              f"{int(np.median(rs[rs > 0])) if (rs > 0).any() else 0}")
    print("\nBy scenario (timesteps):")
    for scn, d in sorted(s["by_scenario_steps"].items()):
        print(f"  {scn:<6} " + "  ".join(f"{c}={d[c]:,}" for c in CLASSES))

    # ---- warnings ----
    warn = []
    for c in ("rainfall", "blockage"):
        if s["timesteps_per_class"][c] == 0:
            warn.append(f"NO {c} timesteps generated.")
        elif s["events_per_class"][c] < 20:
            warn.append(f"only {s['events_per_class'][c]} {c} events — thin minority coverage.")
    if s["blockage_locations"] and s["blockage_locations"] < 10:
        warn.append(f"blockage injected at only {s['blockage_locations']} locations "
                    f"— raise --top-k-targets / --n-per-scenario for location diversity.")
    empty_bins = [k for k, v in s["severity_hist"].items() if v == 0]
    if empty_bins and s["timesteps_per_class"]["blockage"]:
        warn.append(f"empty severity bins: {', '.join(empty_bins)}.")
    print("\n" + L)
    if warn:
        print("⚠  COVERAGE WARNINGS:")
        for w in warn:
            print(f"   - {w}")
    else:
        print("✓ no coverage warnings.")
    print(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    summarize(ap.parse_args().data)
