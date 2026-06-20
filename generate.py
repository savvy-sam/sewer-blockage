"""
generate.py
===========
Core-4 sewer-blockage data generator for the Bellinge SWMM model.

Scenarios (per PySWMM_Scenarios_Revised.docx, Part 1):
  1 Baseline Normal        7 d     dry, diurnal only            -> normal
  2 Pure Rainfall Rise     48-72 h 24h dry -> storm -> recovery -> rainfall
  3 Dry Weather Blockage   48-72 h blockage ramps hrs 24-36     -> blockage
  4 Wet Weather Blockage   5-7 d   blockage forms, storm ~1d on -> blockage

Per-run randomisation (Part 2 axes that apply to Core-4): blockage severity,
instant vs gradual ramp, injection node, rainfall intensity/duration, rainfall
spatial heterogeneity, antecedent dry-weather duration + antecedent precip index.
Onset time and injection node are randomised so timing/location are not learnable
artefacts. A controllable inline orifice is inserted in EVERY run (held open when
no blockage) to keep topology decorrelated from the label.

Labelling priority: blockage > rainfall-driven > normal.
  * blockage  : severity(t) >= ONSET_SEVERITY (explicit, constant threshold)
  * rainfall  : response-based — a storm has begun AND sensor depth exceeds its
                diurnal dry-weather baseline by a margin (rise + recession until
                depth recovers), when not blockage
  * normal    : everything else (dry-weather diurnal flow, incl. DWF peaks)
Metrics for instant vs gradual blockage should be reported separately downstream
(the ramp type is recorded in the manifest).

Outputs (per run, written under --out):
  runs/<run_id>.parquet   wide 1-min time series (clean + _meas/_missing)
  manifest.csv            one row per run with all sampled parameters

Usage (Colab):
  !pip -q install pyswmm pandas pyarrow numpy
  !python generate.py --inp BellingeSWMM_v021_nopervious.inp \
                      --targets blockage_targets.csv --out data --n-per-scenario 5
(Set --routing-step / --report-step to trade fidelity for speed.)
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from pyswmm import Simulation, Links, Nodes

import swmm_inp as S
from rainfall import (Storm, build_intensity_series, write_rain_dat,
                      sample_storm, gage_multipliers)
from hydraulics import channel_features, circular_geometry
from sensor_realism import apply_realism

ONSET_SEVERITY = 0.10            # area fraction removed that counts as "blockage onset"
RAIN_REL_MARGIN = 0.10           # sensor depth must exceed dry baseline by >10% ...
RAIN_ABS_MARGIN_M = 0.02         # ... and by >=0.02 m (absolute floor for small pipes)
RAIN_BIN_MIN = 30                # time-of-day bin (min) for the diurnal dry baseline
LABELS = {"normal": 0, "rainfall": 1, "blockage": 2}
BASE_START = dt.datetime(2012, 6, 29, 0, 1)   # arbitrary dry anchor in the model calendar


# --------------------------------------------------------------------------- #
# Sensor network selection
# --------------------------------------------------------------------------- #
def select_sensor_conduits(model: S.InpModel, target: str, k_hops: int = 2) -> list:
    """Target conduit + conduits within k hops up/downstream (spatial context)."""
    conduits = model.conduits()
    graph = model.conduit_graph()
    c = conduits[target]
    seen_nodes = {c["n1"], c["n2"]}
    frontier = {c["n1"], c["n2"]}
    sensor_conduits = {target}
    for _ in range(k_hops):
        nxt = set()
        for node in frontier:
            for nb, cond, _dir in graph.get(node, []):
                sensor_conduits.add(cond)
                if nb not in seen_nodes:
                    seen_nodes.add(nb)
                    nxt.add(nb)
        frontier = nxt
    # keep only conduits we have a diameter for (circular)
    diam = model.xsection_diam()
    return [c for c in sorted(sensor_conduits)
            if diam.get(c, {}).get("shape") == "CIRCULAR" and diam[c]["geom1"] > 0]


# --------------------------------------------------------------------------- #
# Severity schedule + labels
# --------------------------------------------------------------------------- #
def severity_at(t_min, onset_min, ramp_min, final_sev):
    if final_sev <= 0 or t_min < onset_min:
        return 0.0
    if ramp_min <= 0 or t_min >= onset_min + ramp_min:
        return final_sev
    return final_sev * (t_min - onset_min) / ramp_min


def _diurnal_dry_baseline(depth, tod_min, dry_mask, bin_min):
    """Expected dry-weather sensor depth as a function of time-of-day.

    Estimated from this run's own dry, unblocked timesteps (median per
    time-of-day bin) so the diurnal DWF cycle is not mistaken for a rain rise.
    Empty bins fall back to the global dry median.
    """
    nbins = (1440 + bin_min - 1) // bin_min
    bin_idx = (tod_min // bin_min).astype(int)
    base = np.full(nbins, np.nan)
    for b in range(nbins):
        vals = depth[dry_mask & (bin_idx == b)]
        if len(vals):
            base[b] = np.median(vals)
    if np.all(np.isnan(base)):
        base[:] = np.median(depth) if len(depth) else 0.0
    else:
        base = np.where(np.isnan(base), np.nanmedian(base), base)
    return base[bin_idx]


def rainfall_response_mask(depth, tod_min, intensity, blk_mask,
                           rel_margin, abs_margin, bin_min):
    """Rainfall-driven label: a storm has begun AND the sensor depth exceeds its
    diurnal dry-weather baseline by the margin (captures the rise + recession
    until depth recovers). Blockage timesteps are excluded by the caller."""
    rain_active = intensity > 0
    storm_started = np.cumsum(rain_active) > 0          # True from first rain onward
    dry = (~rain_active) & (~blk_mask)                  # baseline-eligible timesteps
    baseline = _diurnal_dry_baseline(depth, tod_min, dry, bin_min)
    threshold = baseline * (1.0 + rel_margin) + abs_margin
    excess = depth > threshold
    return storm_started & excess & (~blk_mask)


# --------------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------------- #
def run_one(base_model: S.InpModel, base_inp_path: str, params: dict, out_dir: str) -> dict:
    """Execute one simulation, label it, write parquet, return a manifest row."""
    rng = np.random.default_rng(params["seed"])
    n_min = params["duration_h"] * 60
    start = BASE_START
    end = start + dt.timedelta(minutes=n_min)

    # --- inject blockage orifice (held open if final_sev == 0) ---
    text, blk = S.inject_inline_orifice(base_model, params["target"])
    text = S.set_simulation_window(text, start, end,
                                   report_step_s=params["report_step_s"],
                                   routing_step_s=params["routing_step_s"])

    # --- rainfall ---
    gages = S.raingage_names(base_model)
    storms = [Storm(**s) for s in params["storms"]]
    basin_series = build_intensity_series(n_min, storms)
    mults = params["gage_mults"]
    gage_series = {g: basin_series * mults.get(g, 1.0) for g in gages}
    rundir = tempfile.mkdtemp(prefix="swmmrun_")
    dat_name = "rain_run.dat"
    write_rain_dat(os.path.join(rundir, dat_name), start, gage_series)
    text = S.set_rain_file(text, dat_name)
    inp_path = os.path.join(rundir, "run.inp")
    with open(inp_path, "w") as fh:
        fh.write(text)

    # --- sensor set ---
    sensor_conduits = select_sensor_conduits(base_model, params["target"],
                                             k_hops=params["k_hops"])
    diam = base_model.xsection_diam()
    rough = {c["name"]: c["rough"] for c in base_model.conduits().values()}
    sensor_nodes = sorted({base_model.conduits()[c]["n1"] for c in sensor_conduits} |
                          {base_model.conduits()[c]["n2"] for c in sensor_conduits})

    # --- simulate, sampling at 1-min ---
    rows = []
    severities = []
    with Simulation(inp_path) as sim:
        links, nodes = Links(sim), Nodes(sim)
        orf = links[blk.orifice_name]
        sim.step_advance(60)
        i = 0
        for _ in sim:
            t_min = i
            sev = severity_at(t_min, params["onset_min"], params["ramp_min"],
                              params["final_sev"])
            orf.target_setting = S.severity_to_setting(sev)
            severities.append(sev)
            row = {"t_min": t_min,
                   "timestamp": (start + dt.timedelta(minutes=t_min)).isoformat()}
            for c in sensor_conduits:
                lk = links[c]
                feat = channel_features(lk.flow, lk.ds_xsection_area, lk.depth,
                                        diam[c]["geom1"], rough[c])
                for k, v in feat.items():
                    row[f"{k}__{c}"] = v
            for nd in sensor_nodes:
                row[f"depth__node_{nd}"] = nodes[nd].depth
            for g in gages:
                row[f"rain__{g}"] = gage_series[g][min(t_min, n_min - 1)]
            rows.append(row)
            i += 1
            if i >= n_min:
                break

    df = pd.DataFrame(rows)
    severities = np.array(severities[:len(df)])
    intensity = basin_series[:len(df)]

    # --- labels (priority blockage > rainfall > normal) ---
    # rainfall is now a *response-based* label: depth must rise above its diurnal
    # dry-weather baseline during/after rain (not merely "it is raining").
    blk_mask = severities >= ONSET_SEVERITY
    start_tod = BASE_START.hour * 60 + BASE_START.minute
    tod_min = (start_tod + np.arange(len(df))) % 1440
    sensor_depth = df[f"depth__node_{blk.n1}"].to_numpy(dtype=float)
    rain_mask = rainfall_response_mask(sensor_depth, tod_min, intensity, blk_mask,
                                       RAIN_REL_MARGIN, RAIN_ABS_MARGIN_M, RAIN_BIN_MIN)
    label = np.where(blk_mask, "blockage", np.where(rain_mask, "rainfall", "normal"))
    df["label"] = label
    df["label_id"] = [LABELS[x] for x in label]

    # --- ground-truth / context columns (not features unless noted) ---
    df["gt_severity"] = severities
    df["gt_setting"] = [S.severity_to_setting(s) for s in severities]
    df["ctx_antecedent_dry_days"] = params["antecedent_dry_days"]
    df["ctx_antecedent_precip_index"] = params["antecedent_precip_index"]
    df["scenario"] = params["scenario"]
    df["run_id"] = params["run_id"]
    df["target_conduit"] = params["target"]

    # --- sensor-realism on the two physical sensors (depth + flow/velocity) ---
    primary = params["target"]
    realism_cols = [f"depth__node_{blk.n1}", f"flow__{primary}", f"vel__{primary}",
                    f"ushear__{primary}"]
    realism_cols = [c for c in realism_cols if c in df.columns]
    df, realism_params = apply_realism(df, realism_cols, rng)

    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    pq = os.path.join(out_dir, "runs", f"{params['run_id']}.parquet")
    df.to_parquet(pq, index=False)
    shutil.rmtree(rundir, ignore_errors=True)

    return {**{k: params[k] for k in
               ("run_id", "scenario", "seed", "duration_h", "target",
                "final_sev", "ramp_min", "onset_min", "antecedent_dry_days",
                "antecedent_precip_index")},
            "ramp_type": "instant" if params["ramp_min"] == 0 else "gradual",
            "n_rows": len(df),
            "n_blockage": int((label == "blockage").sum()),
            "n_rainfall": int((label == "rainfall").sum()),
            "n_normal": int((label == "normal").sum()),
            "sensor_conduits": ";".join(sensor_conduits),
            "realism": json.dumps(realism_params),
            "parquet": os.path.relpath(pq, out_dir)}


# --------------------------------------------------------------------------- #
# Scenario parameter sampling
# --------------------------------------------------------------------------- #
def choke_targets(targets_csv: str, top_k: int = 40, sort_col: str = "V_p10_dry") -> list:
    """Deposition-prone choke points: lowest dry-weather velocity first (most
    silt-deposition / blockage-prone). Swap `sort_col` (e.g. 'tau_full_Pa') to
    rank by self-cleansing shear instead, or pre-filter the CSV to a curated
    list from Bellinge_Blockage_Prone_Locations.docx."""
    t = pd.read_csv(targets_csv)
    t = t.sort_values(sort_col).head(top_k)
    return t["conduit"].tolist()


def sample_params(scenario: int, run_idx: int, rng, targets: list, gages: list,
                  report_step_s: int, routing_step_s: int, k_hops: int) -> dict:
    seed = int(rng.integers(0, 2 ** 31))
    r = np.random.default_rng(seed)
    target = str(r.choice(targets))
    antecedent_dry_days = int(r.integers(1, 15))
    api = float(r.uniform(0, 20))          # antecedent precipitation index (mm-equiv)
    base = dict(seed=seed, target=target, gage_mults=gage_multipliers(r, gages),
                report_step_s=report_step_s, routing_step_s=routing_step_s,
                k_hops=k_hops, antecedent_dry_days=antecedent_dry_days,
                antecedent_precip_index=api,
                run_id=f"s{scenario}_r{run_idx:03d}", scenario=f"S{scenario}")

    if scenario == 1:                       # Baseline Normal, 7 d, dry
        base.update(duration_h=int(r.choice([7 * 24])), storms=[],
                    final_sev=0.0, onset_min=0, ramp_min=0)
    elif scenario == 2:                     # Pure Rainfall Rise, 48-72 h
        dur = int(r.choice([48, 60, 72]))
        onset = int(r.integers(20 * 60, 28 * 60))     # ~24 h dry first
        st = sample_storm(r, onset)
        base.update(duration_h=dur, storms=[st.__dict__],
                    final_sev=0.0, onset_min=0, ramp_min=0)
    elif scenario == 3:                     # Dry Weather Blockage, 48-72 h
        dur = int(r.choice([48, 60, 72]))
        onset = int(r.integers(24 * 60, 36 * 60))
        ramp = int(r.choice([0, 240, 480, 720]))       # instant or 4/8/12 h
        sev = float(r.uniform(0.2, 0.9))
        base.update(duration_h=dur, storms=[],
                    final_sev=sev, onset_min=onset, ramp_min=ramp)
    elif scenario == 4:                     # Wet Weather Blockage, 5-7 d
        dur = int(r.choice([5 * 24, 6 * 24, 7 * 24]))
        blk_onset = int(r.integers(12 * 60, 24 * 60))  # blockage forms day ~1
        ramp = int(r.choice([0, 240, 480, 720]))
        sev = float(r.uniform(0.2, 0.9))
        storm_onset = blk_onset + int(r.integers(18 * 60, 30 * 60))  # storm ~1 day later
        st = sample_storm(r, storm_onset)
        base.update(duration_h=dur, storms=[st.__dict__],
                    final_sev=sev, onset_min=blk_onset, ramp_min=ramp)
    else:
        raise ValueError(scenario)
    return base


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--targets", required=True, help="blockage_targets.csv")
    ap.add_argument("--target-sort", default="V_p10_dry",
                    help="targets.csv column to rank choke points by (ascending)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--scenarios", default="1,2,3,4")
    ap.add_argument("--n-per-scenario", type=int, default=5)
    ap.add_argument("--top-k-targets", type=int, default=40)
    ap.add_argument("--k-hops", type=int, default=2)
    ap.add_argument("--report-step", type=int, default=60)
    ap.add_argument("--routing-step", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = S.InpModel.load(args.inp)
    gages = S.raingage_names(model)
    targets = choke_targets(args.targets, args.top_k_targets, args.target_sort)
    rng = np.random.default_rng(args.seed)

    manifest = []
    for sc in [int(x) for x in args.scenarios.split(",")]:
        for ri in range(args.n_per_scenario):
            p = sample_params(sc, ri, rng, targets, gages,
                              args.report_step, args.routing_step, args.k_hops)
            print(f"[run] {p['run_id']}  target={p['target']}  "
                  f"sev={p['final_sev']:.2f}  ramp={p['ramp_min']}min  "
                  f"dur={p['duration_h']}h", flush=True)
            try:
                manifest.append(run_one(model, args.inp, p, args.out))
            except Exception as e:                       # keep the batch alive
                print(f"   !! failed: {type(e).__name__}: {e}", flush=True)
                manifest.append({"run_id": p["run_id"], "scenario": f"S{sc}",
                                 "error": f"{type(e).__name__}: {e}"})
            pd.DataFrame(manifest).to_csv(os.path.join(args.out, "manifest.csv"),
                                          index=False)
    print(f"done -> {args.out}/manifest.csv  ({len(manifest)} runs)")


if __name__ == "__main__":
    main()
