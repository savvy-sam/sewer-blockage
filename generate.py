"""
generate.py
===========
Sewer-blockage data generator for the Bellinge SWMM model (Scenarios 1-7).

Scenarios (per PySWMM_Scenarios_Revised.docx, Part 1):
  1 Baseline Normal        7 d     dry, diurnal only            -> normal
  2 Pure Rainfall Rise     48-72 h 24h dry -> storm -> recovery -> rainfall
  3 Dry Weather Blockage   48-72 h blockage ramps hrs 24-36     -> blockage
  4 Wet Weather Blockage   5-7 d   blockage forms, storm ~1d on -> blockage
  5 Non-Blockage Backwater 48-72 h downstream surge raises depth-> normal (hard neg)
  6 Blockage Clearance     3-5 d   blockage forms then removed  -> blockage->normal
  7 Near-Surcharge Storm   48-72 h extreme storm, no blockage   -> rainfall (hard neg)

Per-run randomisation: blockage severity, instant vs gradual ramp, injection node,
rainfall intensity/duration, rainfall spatial heterogeneity, antecedent dry-weather
duration + antecedent precip index. Onset time and injection node are randomised so
timing/location are not learnable artefacts. A controllable inline orifice is
inserted in EVERY run (held open when no blockage) to keep topology decorrelated
from the label.

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
import multiprocessing as mp
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

ONSET_SEVERITY = 0.10            # blockage proportion p (ICM vertical area fraction) that
                                 # counts as "blockage onset" for the cause-based label
RAIN_REL_MARGIN = 0.10           # sensor depth must exceed dry baseline by >10% ...
RAIN_ABS_MARGIN_M = 0.02         # ... and by >=0.02 m (absolute floor for small pipes)
RAIN_BIN_MIN = 30                # time-of-day bin (min) for the diurnal dry baseline
LABELS = {"normal": 0, "rainfall": 1, "blockage": 2}
BASE_START = dt.datetime(2012, 6, 29, 0, 1)   # arbitrary dry anchor in the model calendar

# --- blockage mechanism: runtime average-loss-coefficient on the real target conduit ---
# We add a minor-loss coefficient K to the target conduit and raise it over time to
# represent the obstruction. `severity` here IS the InfoWorks-ICM "vertical blockage
# proportion" p in [0,1): the fraction of the flow AREA obstructed at EVERY water level
# (a vertical curtain / fatberg, level-independent — NOT a bottom silt deposit). For a
# partial blockage of area fraction p the sudden-contraction (orifice) analogy gives
#   K(p) = 1 / (1 - p)^2   (head loss h_L = K * V^2 / 2g). The flow is then an EMERGENT
# response the solver computes (water backs up, less gets through) rather than a value we
# impose — more defensible than the earlier flow_limit cap. This matches how industry
# hydraulic software (InfoWorks ICM) represents a blockage, EXCEPT that SWMM exposes only a
# single conduit Kavg, so we collapse ICM's separate contraction+expansion coefficients into
# one Kavg via K(p) (see Findings.md F7 addendum). NB: field/column names keep the word
# `sev`/`severity` (final_sev, gt_severity) for data reproducibility — read them as p.
# Runtime-settable via pyswmm `link.average_head_loss` PROVIDED the conduit already has a
# [LOSSES] entry, so we pre-seed one at K_BASE (see swmm_inp.set_conduit_loss). Confirmed
# runtime-settable on Bellinge, overturning the earlier read (Findings.md F7).
K_BASE = 1.0                     # baseline loss coeff seeded on the target (= K(s=0)); makes
                                 # average_head_loss runtime-settable; applied to the twins too
S_MAX = 0.995                    # clamp severity so K stays finite (K(0.995) ~ 40000)

# --- Approach 1: observability-gated labels (label_obs), computed from paired
#     counterfactual twins vs the physical sensor visibility floor (Findings.md F8) ---
OBS_K_SIGMA = 3.0                # a change < k*sigma of the sensor is invisible
OBS_TAU_ABS_M = 0.01             # absolute depth floor (m) a level sensor can resolve
OBS_SUSTAIN_MIN = 5              # effect must exceed the floor for >= this many minutes
DEFAULT_LABEL_SCHEME = "obs_and" # which variant populates `label`/`label_id`
                                 # (cause | obs_or | obs_and | obs_depth); ALL are always
                                 # written so the schemes can be A/B-compared without regen.
                                 # obs_depth = upstream level only (deployable/Branch-B).


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
    # keep conduits with a circular diameter (needed for channel features), but
    # ALWAYS keep the target conduit even if non-circular (bug fix: it was silently
    # dropped when non-circular, so the target's own channel was never recorded).
    diam = model.xsection_diam()
    keep = [c for c in sorted(sensor_conduits)
            if c == target or (diam.get(c, {}).get("shape") == "CIRCULAR"
                               and diam.get(c, {}).get("geom1", 0) > 0)]
    if target not in keep:
        keep.insert(0, target)
    return keep


# --------------------------------------------------------------------------- #
# Severity schedule + labels
# --------------------------------------------------------------------------- #
def severity_at(t_min, onset_min, ramp_min, final_sev,
                clear_onset=None, clear_ramp=0):
    """Blockage proportion p(t) over time (ICM vertical-blockage convention: fraction of
    flow area obstructed at every water level): ramp up to final_sev (= p_max), hold, and
    (Scenario 6) optionally clear back to 0 starting at clear_onset over clear_ramp minutes
    (clear_ramp=0 => instant removal)."""
    if final_sev <= 0 or t_min < onset_min:
        return 0.0
    # clearance phase (overrides plateau once it begins)
    if clear_onset is not None and t_min >= clear_onset:
        if clear_ramp <= 0 or t_min >= clear_onset + clear_ramp:
            return 0.0
        return max(0.0, final_sev * (1.0 - (t_min - clear_onset) / clear_ramp))
    # rising / plateau
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


def _sustained(mask, m):
    """True where `mask` has been True for >= m consecutive samples (incl. current)."""
    mask = np.asarray(mask, dtype=bool)
    if m <= 1:
        return mask
    run = np.zeros(len(mask), dtype=int)
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        run[i] = c
    return run >= m


def _simulate(inp_path, sensor_conduits, sensor_nodes, diam, rough,
              gages, gage_series, n_min, control=None):
    """Run one SWMM simulation and return a clean (pre-realism) per-minute DataFrame.
    `control(i, links)` is called each step (used to set the target's loss coefficient)."""
    rows = []
    with Simulation(inp_path) as sim:
        links, nodes = Links(sim), Nodes(sim)
        sim.step_advance(60)
        i = 0
        for _ in sim:
            if control is not None:
                control(i, links)
            row = {"t_min": i,
                   "timestamp": (BASE_START + dt.timedelta(minutes=i)).isoformat()}
            for c in sensor_conduits:
                lk = links[c]
                if c in diam and diam[c].get("geom1", 0) > 0:
                    feat = channel_features(lk.flow, lk.ds_xsection_area, lk.depth,
                                            diam[c]["geom1"], rough[c])
                    for k, v in feat.items():
                        row[f"{k}__{c}"] = v
                else:                       # non-circular target: raw flow/depth only
                    row[f"flow__{c}"] = lk.flow
                    row[f"depth__{c}"] = lk.depth
            for nd in sensor_nodes:
                row[f"depth__node_{nd}"] = nodes[nd].depth
            for g in gages:
                row[f"rain__{g}"] = gage_series[g][min(i, n_min - 1)]
            rows.append(row)
            i += 1
            if i >= n_min:
                break
    return pd.DataFrame(rows)


def _sensor_sigma(clean, realism_params, col):
    """Effective sensor noise sigma (m) for a depth column: additive noise +
    quantisation, from the realism model applied to that column."""
    p = realism_params.get(col)
    if not p:
        return 0.0
    x = np.asarray(clean, dtype=float)
    rms = float(np.sqrt(np.nanmean(x * x))) if len(x) else 0.0
    sig_noise = rms * (10.0 ** (-p.get("snr_db", 60.0) / 20.0))
    res = p.get("resolution", 0.0)
    return float(np.sqrt(sig_noise ** 2 + (res ** 2) / 12.0))


# --------------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------------- #
def run_one(base_model: S.InpModel, base_inp_path: str, params: dict, out_dir: str) -> dict:
    """Execute the observed run + counterfactual twins, label (cause & observability),
    write parquet, return a manifest row.

    Blockage mechanism: runtime average-loss-coefficient K(s)=1/(1-s)^2 on the real target
    conduit (the flow drop is emergent, not imposed). Twins (identical seed/window; one
    factor toggled) give the clean causal effect used for the observability-gated label
    (Findings.md F8):
      * dry   : no rain,  no blockage  -> baseline (target carries K_BASE)
      * noblk : rain on,  no blockage  -> isolates the rain effect
      * main  : rain on,  blockage K(s) -> isolates the blockage (vs noblk); the observed run
    """
    rng = np.random.default_rng(params["seed"])
    n_min = params["duration_h"] * 60
    start = BASE_START
    end = start + dt.timedelta(minutes=n_min)
    target = params["target"]
    conduits = base_model.conduits()
    tc = conduits[target]
    n1, n2 = tc["n1"], tc["n2"]

    # --- sensor set ---
    sensor_conduits = select_sensor_conduits(base_model, target, k_hops=params["k_hops"])
    # downstream conduit (starts at the target's outlet node): its flow is the
    # EMERGENT flow response to the blockage (drops as less water gets through), so
    # it — not the capped target flow — is the non-circular flow feature source (F8).
    # Ensure it is recorded even if select_sensor_conduits didn't pick it up.
    downstream = next((c for c in sensor_conduits if conduits[c]["n1"] == n2),
                      next((c for c, d in conduits.items() if d["n1"] == n2), ""))
    if downstream and downstream not in sensor_conduits:
        sensor_conduits = sensor_conduits + [downstream]
    diam = base_model.xsection_diam()
    rough = {c["name"]: c["rough"] for c in conduits.values()}
    sensor_nodes = sorted({conduits[c]["n1"] for c in sensor_conduits} |
                          {conduits[c]["n2"] for c in sensor_conduits})
    sensor_node = n1                       # upstream manhole = primary detection sensor

    # --- rainfall series ---
    gages = S.raingage_names(base_model)
    storms = [Storm(**s) for s in params["storms"]]
    basin_series = build_intensity_series(n_min, storms)
    mults = params["gage_mults"]
    gage_series = {g: basin_series * mults.get(g, 1.0) for g in gages}
    zero_series = {g: np.zeros(n_min) for g in gages}
    has_rain = float(basin_series.sum()) > 0
    has_blk = params["final_sev"] > 0

    # --- base text (no orifice) + window + optional downstream backwater surge ---
    rundir = tempfile.mkdtemp(prefix="swmmrun_")
    base_text = S.set_simulation_window(base_model.text, start, end,
                                        report_step_s=params["report_step_s"],
                                        routing_step_s=params["routing_step_s"])
    # seed a [LOSSES] entry on the target so average_head_loss is runtime-settable;
    # K_BASE is the s=0 value and is carried identically by the twins (consistent baseline).
    base_text = S.set_conduit_loss(base_text, target, kavg=K_BASE)
    if params.get("bw"):
        bw = params["bw"]
        o, rmp, hold, pk = bw["onset_min"], bw["ramp_min"], bw["hold_min"], bw["peak_cms"]
        bps = [(0, 0.0), (o, 0.0), (o + rmp, pk), (o + rmp + hold, pk),
               (o + rmp + hold + rmp, 0.0), (n_min, 0.0)]
        base_text = S.add_inflow_surge(base_text, n2, f"BW_{target}", bps, start)

    def _write_inp(series, name):
        write_rain_dat(os.path.join(rundir, name + ".dat"), start, series)
        t = S.set_rain_file(base_text, name + ".dat")
        p = os.path.join(rundir, name + ".inp")
        with open(p, "w") as fh:
            fh.write(t)
        return p
    inp_rain = _write_inp(gage_series, "rain_on")
    inp_dry = _write_inp(zero_series, "rain_off") if has_rain else inp_rain

    def _sim(inp_path, series, control=None):
        return _simulate(inp_path, sensor_conduits, sensor_nodes, diam, rough,
                         gages, series, n_min, control=control)

    # --- pass 1: dry, no-blockage baseline ---
    df_dry = _sim(inp_dry, zero_series)
    # --- pass 2: rain-on, no-blockage (isolates the rain effect) ---
    df_noblk = _sim(inp_rain, gage_series) if has_rain else df_dry

    # --- severity schedule + pass 3: observed run with the loss-coefficient obstruction ---
    severities = np.array([severity_at(i, params["onset_min"], params["ramp_min"],
                                       params["final_sev"], params.get("clear_onset_min"),
                                       params.get("clear_ramp_min", 0))
                           for i in range(n_min)])

    def _loss(i, links):
        s = min(severities[i] if i < len(severities) else 0.0, S_MAX)
        links[target].average_head_loss = 1.0 / ((1.0 - s) ** 2)   # K(s); = K_BASE at s=0

    df = _sim(inp_rain, gage_series, control=_loss) if has_blk else df_noblk

    m = min(len(df), len(df_noblk), len(df_dry), n_min)
    df = df.iloc[:m].reset_index(drop=True)
    severities = severities[:m]
    intensity = basin_series[:m]
    ncol = f"depth__node_{sensor_node}"

    # ================= LABEL A: cause-based (intervention active) ==================
    blk_cause = severities >= ONSET_SEVERITY
    start_tod = BASE_START.hour * 60 + BASE_START.minute
    tod_min = (start_tod + np.arange(m)) % 1440
    sensor_depth = df[ncol].to_numpy(dtype=float)
    rain_cause = rainfall_response_mask(sensor_depth, tod_min, intensity, blk_cause,
                                        RAIN_REL_MARGIN, RAIN_ABS_MARGIN_M, RAIN_BIN_MIN)
    label_cause = np.where(blk_cause, "blockage",
                           np.where(rain_cause, "rainfall", "normal"))

    # --- context / ground truth (not features) ---
    df["gt_severity"] = severities        # = ICM vertical blockage proportion p(t) in [0,1)
    _s_clamped = np.minimum(severities, S_MAX)
    df["gt_k_loss"] = np.where(severities > 0,
                               1.0 / (1.0 - _s_clamped) ** 2, K_BASE)
    df["ctx_antecedent_dry_days"] = params["antecedent_dry_days"]
    df["ctx_antecedent_precip_index"] = params["antecedent_precip_index"]
    df["scenario"] = params["scenario"]
    df["run_id"] = params["run_id"]
    df["target_conduit"] = target
    df["ctx_downstream_conduit"] = downstream

    # --- sensor realism: upstream depth gauge + DOWNSTREAM flow meter (the real
    #     instrument locations; the target's own flow is the imposed knob, not sensed) ---
    fsrc = downstream if downstream else target
    realism_cols = [ncol, f"flow__{fsrc}", f"vel__{fsrc}", f"ushear__{fsrc}"]
    realism_cols = [c for c in realism_cols if c in df.columns]
    clean_depth = df[ncol].to_numpy(dtype=float).copy()
    clean_flow_ds = (df[f"flow__{downstream}"].to_numpy(dtype=float).copy()
                     if downstream and f"flow__{downstream}" in df.columns else None)
    df, realism_params = apply_realism(df, realism_cols, rng)

    # ============ LABEL B: observability-gated (counterfactual Δ vs sensor floor) ====
    # "What a sensor can see" = an ABSOLUTE physical change at a real instrument that
    # exceeds that instrument's own ABSOLUTE noise floor (NOT a relative/normalised
    # value), on the CLEAN pre-noise signal from the counterfactual twins. A moment is
    # visible if EITHER the upstream depth gauge OR the downstream flow meter moves
    # beyond its floor, sustained. (The target's own flow is the imposed knob and is
    # never used here; the downstream flow is the emergent response.)
    d_full = df[ncol].to_numpy(dtype=float)                      # blocked + rain (clean)
    d_noblk = df_noblk[ncol].to_numpy(dtype=float)[:m]           # rain, no blockage
    d_dry = df_dry[ncol].to_numpy(dtype=float)[:m]               # no rain, no blockage
    delta_blk = d_full - d_noblk                                 # blockage effect on DEPTH (m)
    delta_rain = d_noblk - d_dry                                 # rain effect on DEPTH (m)
    tau_d = max(OBS_TAU_ABS_M, OBS_K_SIGMA * _sensor_sigma(clean_depth, realism_params, ncol))
    dep_blk = np.abs(delta_blk) >= tau_d
    dep_rain = np.abs(delta_rain) >= tau_d

    # downstream FLOW meter — absolute m3/s vs the flow sensor's own absolute noise floor
    delta_blk_f = np.zeros(m); delta_rain_f = np.zeros(m); tau_f = float("inf")
    fcol_ds = f"flow__{downstream}" if downstream else ""
    if fcol_ds and fcol_ds in df.columns and fcol_ds in df_noblk.columns and clean_flow_ds is not None:
        f_full = df[fcol_ds].to_numpy(dtype=float)
        f_noblk = df_noblk[fcol_ds].to_numpy(dtype=float)[:m]
        f_dry = (df_dry[fcol_ds].to_numpy(dtype=float)[:m] if fcol_ds in df_dry.columns else f_noblk)
        delta_blk_f = f_full - f_noblk
        delta_rain_f = f_noblk - f_dry
        sig_f = _sensor_sigma(clean_flow_ds, realism_params, fcol_ds)
        tau_f = OBS_K_SIGMA * sig_f if sig_f > 0 else float("inf")
    flo_blk = np.abs(delta_blk_f) >= tau_f
    flo_rain = np.abs(delta_rain_f) >= tau_f

    # Compute BOTH combination rules so OR vs AND can be A/B-compared:
    #   OR  = either sensor suffices (earlier onset, higher coverage)
    #   AND = agreement — both the upstream depth AND the downstream flow must show it
    #         (higher confidence; onset set by the slower sensor — apt since a real
    #         blockage is rarely sudden). Falls back to depth-only w/o a flow sensor.
    have_flow = np.isfinite(tau_f)
    blk_or = _sustained(dep_blk | flo_blk, OBS_SUSTAIN_MIN)
    rain_or = _sustained(dep_rain | flo_rain, OBS_SUSTAIN_MIN)
    if have_flow:
        blk_and = _sustained(dep_blk & flo_blk, OBS_SUSTAIN_MIN)
        rain_and = _sustained(dep_rain & flo_rain, OBS_SUSTAIN_MIN)
    else:
        blk_and, rain_and = blk_or, rain_or
    # depth-only: upstream LEVEL sensor alone (what almost every real CSO site has).
    # This is the deployable-consistent label — matches the field-ready (Branch B) model.
    blk_depth = _sustained(dep_blk, OBS_SUSTAIN_MIN)
    rain_depth = _sustained(dep_rain, OBS_SUSTAIN_MIN)
    label_obs_or = np.where(blk_or, "blockage", np.where(rain_or, "rainfall", "normal"))
    label_obs_and = np.where(blk_and, "blockage", np.where(rain_and, "rainfall", "normal"))
    label_obs_depth = np.where(blk_depth, "blockage", np.where(rain_depth, "rainfall", "normal"))
    df["gt_delta_blk"] = delta_blk                # depth effect
    df["gt_delta_rain"] = delta_rain
    df["gt_delta_blk_flow"] = delta_blk_f         # downstream flow effect
    df["gt_obs_tau_depth"] = tau_d
    df["gt_obs_tau_flow"] = tau_f if np.isfinite(tau_f) else 0.0

    # --- write ALL label variants; `label` = the chosen one (A/B decide later) ---
    variants = {"cause": label_cause, "obs_or": label_obs_or,
                "obs_and": label_obs_and, "obs_depth": label_obs_depth}
    for name, lab in variants.items():
        df[f"label_{name}"] = lab
        df[f"label_{name}_id"] = [LABELS[x] for x in lab]
    scheme = params.get("label_scheme", DEFAULT_LABEL_SCHEME)
    chosen = variants.get(scheme, label_cause)
    df["label"] = chosen
    df["label_id"] = [LABELS[x] for x in chosen]

    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    pq = os.path.join(out_dir, "runs", f"{params['run_id']}.parquet")
    df.to_parquet(pq, index=False)
    shutil.rmtree(rundir, ignore_errors=True)

    return {**{k: params[k] for k in
               ("run_id", "scenario", "seed", "duration_h", "target",
                "final_sev", "ramp_min", "onset_min", "antecedent_dry_days",
                "antecedent_precip_index")},
            "ramp_type": "instant" if params["ramp_min"] == 0 else "gradual",
            "label_scheme": scheme, "k_loss_max": float(df["gt_k_loss"].max()),
            "obs_tau_depth": tau_d, "obs_tau_flow": (tau_f if np.isfinite(tau_f) else 0.0),
            "n_rows": m,
            "n_blk_cause": int((label_cause == "blockage").sum()),
            "n_blk_obs_or": int((label_obs_or == "blockage").sum()),
            "n_blk_obs_and": int((label_obs_and == "blockage").sum()),
            "n_blk_obs_depth": int((label_obs_depth == "blockage").sum()),
            "n_rain_obs_and": int((label_obs_and == "rainfall").sum()),
            "n_normal_obs_and": int((label_obs_and == "normal").sum()),
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
                  report_step_s: int, routing_step_s: int, k_hops: int,
                  target_qmax: dict | None = None) -> dict:
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
    elif scenario == 5:                     # Non-Blockage Backwater, 48-72 h -> normal
        dur = int(r.choice([48, 60, 72]))
        onset = int(r.integers(24 * 60, 36 * 60))
        ramp = int(r.choice([60, 120]))                # surge rise time
        hold = int(r.integers(2 * 60, 8 * 60))
        qmax_ls = float((target_qmax or {}).get(target, 2.0))   # pipe capacity, L/s
        # surge must dwarf pipe capacity to back water up (flow falls/reverses);
        # too small and it just drains downstream. Magnitude is location-dependent
        # (downstream capacity) — inspect S5 depth/flow and tune this range if needed.
        peak = float(r.uniform(8.0, 20.0)) * qmax_ls / 1000.0   # CMS
        base.update(duration_h=dur, storms=[], final_sev=0.0, onset_min=0, ramp_min=0,
                    bw=dict(onset_min=onset, ramp_min=ramp, hold_min=hold, peak_cms=peak))
    elif scenario == 6:                     # Blockage Clearance/Recovery, 3-5 d
        dur = int(r.choice([3 * 24, 4 * 24, 5 * 24]))
        onset = int(r.integers(12 * 60, 24 * 60))
        ramp = int(r.choice([0, 240, 480]))            # instant or 4/8 h onset
        sev = float(r.uniform(0.3, 0.9))
        hold = int(r.integers(12 * 60, 36 * 60))       # persist before removal
        clear_ramp = int(r.choice([0, 120, 240, 480]))  # instant clear or self-clearing
        base.update(duration_h=dur, storms=[], final_sev=sev,
                    onset_min=onset, ramp_min=ramp,
                    clear_onset_min=onset + ramp + hold, clear_ramp_min=clear_ramp)
    elif scenario == 7:                     # Extreme Near-Surcharge Storm, 48-72 h
        dur = int(r.choice([48, 60, 72]))
        onset = int(r.integers(20 * 60, 28 * 60))
        st = sample_storm(r, onset, extreme=True)
        base.update(duration_h=dur, storms=[st.__dict__],
                    final_sev=0.0, onset_min=0, ramp_min=0)
    else:
        raise ValueError(scenario)
    return base


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
# Parallel workers (each run is independent; one model loaded per worker process)
# --------------------------------------------------------------------------- #
_WORKER_MODEL = None


def _init_worker(inp_path):
    global _WORKER_MODEL
    _WORKER_MODEL = S.InpModel.load(inp_path)


def _run_job(job):
    params, inp_path, out_dir = job
    try:
        return run_one(_WORKER_MODEL, inp_path, params, out_dir)
    except Exception as e:                                # keep the batch alive
        return {"run_id": params["run_id"], "scenario": params["scenario"],
                "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--targets", required=True, help="blockage_targets.csv")
    ap.add_argument("--target-sort", default="V_p10_dry",
                    help="targets.csv column to rank choke points by (ascending)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--scenarios", default="1,2,3,4,5,6,7")
    ap.add_argument("--n-per-scenario", type=int, default=5)
    ap.add_argument("--top-k-targets", type=int, default=40)
    ap.add_argument("--k-hops", type=int, default=2)
    ap.add_argument("--report-step", type=int, default=60)
    ap.add_argument("--routing-step", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes; 0 = auto (use all CPU cores), "
                         "1 = serial (default). Never exceeds the number of runs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip runs already completed (present in the existing "
                         "manifest.csv, else existing runs/*.parquet) and carry their "
                         "manifest rows forward, so a stopped/crashed run continues "
                         "instead of restarting. Use the SAME args + --seed as the "
                         "original call so the run set is identical.")
    ap.add_argument("--label-scheme",
                    choices=["cause", "obs_or", "obs_and", "obs_depth"],
                    default=DEFAULT_LABEL_SCHEME,
                    help="which variant populates `label`/`label_id`; label_cause, "
                         "label_obs_or, label_obs_and and label_obs_depth (level-only, "
                         "deployable) are ALL written for A/B comparison")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = S.InpModel.load(args.inp)
    gages = S.raingage_names(model)
    targets = choke_targets(args.targets, args.top_k_targets, args.target_sort)
    tdf = pd.read_csv(args.targets)
    qmax = dict(zip(tdf["conduit"], tdf["Q_max_Ls"])) if "Q_max_Ls" in tdf.columns else {}
    rng = np.random.default_rng(args.seed)

    # Build the full param list first, sequentially: the RNG draw order (and hence
    # which runs are produced) is identical regardless of --workers, so parallelism
    # never changes WHAT is generated, only how fast.
    jobs = [sample_params(sc, ri, rng, targets, gages, args.report_step,
                          args.routing_step, args.k_hops, qmax)
            for sc in [int(x) for x in args.scenarios.split(",")]
            for ri in range(args.n_per_scenario)]
    for p in jobs:
        p["label_scheme"] = args.label_scheme

    # --- resume: skip runs already completed and carry their manifest rows forward, so
    #     the rewritten manifest.csv stays complete. "Done" = present in the existing
    #     manifest (authoritative); if there is no manifest, fall back to an existing
    #     parquet. A single crash-orphan parquet (written but not yet in the manifest) is
    #     re-run deterministically, which restores its manifest row. ---
    manifest = []
    if args.resume:
        mpath = os.path.join(args.out, "manifest.csv")
        done = set()
        if os.path.exists(mpath):
            prev = pd.read_csv(mpath)
            manifest = prev.to_dict("records")
            done = set(prev["run_id"].astype(str))
        else:
            rdir = os.path.join(args.out, "runs")
            if os.path.isdir(rdir):
                done = {f[:-8] for f in os.listdir(rdir) if f.endswith(".parquet")}
        n0 = len(jobs)
        jobs = [p for p in jobs if p["run_id"] not in done]
        print(f"resume: {len(done)} already done, {n0 - len(jobs)} skipped, "
              f"{len(jobs)} to run", flush=True)

    # choose worker count: --workers 0 (or negative) -> use all CPU cores;
    # never spawn more workers than there are runs.
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(jobs)))

    def record(row):
        manifest.append(row)
        pd.DataFrame(manifest).sort_values("run_id").to_csv(
            os.path.join(args.out, "manifest.csv"), index=False)

    if workers > 1:
        print(f"running {len(jobs)} simulations on {workers} workers "
              f"(detected {os.cpu_count()} cores) ...", flush=True)
        payload = [(p, args.inp, args.out) for p in jobs]
        with mp.Pool(workers, initializer=_init_worker, initargs=(args.inp,)) as pool:
            for row in pool.imap_unordered(_run_job, payload):
                tag = row.get("error") or f"ok ({row.get('n_rows', '?')} rows)"
                print(f"[done] {row['run_id']}  {tag}", flush=True)
                record(row)
    else:
        for p in jobs:
            print(f"[run] {p['run_id']}  target={p['target']}  "
                  f"sev={p['final_sev']:.2f}  ramp={p['ramp_min']}min  "
                  f"dur={p['duration_h']}h", flush=True)
            try:
                record(run_one(model, args.inp, p, args.out))
            except Exception as e:                       # keep the batch alive
                print(f"   !! failed: {type(e).__name__}: {e}", flush=True)
                record({"run_id": p["run_id"], "scenario": p["scenario"],
                        "error": f"{type(e).__name__}: {e}"})
    print(f"done -> {args.out}/manifest.csv  ({len(manifest)} runs)")


if __name__ == "__main__":
    main()
# end of file (severity ≡ ICM vertical blockage proportion p; see F7 addendum)
