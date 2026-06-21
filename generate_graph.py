"""
generate_graph.py
=================
GRAPH-view data generator for the Bellinge model — a standalone companion to
generate.py. Same pyswmm engine, same 7 scenarios, same *seeded* scenario sampling
(imported from generate.py), so with matching --seed / --scenarios / --n-per-scenario
it reproduces the IDENTICAL runs as generate.py and exports them as a graph view,
keeping the two datasets aligned for a fair, leakage-free comparison.

Run it independently of generate.py:
  python generate_graph.py --inp BellingeSWMM_v021_nopervious.inp \
      --targets blockage_targets.csv --out graph_data --n-per-scenario 5

Outputs under --out:
  graph_static.npz   ONCE: node_ids, node_static[N,5], edge_index[2,E],
                     edge_static[E,5], edge_ids   (the fixed network topology)
  runs/<run_id>.npz  per run: node_feat[T,N,4], edge_feat[T,E,4], times[T],
                     y_global[T], gt_severity[T], target_edge(int),
                     sensor_node_mask[N], scenario, run_id, target
  manifest.csv       one row per run

  node_static = [invert_elev, max_depth, node_type, x, y]
  edge_static = [length, diameter, roughness, slope, edge_type]
  node_feat   = [depth, head, total_inflow, flooding]
  edge_feat   = [flow, depth, velocity(=flow/area), froude]

Topology is the BASE network. The per-run injected orifice/mid-node are simulation
internals and are NOT added to the graph; the blockage lives as a dynamic property
(gt_severity) + label on the target edge, so the node/edge sets stay identical
across all runs.

Storage: float32, np.savez_compressed. The graph view defaults to a coarser
temporal step (--graph-step-min, default 5) to keep full-network tensors manageable;
the underlying SWMM solver still runs at --routing-step, so hydraulics match
generate.py exactly.
"""
from __future__ import annotations
import argparse
import datetime as dt
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from pyswmm import Simulation, Links, Nodes

import swmm_inp as S
from rainfall import Storm, build_intensity_series, write_rain_dat
from generate import (sample_params, choke_targets, select_sensor_conduits,
                      severity_at, rainfall_response_mask,
                      ONSET_SEVERITY, RAIN_REL_MARGIN, RAIN_ABS_MARGIN_M,
                      RAIN_BIN_MIN, LABELS, BASE_START)

NODE_TYPES = {"junction": 0, "outfall": 1, "storage": 2}
EDGE_TYPES = {"conduit": 0, "pump": 1, "weir": 2, "orifice": 3, "outlet": 4}


# --------------------------------------------------------------------------- #
# Static graph (built once from the base model)
# --------------------------------------------------------------------------- #
def build_static_graph(model: S.InpModel) -> dict:
    inv = model.node_inverts()
    coords = model.coordinates()
    oset = {t[0] for t in model._data_rows("OUTFALLS")}
    sset = {t[0] for t in model._data_rows("STORAGE")}
    node_ids = sorted(inv)
    nidx = {n: i for i, n in enumerate(node_ids)}

    def ntype(n):
        return (NODE_TYPES["outfall"] if n in oset else
                NODE_TYPES["storage"] if n in sset else NODE_TYPES["junction"])

    node_static = np.zeros((len(node_ids), 5), dtype=np.float32)
    for n, i in nidx.items():
        elev, maxd = inv.get(n, (0.0, 0.0))
        x, y = coords.get(n, (0.0, 0.0))
        node_static[i] = [elev, maxd, ntype(n), x, y]

    conduits = model.conduits()
    diam = model.xsection_diam()
    edges = [(c["name"], c["n1"], c["n2"], "conduit") for c in conduits.values()]
    for sec, typ in [("PUMPS", "pump"), ("WEIRS", "weir"),
                     ("ORIFICES", "orifice"), ("OUTLETS", "outlet")]:
        for tok in model._data_rows(sec):
            if len(tok) >= 3:
                edges.append((tok[0], tok[1], tok[2], typ))
    edges = [e for e in edges if e[1] in nidx and e[2] in nidx]

    E = len(edges)
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_static = np.zeros((E, 5), dtype=np.float32)
    edge_ids = []
    for j, (name, n1, n2, typ) in enumerate(edges):
        edge_index[0, j], edge_index[1, j] = nidx[n1], nidx[n2]
        length = conduits[name]["length"] if name in conduits else 0.0
        rough = conduits[name]["rough"] if name in conduits else 0.0
        d = diam.get(name, {}).get("geom1", 0.0)
        e1, e2 = inv.get(n1, (0, 0))[0], inv.get(n2, (0, 0))[0]
        slope = (e1 - e2) / length if length > 0 else 0.0
        edge_static[j] = [length, d, rough, slope, EDGE_TYPES[typ]]
        edge_ids.append(name)

    return dict(node_ids=np.array(node_ids), nidx=nidx, node_static=node_static,
                edge_index=edge_index, edge_static=edge_static,
                edge_ids=np.array(edge_ids),
                edge_name_to_idx={n: j for j, n in enumerate(edge_ids)})


# --------------------------------------------------------------------------- #
# One run -> full-network tensors
# --------------------------------------------------------------------------- #
def run_graph(base_model, params, out_dir, gs, graph_step_min=5) -> dict:
    n_min = params["duration_h"] * 60
    start = BASE_START
    end = start + dt.timedelta(minutes=n_min)
    step_s = graph_step_min * 60

    text, blk = S.inject_inline_orifice(base_model, params["target"])
    text = S.set_simulation_window(text, start, end, report_step_s=step_s,
                                   routing_step_s=params["routing_step_s"])
    gages = S.raingage_names(base_model)
    storms = [Storm(**s) for s in params["storms"]]
    basin = build_intensity_series(n_min, storms)
    gser = {g: basin * params["gage_mults"].get(g, 1.0) for g in gages}
    rundir = tempfile.mkdtemp(prefix="swmmg_")
    write_rain_dat(os.path.join(rundir, "rain_run.dat"), start, gser)
    text = S.set_rain_file(text, "rain_run.dat")
    if params.get("bw"):
        bw = params["bw"]
        o, rmp, hold, pk = bw["onset_min"], bw["ramp_min"], bw["hold_min"], bw["peak_cms"]
        bps = [(0, 0.0), (o, 0.0), (o + rmp, pk), (o + rmp + hold, pk),
               (o + rmp + hold + rmp, 0.0), (n_min, 0.0)]
        text = S.add_inflow_surge(text, blk.n2, f"BW_{params['target']}", bps, start)
    inp_path = os.path.join(rundir, "run.inp")
    with open(inp_path, "w") as fh:
        fh.write(text)

    node_ids, edge_ids = gs["node_ids"], gs["edge_ids"]
    N, Ec = len(node_ids), len(edge_ids)
    n_samp = n_min // graph_step_min
    node_feat = np.zeros((n_samp, N, 4), dtype=np.float32)
    edge_feat = np.zeros((n_samp, Ec, 4), dtype=np.float32)
    severities = np.zeros(n_samp, dtype=np.float32)
    sensor_depth = np.zeros(n_samp, dtype=np.float32)

    with Simulation(inp_path) as sim:
        links, nodes = Links(sim), Nodes(sim)
        orf = links[blk.orifice_name]
        node_objs = [nodes[str(n)] for n in node_ids]
        edge_objs = [links[str(e)] for e in edge_ids]
        sim.step_advance(step_s)
        i = 0
        for _ in sim:
            t_min = i * graph_step_min
            sev = severity_at(t_min, params["onset_min"], params["ramp_min"],
                              params["final_sev"], params.get("clear_onset_min"),
                              params.get("clear_ramp_min", 0))
            orf.target_setting = S.severity_to_setting(sev)
            severities[i] = sev
            for k, nd in enumerate(node_objs):
                node_feat[i, k] = (getattr(nd, "depth", 0.0), getattr(nd, "head", 0.0),
                                   getattr(nd, "total_inflow", 0.0),
                                   getattr(nd, "flooding", 0.0))
            for k, lk in enumerate(edge_objs):
                q = getattr(lk, "flow", 0.0)
                a = getattr(lk, "ds_xsection_area", 0.0) or 0.0
                v = q / a if a > 1e-9 else 0.0
                edge_feat[i, k] = (q, getattr(lk, "depth", 0.0), v,
                                   getattr(lk, "froude", 0.0))
            sensor_depth[i] = nodes[blk.n1].depth
            i += 1
            if i >= n_samp:
                break

    node_feat, edge_feat = node_feat[:i], edge_feat[:i]
    severities, sensor_depth = severities[:i], sensor_depth[:i]
    times = (np.arange(i) * graph_step_min).astype(np.int64)
    intensity = basin[np.clip(times, 0, n_min - 1)]
    blk_mask = severities >= ONSET_SEVERITY
    start_tod = BASE_START.hour * 60 + BASE_START.minute
    tod = (start_tod + times) % 1440
    rain_mask = rainfall_response_mask(sensor_depth, tod, intensity, blk_mask,
                                       RAIN_REL_MARGIN, RAIN_ABS_MARGIN_M, RAIN_BIN_MIN)
    y = np.where(blk_mask, LABELS["blockage"],
                 np.where(rain_mask, LABELS["rainfall"], LABELS["normal"])).astype(np.int64)

    sconds = select_sensor_conduits(base_model, params["target"], k_hops=params["k_hops"])
    cmap = base_model.conduits()
    obs = set()
    for c in sconds:
        obs.add(cmap[c]["n1"]); obs.add(cmap[c]["n2"])
    sensor_node_mask = np.zeros(N, dtype=bool)
    for nn in obs:
        if nn in gs["nidx"]:
            sensor_node_mask[gs["nidx"][nn]] = True
    target_edge = int(gs["edge_name_to_idx"].get(params["target"], -1))

    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "runs", f"{params['run_id']}.npz"),
        node_feat=node_feat, edge_feat=edge_feat, times=times, y_global=y,
        gt_severity=severities, target_edge=target_edge,
        sensor_node_mask=sensor_node_mask,
        scenario=params["scenario"], run_id=params["run_id"], target=params["target"])
    shutil.rmtree(rundir, ignore_errors=True)

    return dict(run_id=params["run_id"], scenario=params["scenario"],
                target=params["target"], final_sev=params["final_sev"],
                ramp_min=params["ramp_min"], duration_h=params["duration_h"],
                graph_step_min=graph_step_min, T=int(i), N=int(N), E=int(Ec),
                n_blockage=int((y == LABELS["blockage"]).sum()),
                n_rainfall=int((y == LABELS["rainfall"]).sum()),
                n_normal=int((y == LABELS["normal"]).sum()))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--target-sort", default="V_p10_dry")
    ap.add_argument("--out", default="graph_data")
    ap.add_argument("--scenarios", default="1,2,3,4,5,6,7")
    ap.add_argument("--n-per-scenario", type=int, default=5)
    ap.add_argument("--top-k-targets", type=int, default=40)
    ap.add_argument("--k-hops", type=int, default=2)
    ap.add_argument("--routing-step", type=int, default=4)
    ap.add_argument("--graph-step-min", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = S.InpModel.load(args.inp)
    gs = build_static_graph(model)
    np.savez_compressed(os.path.join(args.out, "graph_static.npz"),
                        node_ids=gs["node_ids"], node_static=gs["node_static"],
                        edge_index=gs["edge_index"], edge_static=gs["edge_static"],
                        edge_ids=gs["edge_ids"])
    print(f"static graph: N={len(gs['node_ids'])}  E={len(gs['edge_ids'])}", flush=True)

    gages = S.raingage_names(model)
    targets = choke_targets(args.targets, args.top_k_targets, args.target_sort)
    tdf = pd.read_csv(args.targets)
    qmax = dict(zip(tdf["conduit"], tdf["Q_max_Ls"])) if "Q_max_Ls" in tdf.columns else {}
    rng = np.random.default_rng(args.seed)

    manifest = []
    for sc in [int(x) for x in args.scenarios.split(",")]:
        for ri in range(args.n_per_scenario):
            # NB: pass graph_step as report_step so sampling matches; routing_step
            # and the RNG draw order are identical to generate.py, so runs align.
            p = sample_params(sc, ri, rng, targets, gages,
                              args.graph_step_min * 60, args.routing_step,
                              args.k_hops, qmax)
            print(f"[graph] {p['run_id']}  target={p['target']}  "
                  f"sev={p['final_sev']:.2f}  dur={p['duration_h']}h", flush=True)
            try:
                manifest.append(run_graph(model, p, args.out, gs, args.graph_step_min))
            except Exception as e:
                print(f"   !! failed: {type(e).__name__}: {e}", flush=True)
                manifest.append({"run_id": p["run_id"], "scenario": f"S{sc}",
                                 "error": f"{type(e).__name__}: {e}"})
            pd.DataFrame(manifest).to_csv(os.path.join(args.out, "manifest.csv"),
                                          index=False)
    print(f"done -> {args.out}/manifest.csv  ({len(manifest)} runs)")


if __name__ == "__main__":
    main()
