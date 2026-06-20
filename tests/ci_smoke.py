"""Fast end-to-end smoke test for CI: one short SWMM run must produce a
labelled parquet. Catches breakage that a syntax check would miss."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
import swmm_inp as S
import generate as G

INP = "inputs/BellingeSWMM_v021_nopervious.inp"
TGT = "inputs/blockage_targets.csv"
model = S.InpModel.load(INP)
gages = S.raingage_names(model)
targets = G.choke_targets(TGT, 40)

p = dict(seed=1, target=targets[0], gage_mults={g: 1.0 for g in gages},
         report_step_s=60, routing_step_s=6, k_hops=1,
         antecedent_dry_days=3, antecedent_precip_index=5.0,
         run_id="ci", scenario="S3",
         duration_h=2, storms=[], final_sev=0.7, onset_min=30, ramp_min=30)
row = G.run_one(model, INP, p, "ci_out")
df = pd.read_parquet("ci_out/runs/ci.parquet")
assert len(df) > 0, "no rows produced"
assert "label" in df.columns, "missing label column"
assert row["n_blockage"] > 0, "blockage never labelled"
print(f"CI smoke OK: {df.shape}, {row['n_blockage']} blockage timesteps")
