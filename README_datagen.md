# Core-4 Blockage Data Generation (Bellinge SWMM)

Generates labelled per-minute sensor time series for the first four scenarios of
`PySWMM_Scenarios_Revised.docx`, on the calibrated Bellinge model, with pumps/
weirs/orifices kept in place.

## Files
| file | role |
|---|---|
| `generate.py` | orchestrator: scenarios 1–4, Monte-Carlo sweep, labelling, export |
| `swmm_inp.py` | `.inp` parsing + inline-orifice blockage injection + run rewiring |
| `rainfall.py` | synthetic design-storm `.dat` generation (controllable axes) |
| `hydraulics.py` | depth/flow → velocity, hydraulic radius, shear-velocity proxy |
| `sensor_realism.py` | noise / drift / quantisation / dropout (+ missing flags) |

## How a blockage is injected (cross-section reduction)
A controllable **inline orifice** is inserted on the target conduit (the conduit
is split by a tiny mid-node; a SIDE rectangular orifice sits on the short stub).
Its `setting` scales open area linearly, so a runtime ramp of `setting` is a
gradual cross-section reduction. Severity `s` = fraction of pipe area removed,
mapped exactly by `setting(s) = (1−s)·π/4`. The orifice is inserted in **every**
run (held open for non-blockage) so network topology stays decorrelated from the
class label.

## Run on Colab
```python
!pip -q install pyswmm pandas pyarrow numpy
# upload the model + rain file + targets, or mount Drive, then:
!python generate.py \
    --inp BellingeSWMM_v021_nopervious.inp \
    --targets blockage_targets.csv \
    --out data \
    --scenarios 1,2,3,4 \
    --n-per-scenario 25
```
Keep `BellingeSWMM_v021_nopervious.inp` and `blockage_targets.csv` in the working
dir. The rain `.dat` is generated per run, so the original historical `.dat` is
not needed.

### Useful flags
- `--n-per-scenario N` runs per scenario (start small, e.g. 3, to smoke-test)
- `--routing-step 4 --report-step 60` fidelity/speed trade-off (4 s / 1 min default)
- `--top-k-targets 40 --target-sort V_p10_dry` candidate injection conduits and
  ranking column (use `tau_full_Pa` to rank by self-cleansing shear, or pre-filter
  the CSV to a curated blockage-prone list)
- `--k-hops 2` spatial extent of the sensor neighbourhood
- `--seed 0` reproducibility

## Output
- `data/runs/<run_id>.parquet` — wide 1-min series: per-conduit `flow/depth/vel/
  ushear/fill/froude`, node depths, rain per gage, labels (`label`, `label_id`),
  ground-truth `gt_severity`/`gt_setting`, antecedent context, and `*_meas` /
  `*_missing` for the two physical sensors. Clean channels are always retained.
- `data/manifest.csv` — one row per run with every sampled parameter, the
  instant-vs-gradual `ramp_type`, and per-class row counts.

## Labelling
Per-timestep, priority **blockage > rainfall > normal**:
- blockage when `gt_severity ≥ 0.10` (explicit, constant onset threshold)
- rainfall is **response-based**: a storm has begun *and* the sensor depth exceeds
  its own diurnal dry-weather baseline by a margin (`>10%` and `≥0.02 m`), covering
  the rise and recession until depth recovers. A drizzle that doesn't lift levels is
  *not* labelled rainfall. The baseline is estimated per run from its dry, unblocked
  timesteps (median per 30-min time-of-day bin), so DWF diurnal peaks stay "normal".
  Tunable via `RAIN_REL_MARGIN`, `RAIN_ABS_MARGIN_M`, `RAIN_BIN_MIN`.
- normal otherwise (dry-weather diurnal flow, including morning/evening DWF peaks)

Report instant vs gradual blockage metrics separately (use `ramp_type`) so the
easy instant case doesn't inflate aggregates.

## Coverage / class-balance check
The generator does **not** rebalance classes — normal dominates by design (realistic
event rate). Handle imbalance at training/eval (class weights, minority augmentation,
recall-weighted metrics), not here. Before training, confirm you have enough
**absolute** minority coverage with `class_summary.py`:

```python
from class_summary import summarize
summarize("data")    # prints report; writes data/class_summary.csv + class_counts_per_run.csv
```
Reports per-class timestep totals + %, contiguous-event counts, blockage coverage
(distinct injection locations, severity histogram, instant-vs-gradual split), a
per-scenario breakdown, and warnings if a minority class is thin (few events, few
injection locations, or empty severity bins). CLI: `python class_summary.py --data data`.

## Known limitations (be explicit in the thesis)
- The inline orifice is an abstraction of a physical blockage; at `s=0` it matches
  the pipe's flow area but adds a small local loss (kept identical across classes).
- `ushear` is a Manning uniform-flow shear-velocity **proxy**, weakest under the
  strong backwater of a blockage — labelled as a proxy, not a measured shear.
- Synthetic triangular storms randomise intensity/duration/shape but do not carry
  the temporal structure of the historical Bellinge record (a real-event sampling
  mode can be added if you want that realism).
- Scenarios 5–7 (non-blockage backwater, clearance/recovery, near-surcharge) are
  not in this Core-4 build.
```
