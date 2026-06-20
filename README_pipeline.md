# Automated pipeline: GitHub + Google Colab

Code lives in GitHub (versioned, never truncates on upload); Colab is the compute.
GitHub Actions checks every push so a broken or partial file is caught before you
run anything.

```
GitHub repo  ──git clone/pull──►  Colab (runs generate.py)  ──►  outputs to Drive
     ▲                                                              
     └── GitHub Actions CI smoke-tests every push (compile + 1 short SWMM run)
```

## Repository layout (this folder = repo root)
```
generate.py  swmm_inp.py  rainfall.py  hydraulics.py  sensor_realism.py
class_summary.py  requirements.txt  run_pipeline.ipynb  README_pipeline.md
inputs/   BellingeSWMM_v021_nopervious.inp,  blockage_targets.csv
tests/    ci_smoke.py
.github/workflows/ci.yml
.gitignore           # data/ and *.parquet are NOT committed
```

## One-time setup
1. Create a new (private) GitHub repo, e.g. `sewer-datagen`.
2. Push this folder as the repo root:
   ```bash
   cd "data_generation"
   git init && git add . && git commit -m "data generation pipeline"
   git branch -M main
   git remote add origin https://github.com/<you>/sewer-datagen.git
   git push -u origin main
   ```
   (`inputs/` is committed so the repo is self-contained; the 1.2 MB `.inp` is fine
   for git. Generated `data/` is git-ignored.)
3. Open `run_pipeline.ipynb`, set `REPO_URL` to your repo, commit that change.

## Run it (every time)
Open the notebook in Colab — either upload `run_pipeline.ipynb` once, or use the
badge URL (replace placeholders):
```
https://colab.research.google.com/github/<you>/sewer-datagen/blob/main/run_pipeline.ipynb
```
The notebook: clones/pulls latest → installs deps → integrity check → smoke run →
full Core-4 batch → coverage summary → copies outputs to Drive. To get new code
into Colab later, just `git push` from your machine and re-run the clone/pull cell
(no re-uploading files).

## Continuous integration (automatic)
`.github/workflows/ci.yml` runs on every push / pull request:
- `py_compile` on all modules — **catches truncated/syntax-broken files** (the
  exact failure mode that produced an empty `data/runs/`).
- asserts `generate.py` still exposes `main()`.
- a 2-hour `run_one` smoke test that must produce a labelled parquet.

A red check on GitHub means don't bother running in Colab — fix first.

## Why this avoids the truncation problem
`git clone`/`git pull` transfer atomically and verify object integrity, so a file
is never half-written the way a manual Drive upload can be. If you ever do change a
file directly in Colab, commit it back (`!git config`, `!git commit`, `!git push`
with a token) rather than copying through Drive.

## Notes
- No GPU needed (SWMM is CPU-bound) — use a standard Colab runtime.
- Keep the Colab tab active; long batches (Scenarios 1 & 4 are 5–7-day sims) can
  idle out otherwise.
- Heavy outputs go to Drive, not git. If you want them versioned, use a release
  artifact or Git LFS — don't commit raw parquet to the repo.
