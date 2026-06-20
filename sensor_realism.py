"""
sensor_realism.py
=================
Post-export sensor-realism layer. Applied to the clean simulator time series so
the models never train on pristine signals (and so the m-of-n fault tolerance and
the sim-to-real argument are exercised). Each effect is a documented, randomised
axis; the clean channel is always retained alongside the measured one.

Effects:
  * measurement noise  – additive Gaussian at a sampled SNR (dB)
  * drift / fouling    – slow random-walk offset (gradual bias)
  * quantisation       – rounding to finite sensor resolution
  * dropout            – random + bursty gaps, each with a *_missing flag (1=gap)

For each treated column `c` we add `c_meas` (corrupted, NaN in gaps) and
`c_missing` (gap indicator). The original `c` (clean) is kept for ablation.
"""
from __future__ import annotations
import numpy as np


def _add_noise(x, snr_db, rng):
    sig_p = np.nanmean(x ** 2) + 1e-12
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    return x + rng.normal(0, np.sqrt(noise_p), size=x.shape)


def _add_drift(x, max_drift_frac, rng):
    scale = (np.nanmax(x) - np.nanmin(x) + 1e-9) * max_drift_frac
    steps = rng.normal(0, scale / np.sqrt(len(x)), size=len(x))
    return x + np.cumsum(steps)


def _quantise(x, resolution):
    if resolution <= 0:
        return x
    return np.round(x / resolution) * resolution


def _dropout_mask(n, p_random, n_bursts, burst_len_range, rng):
    mask = rng.random(n) < p_random
    for _ in range(n_bursts):
        start = rng.integers(0, n)
        length = rng.integers(*burst_len_range)
        mask[start:start + length] = True
    return mask


def apply_realism(df, columns, rng, *,
                  snr_db_range=(20, 40),
                  drift_frac_range=(0.0, 0.05),
                  resolution_frac=0.01,
                  p_random_dropout=0.002,
                  n_bursts=3,
                  burst_len_range=(5, 30)):
    """Add `<col>_meas` and `<col>_missing` for each column. Returns df + a params dict."""
    out = df.copy()
    params = {}
    for col in columns:
        if col not in out.columns:
            continue
        x = out[col].to_numpy(dtype=float)
        snr = float(rng.uniform(*snr_db_range))
        drift_frac = float(rng.uniform(*drift_frac_range))
        res = resolution_frac * (np.nanmax(x) - np.nanmin(x) + 1e-9)
        y = _add_noise(x, snr, rng)
        y = _add_drift(y, drift_frac, rng)
        y = _quantise(y, res)
        gaps = _dropout_mask(len(x), p_random_dropout, n_bursts, burst_len_range, rng)
        y[gaps] = np.nan
        out[f"{col}_meas"] = y
        out[f"{col}_missing"] = gaps.astype(int)
        params[col] = dict(snr_db=snr, drift_frac=drift_frac, resolution=res,
                           dropout_frac=float(gaps.mean()))
    return out, params
