"""
rainfall.py
===========
Synthetic rainfall generation, written to SWMM's external rain-file format:

    GAGE  YYYY  M  D  H  MM  VALUE     (VOLUME mm over the 1-min interval)

We use synthetic design storms rather than slicing the historical Bellinge
record so that storm intensity, duration, shape and spatial heterogeneity become
*controllable randomisation axes* (domain randomisation for the classifier).
A Chicago-style (front/centre/back-loaded triangular) hyetograph is used; total
depth and duration are sampled per run. Per-gage multipliers create non-uniform
spatial loading (heavy one side, dry the other).

`sample_storm` returns the per-minute intensity series so the labelling code can
reuse it (rain-active mask) without re-reading the file.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Storm:
    onset_min: int          # minutes after window start
    duration_min: int
    total_mm: float
    peak_frac: float        # 0=front-loaded, 0.5=centre, 1=back-loaded


def triangular_hyetograph(duration_min: int, total_mm: float, peak_frac: float) -> np.ndarray:
    """Per-minute mm volumes summing to total_mm, triangular with peak at peak_frac."""
    if duration_min <= 0 or total_mm <= 0:
        return np.zeros(max(duration_min, 0))
    t = np.arange(duration_min) + 0.5
    peak = peak_frac * duration_min
    w = np.where(t <= peak, t / max(peak, 1e-6),
                 (duration_min - t) / max(duration_min - peak, 1e-6))
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        w = np.ones(duration_min)
    return w / w.sum() * total_mm


def build_intensity_series(n_minutes: int, storms: list) -> np.ndarray:
    """Superpose storms onto a zero baseline -> per-minute mm volume array."""
    series = np.zeros(n_minutes)
    for s in storms:
        hy = triangular_hyetograph(s.duration_min, s.total_mm, s.peak_frac)
        a = s.onset_min
        b = min(a + len(hy), n_minutes)
        series[a:b] += hy[: b - a]
    return series


def write_rain_dat(path: str, start_dt, gage_series: dict) -> None:
    """gage_series: {gage_name: per-minute mm array}. All arrays same length."""
    import datetime as _dt
    lines = []
    names = list(gage_series)
    n = len(gage_series[names[0]])
    for g in names:
        arr = gage_series[g]
        for i in range(n):
            ts = start_dt + _dt.timedelta(minutes=i)
            lines.append(f"{g} {ts.year} {ts.month} {ts.day} {ts.hour} {ts.minute} "
                         f"{arr[i]:.5f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def sample_storm(rng: np.random.Generator, onset_min: int, *, extreme: bool = False) -> Storm:
    """Sample a storm. `extreme=True` biases toward near-surcharge intensities."""
    if extreme:
        duration = int(rng.integers(45, 180))
        total = float(rng.uniform(35, 70))          # mm — heavy
    else:
        duration = int(rng.integers(60, 600))       # 1-10 h
        total = float(rng.uniform(5, 30))           # mm — ordinary
    peak = float(rng.choice([0.25, 0.4, 0.5, 0.6, 0.75]))
    return Storm(onset_min=onset_min, duration_min=duration, total_mm=total, peak_frac=peak)


def gage_multipliers(rng: np.random.Generator, gages: list) -> dict:
    """Per-gage intensity multipliers for spatial heterogeneity (0.3-1.7)."""
    return {g: float(rng.uniform(0.3, 1.7)) for g in gages}
