"""
hydraulics.py
=============
Per-timestep hydraulic feature helpers.

The classifier's two physical sensors are an ultrasonic depth gauge and a
flow meter whose core engineered feature is shear velocity. We therefore derive,
from the SWMM link state (flow Q, flow depth y, flow area A) and the pipe
diameter D:

  * mean section velocity     V  = Q / A
  * hydraulic radius          R  (partially-full circular geometry)
  * shear velocity            u* = sqrt(g R S_f),  S_f = (n V / R^(2/3))^2
                                 => u* = sqrt(g) * n * |V| / R^(1/6)
  * filling ratio             y / D

Shear velocity here is a Manning-friction estimate (uniform-flow assumption),
not a measured boundary shear. It is a defensible proxy for the flow-meter
feature; the uniform-flow assumption is a known limitation under strong backwater
(exactly the blockage case), so it is exported as an explicit, labelled proxy.
"""
from __future__ import annotations
import math

G = 9.81


def circular_geometry(y: float, D: float) -> tuple:
    """Return (area, wetted_perimeter, hydraulic_radius) for depth y in pipe D."""
    if D <= 0 or y <= 0:
        return 0.0, 0.0, 0.0
    if y >= D:
        A = math.pi * D * D / 4.0
        P = math.pi * D
        return A, P, A / P
    theta = 2.0 * math.acos(1.0 - 2.0 * y / D)      # central angle (rad)
    A = (D * D / 8.0) * (theta - math.sin(theta))
    P = (D / 2.0) * theta
    R = A / P if P > 0 else 0.0
    return A, P, R


def shear_velocity(V: float, R: float, n: float) -> float:
    """Manning-friction shear-velocity proxy u* = sqrt(g) n |V| / R^(1/6)."""
    if R <= 0:
        return 0.0
    return math.sqrt(G) * n * abs(V) / (R ** (1.0 / 6.0))


def channel_features(Q: float, area: float, depth: float, D: float, n: float) -> dict:
    """Build the per-conduit feature dict from raw SWMM link state."""
    V = Q / area if area > 1e-9 else 0.0
    _, _, R = circular_geometry(depth, D)
    ustar = shear_velocity(V, R, n)
    fill = min(depth / D, 1.5) if D > 0 else 0.0
    froude = abs(V) / math.sqrt(G * (area / max(_top_width(depth, D), 1e-6))) \
        if area > 1e-9 else 0.0
    return dict(flow=Q, depth=depth, vel=V, ushear=ustar, fill=fill, froude=froude)


def _top_width(y: float, D: float) -> float:
    if D <= 0 or y <= 0 or y >= D:
        return D
    return 2.0 * math.sqrt(max(y * (D - y), 0.0))
