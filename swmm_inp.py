"""
swmm_inp.py
===========
Lightweight EPA-SWMM .inp parsing + a blockage-injection transform for the
Bellinge model.

Blockage mechanism (per design decision): *cross-section reduction* implemented
as a controllable inline orifice. We split the target conduit with a tiny extra
junction and place a SIDE orifice (rectangular opening) on the short downstream
stub. The orifice `setting` (0-1) scales the open area linearly, so a runtime
ramp of `setting` is a gradual reduction of effective cross-section -> backwater
upstream + reduced downstream flow, i.e. the blockage signature.

Topology-consistency control (methodological robustness):
The orifice is inserted in EVERY run (blockage and non-blockage alike) at the
same sampled candidate location, and simply held OPEN for non-blockage runs.
This decorrelates network topology from the class label so a model cannot learn
the shortcut "extra-orifice-here => blockage".

Severity->setting mapping:
We define blockage `severity` s in [0,1] as the fraction of the *pipe* flow area
that is removed. With a RECT_CLOSED orifice opening of height=width=D (D = pipe
diameter), open area = setting * D^2. Matching open area to a target fraction of
the circular pipe area (pi*D^2/4):
    setting(s) = (1 - s) * pi / 4          (~0.785 at s=0, ~0.0785 at s=0.9)
So s=0 reproduces ~the pipe's own area (minimal added restriction) and s->1
chokes it almost shut. This mapping is exact for area, documented, and applied
identically across all runs.
"""
from __future__ import annotations
import datetime as _dt
import math
import re
from dataclasses import dataclass, field

PIPE_AREA_SETTING_AT_ZERO = math.pi / 4.0  # open-area setting that ~matches pipe area


def severity_to_setting(severity: float) -> float:
    """Map blockage severity (fraction of pipe area removed) -> orifice setting."""
    s = max(0.0, min(1.0, severity))
    return max(0.01, (1.0 - s) * PIPE_AREA_SETTING_AT_ZERO)


# --------------------------------------------------------------------------- #
# INP parsing
# --------------------------------------------------------------------------- #
@dataclass
class InpModel:
    text: str
    sections: dict = field(default_factory=dict)  # name -> list of (lineno, raw)

    @classmethod
    def load(cls, path: str) -> "InpModel":
        with open(path, "r") as fh:
            text = fh.read()
        return cls(text=text).reindex()

    def reindex(self) -> "InpModel":
        self.sections = {}
        cur = None
        for i, line in enumerate(self.text.splitlines()):
            m = re.match(r"\s*\[(.+?)\]\s*$", line)
            if m:
                cur = m.group(1).upper()
                self.sections[cur] = []
            elif cur is not None:
                self.sections[cur].append((i, line))
        return self

    def _data_rows(self, section: str):
        """Yield non-comment, non-blank rows of a section as token lists."""
        for _, raw in self.sections.get(section.upper(), []):
            s = raw.strip()
            if not s or s.startswith(";"):
                continue
            yield s.split()

    # ---- specific extractors ---------------------------------------------- #
    def node_inverts(self) -> dict:
        inv = {}
        for tok in self._data_rows("JUNCTIONS"):
            inv[tok[0]] = (float(tok[1]), float(tok[2]) if len(tok) > 2 else 0.0)  # elev, maxdepth
        for tok in self._data_rows("OUTFALLS"):
            inv[tok[0]] = (float(tok[1]), 0.0)
        for tok in self._data_rows("STORAGE"):
            inv[tok[0]] = (float(tok[1]), float(tok[2]) if len(tok) > 2 else 0.0)
        return inv

    def conduits(self) -> dict:
        out = {}
        for tok in self._data_rows("CONDUITS"):
            out[tok[0]] = dict(name=tok[0], n1=tok[1], n2=tok[2],
                               length=float(tok[3]), rough=float(tok[4]))
        return out

    def xsection_diam(self) -> dict:
        out = {}
        for tok in self._data_rows("XSECTIONS"):
            shape = tok[1].upper()
            geom1 = float(tok[2]) if len(tok) > 2 else 0.0
            out[tok[0]] = dict(shape=shape, geom1=geom1)
        return out

    def coordinates(self) -> dict:
        out = {}
        for tok in self._data_rows("COORDINATES"):
            try:
                out[tok[0]] = (float(tok[1]), float(tok[2]))
            except (ValueError, IndexError):
                pass
        return out

    def conduit_graph(self) -> dict:
        """Adjacency on nodes via conduits: node -> list of (neighbor, conduit, direction)."""
        g = {}
        for c in self.conduits().values():
            g.setdefault(c["n1"], []).append((c["n2"], c["name"], "down"))
            g.setdefault(c["n2"], []).append((c["n1"], c["name"], "up"))
        return g


# --------------------------------------------------------------------------- #
# Section text editing
# --------------------------------------------------------------------------- #
def _append_to_section(text: str, section: str, new_lines: list) -> str:
    """Insert lines at the end of the given [SECTION] block."""
    lines = text.splitlines()
    hdr = f"[{section}]"
    start = next((i for i, l in enumerate(lines) if l.strip().upper() == hdr.upper()), None)
    if start is None:
        # create the section at end of file
        lines += ["", hdr] + new_lines
        return "\n".join(lines) + "\n"
    # find end of section (next [..] header or EOF)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"\s*\[.+?\]\s*$", lines[j]):
            end = j
            break
    # back up over trailing blank lines
    insert_at = end
    while insert_at - 1 > start and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines[insert_at:insert_at] = new_lines
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Blockage injection
# --------------------------------------------------------------------------- #
@dataclass
class BlockageHandle:
    orifice_name: str
    target_conduit: str
    mid_node: str
    diameter: float
    n1: str
    n2: str


def inject_inline_orifice(model: InpModel, target_conduit: str) -> tuple:
    """Return (new_inp_text, BlockageHandle). Idempotent per fresh model load."""
    conduits = model.conduits()
    if target_conduit not in conduits:
        raise KeyError(f"target conduit {target_conduit!r} not in model")
    c = conduits[target_conduit]
    n1, n2 = c["n1"], c["n2"]
    xs = model.xsection_diam().get(target_conduit, {})
    D = xs.get("geom1", 0.0)
    if D <= 0:
        raise ValueError(f"could not read diameter for {target_conduit}")
    inv = model.node_inverts()
    e1 = inv.get(n1, (0.0, 1.0))
    e2 = inv.get(n2, (0.0, 1.0))
    mid = f"BLKM_{target_conduit}"
    mid_invert = e1[0] + (e2[0] - e1[0]) * 0.95
    mid_maxdepth = max(e1[1], D + 0.5)

    text = model.text

    # 1) shorten target conduit to end at mid node
    def _short_conduit(m):
        toks = m.group(0).split()
        toks[2] = mid                     # To Node -> mid
        toks[3] = f"{float(toks[3]) * 0.95:.6g}"  # length * 0.95
        return "  ".join(toks)
    pat = re.compile(rf"^{re.escape(target_conduit)}\s+\S+.*$", re.MULTILINE)
    text, nsub = pat.subn(_short_conduit, text, count=1)
    if nsub != 1:
        raise RuntimeError("failed to rewrite target conduit row")

    # 2) add mid junction
    text = _append_to_section(
        text, "JUNCTIONS",
        [f"{mid:<16} {mid_invert:<10.4g} {mid_maxdepth:<10.4g} 0          0          0"])

    # 3) add inline orifice mid -> n2 (SIDE gate, opening scales linearly with setting)
    orf = f"BLK_{target_conduit}"
    text = _append_to_section(
        text, "ORIFICES",
        [f"{orf:<16} {mid:<16} {n2:<16} SIDE         0.00       0.65       NO       0"])

    # 4) orifice opening cross-section: RECT_CLOSED height=width=D (linear area in setting)
    text = _append_to_section(
        text, "XSECTIONS",
        [f"{orf:<16} RECT_CLOSED  {D:<16.4g} {D:<10.4g} 0          0"])

    # 5) optional map coordinate for mid node (interpolated)
    coords = model.coordinates()
    if n1 in coords and n2 in coords:
        x = coords[n1][0] + (coords[n2][0] - coords[n1][0]) * 0.95
        y = coords[n1][1] + (coords[n2][1] - coords[n1][1]) * 0.95
        text = _append_to_section(text, "COORDINATES", [f"{mid:<16} {x:<18.4f} {y:<18.4f}"])

    return text, BlockageHandle(orf, target_conduit, mid, D, n1, n2)


# --------------------------------------------------------------------------- #
# Simulation options + rain file rewiring
# --------------------------------------------------------------------------- #
def set_simulation_window(text: str, start_dt, end_dt, report_step_s=60, routing_step_s=4) -> str:
    """Rewrite [OPTIONS] dates/times for a run window."""
    def repl(key, value):
        nonlocal text
        text = re.sub(rf"(?m)^({re.escape(key)}\s+).*$", rf"\g<1>{value}", text, count=1)
    repl("START_DATE", start_dt.strftime("%m/%d/%Y"))
    repl("START_TIME", start_dt.strftime("%H:%M:%S"))
    repl("REPORT_START_DATE", start_dt.strftime("%m/%d/%Y"))
    repl("REPORT_START_TIME", start_dt.strftime("%H:%M:%S"))
    repl("END_DATE", end_dt.strftime("%m/%d/%Y"))
    repl("END_TIME", end_dt.strftime("%H:%M:%S"))
    rs = f"{report_step_s // 3600:02d}:{(report_step_s % 3600)//60:02d}:{report_step_s % 60:02d}"
    repl("REPORT_STEP", rs)
    repl("WET_STEP", rs)
    repl("ROUTING_STEP", f"0:00:{routing_step_s:02d}")
    return text


def set_rain_file(text: str, dat_filename: str) -> str:
    """Point every RAINGAGES FILE source at our generated .dat (basename only)."""
    def repl(m):
        return re.sub(r'"[^"]*\.dat"', f'"{dat_filename}"', m.group(0))
    lines = text.splitlines()
    out, in_rg = [], False
    for l in lines:
        if re.match(r"\s*\[.+?\]\s*$", l):
            in_rg = l.strip().upper() == "[RAINGAGES]"
        if in_rg and ".dat" in l.lower():
            l = repl(re.match(".*", l))
        out.append(l)
    return "\n".join(out) + "\n"


def raingage_names(model: InpModel) -> list:
    return [tok[0] for tok in model._data_rows("RAINGAGES")]


def node_is_outfall(model: InpModel, node: str) -> bool:
    return any(tok[0] == node for tok in model._data_rows("OUTFALLS"))


def add_inflow_surge(text: str, node: str, ts_name: str, breakpoints: list, start_dt) -> str:
    """Add a direct external inflow (CMS) at `node`, driven by a time series.

    Used for Scenario 5 (non-blockage backwater): a transient surge injected just
    downstream of the sensor raises the local HGL and backs water up the target
    conduit (depth rises, flow falls) WITHOUT any blockage.

    breakpoints: list of (minute_from_start, value_cms); SWMM interpolates linearly.
    """
    ts_lines = []
    for minute, val in breakpoints:
        t = start_dt + _dt.timedelta(minutes=int(minute))
        ts_lines.append(f"{ts_name:<16} {t:%m/%d/%Y} {t:%H:%M}  {val:.5f}")
    text = _append_to_section(text, "TIMESERIES", ts_lines)
    text = _append_to_section(
        text, "INFLOWS",
        [f"{node:<16} FLOW             {ts_name:<16} FLOW     1.0        1.0        0"])
    return text
