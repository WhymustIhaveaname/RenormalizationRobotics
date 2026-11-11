"""Isaac flow figure, drawn by the same code as the MuJoCo ones (../figures.py).

Three PhysX schemes, all first order, all fitting (rotor inertia, K_P, K_D, command delay).
The measured joint has no rotor inertia and no delay, so those two parameters start at zero.
Each scheme gets its own counterterms, derived in exp1_pendulum/derive_second_order.py:
an explicitly held torque makes PhysX behave as semi-implicit Euler (the step map was measured,
see probe_step_map_isaac.py), while PhysX's own drive is the implicit spring of
DyCpuGpuArticulation.h, whose damping is K_D + delta K_P rather than K_D.  PGS runs that spring
at delta = h; TGS advances time inside its position iterations, so there delta = h / N.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from figures import flow_figure, use_house_style  # noqa: E402

RESULTS = ROOT / "exp1_pendulum" / "results"
KP0, KD0 = 80.0, 2.0
J0 = 0.45359237 * 0.3048 ** 2 / 3  # the measured rod inertia
BASE = np.array([0.0, KP0, KD0, 0.0])
# (key, name, sign of the inertia counterterm, number of substeps the drive sees).
# An explicitly held torque renormalizes like MuJoCo's Euler, +h K_D/2; PhysX's own drive is
# implicit and renormalizes the other way, like implicitfast.  TGS advances time inside its
# position iterations, so the drive there sees h / N rather than h.
TGS_ITERS = 32  # Isaac's default solverPositionIterationCount
SCHEMES = [("A", "explicit PD + TGS", "euler", 1),
           ("B", "PhysX drive + TGS", "drive", TGS_ITERS),
           ("C", "PhysX drive + PGS", "drive", 1)]
# Relative counterterms as (per ms, per ms^2); see derive_second_order.py for the derivation.
#   semi-implicit Euler:  J* - J = +h K_D/2,          K_D* - K_D = 0
#   PhysX force drive:    J* - J = -h K_D/2,          K_D* - K_D = -h K_P
#   both:                 the h^2 parts are (J K_P + K_D^2)/12J and K_P K_D/12J
INERTIA_C1, INERTIA_C2 = KD0 / 2 / J0 * 1e-3, (J0 * KP0 + KD0**2) / (12 * J0) / J0 * 1e-6
DAMPING_C2 = KP0 / (12 * J0) * 1e-6
DAMPING_C1 = {"euler": 0.0, "drive": -KP0 / KD0 * 1e-3}
SIGN = {"euler": +1, "drive": -1}


def theory(kind, n):
    """Counterterms per parameter index for a scheme whose drive sees h / n."""
    return {0: (SIGN[kind] * INERTIA_C1 / n, INERTIA_C2 / n**2),
            1: (0.0, 0.0),
            2: (DAMPING_C1[kind] / n, DAMPING_C2 / n**2),
            3: (0.5 / n, 0.0)}
# (parameter index, y label, unit).  The theory comes from theory() above.
ABSOLUTE = [(0, r"rotor inertia  ($10^{-3}$ kg m$^2$)", 1e-3),
            (2, r"$\Delta K_D = K_D^\star - K_D$  ($10^{-2}$ N m s/rad)", 1e-2),
            (1, r"$\Delta K_P = K_P^\star - K_P$  (N m/rad)", 1.0),
            (3, r"command delay $d^\star$ (ms)", 1e-3)]
# Each shift as a fraction of the measured value. The joint carries no rotor inertia of its own,
# so that row is normalized by the rod inertia J, exactly as the MuJoCo figure's row is.
RELATIVE = [(0, r"rotor inertia $/\, J$", J0),
            (2, r"$\Delta K_D / K_D$", KD0),
            (1, r"$\Delta K_P / K_P$", KP0),
            (3, r"command delay $d^\star$ (ms)", 1e-3)]
ROWS = RELATIVE
KP_ROW = 2  # K_P is the third row of the figure

CAPTION = ("Isaac Sim 5.1 / PhysX, single pendulum, on the same $h$ grid as the MuJoCo figure.\n"
           "First-order theory (solid); fit (dashed), $c_1 h + c_2 h^2$ through the origin, so "
           "$c_1$ is the measured first-order slope.\n"
           "An explicitly held torque makes PhysX behave as semi-implicit Euler; its own drive "
           "is an implicit spring whose damping is $K_D + \\delta K_P$, run at "
           f"$\\delta = h/{TGS_ITERS}$ under TGS and $\\delta = h$ under PGS.")


def column(key, name, kind, substeps, stem):
    rows = sorted(json.loads((RESULTS / f"{stem}_{key}.json").read_text()), key=lambda r: r["h"])
    rows = [r for r in rows if r.get("f_ctrl_nominal", F_CTRL_USED[0]) in F_CTRL_USED]
    h = np.array([r["h"] * 1e3 for r in rows])
    shift = np.array([r["params_identified"] for r in rows]) - BASE
    t = theory(kind, substeps)
    out = {"title": f"{name}\nfitted: rotor inertia, $K_P$, $K_D$, $d$", "h": h,
           "rows": [(shift[:, k] / unit, t[k]) for k, _, unit in ROWS],
           "rmse": np.array([r["rmse_identified"] for r in rows])}
    if "f_ctrl_nominal" in rows[0]:   # the MuJoCo grid runs several control rates per h
        out["groups"] = np.array([r["f_ctrl_nominal"] for r in rows])
    return out


# Only the rates whose fits are trustworthy.  At 10 and 20 Hz the pendulum settles within each
# control period (settling time 2J/K_D = 14 ms against a 100 ms period), so most samples carry no
# information about the parameters and the fit lands in a local minimum - its post-fit RMSE
# there is five times worse than at the faster rates.  The 200 Hz chain was not finished.
F_CTRL_USED = (50, 100)
F_CTRL_COLOR = {50: "#2a78d6", 100: "#104281"}


def main(stem="isaac_single_grid", absolute=False):
    global ROWS
    ROWS = ABSOLUTE if absolute else RELATIVE
    use_house_style()
    columns = [column(*scheme, stem) for scheme in SCHEMES]
    h_max = max(c["h"].max() for c in columns)
    f_ticks = [(f, t) for f, t in ((4000, "4k"), (1000, "1k"), (500, "500"), (400, "400"),
                                   (200, "200")) if 1e3 / f <= h_max]
    # Show K_P on the scale the same scheme's rotor inertia moves on, where it is a flat line.
    ranges = []
    for col in columns:
        y = np.asarray(col["rows"][0][0])
        lo, hi = min(0.0, y.min()), max(0.0, y.max())
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    stem_out = "fig_flow_isaac_single_absolute" if absolute else "fig_flow_isaac_single"
    graded = "groups" in columns[0]
    flow_figure(columns, [label for _, label, _ in ROWS],
                [RESULTS / f"{stem_out}.{e}" for e in ("png", "pdf")],
                h_max, CAPTION, f_ticks, quadratic=True, show_rmse=False,
                no_fit=() if absolute else (KP_ROW,),
                row_ylim=None if absolute else {KP_ROW: ranges},
                point_colors=F_CTRL_COLOR if graded else None,
                legend_entries=[(rf"$f_\mathrm{{ctrl}} \approx$ {fc} Hz", c)
                                for fc, c in F_CTRL_COLOR.items()] if graded else None)
    print("wrote", RESULTS / f"{stem_out}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default="isaac_single_grid")
    ap.add_argument("--absolute", action="store_true",
                    help="plot the raw shifts instead of fractions of the measured value")
    a = ap.parse_args()
    main(a.stem, a.absolute)
