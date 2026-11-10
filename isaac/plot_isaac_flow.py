"""Isaac flow figure, drawn by the same code as the MuJoCo ones (../figures.py).

Three PhysX schemes, all first order, all fitting (rotor inertia, K_P, K_D, command delay).
The measured joint has no rotor inertia and no delay, so those two parameters start at zero.
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
SCHEMES = [("A", "explicit PD + TGS"),
           ("B", "PhysX drive + TGS"),
           ("C", "PhysX drive + PGS")]
# (parameter index, y label, unit, theoretical slope per ms)
ABSOLUTE = [(0, r"rotor inertia  ($10^{-3}$ kg m$^2$)", 1e-3, KD0 / 2),
            (2, r"$\Delta K_D = K_D^\star - K_D$  ($10^{-2}$ N m s/rad)", 1e-2, 0.0),
            (1, r"$\Delta K_P = K_P^\star - K_P$  (N m/rad)", 1.0, 0.0),
            (3, r"command delay $d^\star$ (ms)", 1e-3, 0.5)]
# Each shift as a fraction of the measured value. The joint carries no rotor inertia of its own,
# so that row is normalized by the rod inertia J, exactly as the MuJoCo figure's row is.
RELATIVE = [(0, r"rotor inertia $/\, J$", J0, KD0 / 2 * 1e-3 / J0),
            (2, r"$\Delta K_D / K_D$", KD0, 0.0),
            (1, r"$\Delta K_P / K_P$", KP0, 0.0),
            (3, r"command delay $d^\star$ (ms)", 1e-3, 0.5)]
ROWS = ABSOLUTE

CAPTION = ("Isaac Sim 5.1 / PhysX, single pendulum, on the same $h$ grid as the MuJoCo figure.\n"
           "Parameter rows: first-order theory (solid); fit (dashed), linear, or quadratic where "
           "the theory slope is 0. Last row: post-fit RMSE against $h^2$.")


def column(key, name, stem):
    rows = sorted(json.loads((RESULTS / f"{stem}_{key}.json").read_text()), key=lambda r: r["h"])
    h = np.array([r["h"] * 1e3 for r in rows])
    shift = np.array([r["params_identified"] for r in rows]) - BASE
    return {"title": f"{name}\nfitted: rotor inertia, $K_P$, $K_D$, $d$", "h": h,
            "rows": [(shift[:, k] / unit, theory) for k, _, unit, theory in ROWS],
            "rmse": np.array([r["rmse_identified"] for r in rows])}


def main(stem="isaac_single_smallh", relative=False):
    global ROWS
    ROWS = RELATIVE if relative else ABSOLUTE
    use_house_style()
    columns = [column(k, n, stem) for k, n in SCHEMES]
    h_max = max(c["h"].max() for c in columns)
    f_ticks = [(f, t) for f, t in ((4000, "4k"), (1000, "1k"), (500, "500"), (400, "400"),
                                   (200, "200")) if 1e3 / f <= h_max]
    flow_figure(columns, [label for _, label, _, _ in ROWS],
                [RESULTS / f"fig_flow_isaac_single{'_relative' if relative else ''}.{e}"
                 for e in ("png", "pdf")],
                h_max, CAPTION, f_ticks)
    print("wrote", RESULTS / f"fig_flow_isaac_single{'_relative' if relative else ''}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default="isaac_single_smallh")
    ap.add_argument("--relative", action="store_true",
                    help="plot each shift as a fraction of the measured value")
    a = ap.parse_args()
    main(a.stem, a.relative)
