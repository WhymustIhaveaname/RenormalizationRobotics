"""Figure 1: parameter shifts and post-fit error versus physics time step h = 1/f.

Arms A and C identify a command delay d as well; arm B uses (J, K_P, K_D).
The drawing itself lives in the shared ../figures.py, so the Isaac version cannot drift from it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (ARM_LABEL, ARMS, J0, KD0, KP0, PARAM0, first_order_prediction,
                    first_order_prediction_delay, second_order_shift)
from figures import flow_figure, use_house_style

RESULTS = Path(__file__).parent / "results"
H_MAX = 5.0  # ms; the grid is h = 0.25, 0.5, ..., 5 ms
F_CTRL_COLOR = {10: "#86b6ef", 20: "#5598e7", 50: "#2a78d6", 100: "#1c5cab", 200: "#104281"}
F_TICKS = [(4000, "4k"), (1000, "1k"), (500, "500"), (400, "400"), (200, "200")]
# (parameter index, y label, unit) chosen so the theoretical slopes per ms are simple numbers
ABSOLUTE = [(0, r"$\Delta J = J^\star - J$  ($10^{-3}$ kg m$^2$)", 1e-3),
            (2, r"$\Delta K_D = K_D^\star - K_D$  ($10^{-2}$ N m s/rad)", 1e-2),
            (1, r"$\Delta K_P = K_P^\star - K_P$  (N m/rad)", 1.0),
            (3, r"command delay $d^\star$ (ms)", 1e-3)]
# The same shifts as a fraction of the measured value, so the scale is the same in any engine.
RELATIVE = [(0, r"$\Delta J / J$", J0),
            (2, r"$\Delta K_D / K_D$", KD0),
            (1, r"$\Delta K_P / K_P$", KP0),
            (3, r"command delay $d^\star$ (ms)", 1e-3)]
ROWS = RELATIVE

HEAD = ("Target $h$ = 0.25, 0.5, ..., 5 ms ($f$ = 200-4000 Hz), realized with integer "
        "$f_\\mathrm{ctrl}$ (Hz) and integer decimation, within 2.3% of the target.\n")
CAPTION = (HEAD + "Theory (solid), first order except the $K_D$ row of A and C, which carries "
           "the derived $h^2$ counterterm $K_P K_D / 12 J$.\n"
           "Fit (dashed): $c_1 h + c_2 h^2$ through the origin, so $c_1$ is the measured "
           "first-order slope.")


def load(name):
    return json.loads((RESULTS / name).read_text())


KD_ROW, KP_ROW = 2, 2  # row index into the parameter vector / into the figure's rows


def column(arm, rows, with_delay, second_order=False):
    rows = [r for r in rows if r["arm"] == arm]
    predict = first_order_prediction_delay if with_delay else first_order_prediction
    base = np.append(PARAM0, 0.0) if with_delay else PARAM0
    h = np.array([r["h"] * 1e3 for r in rows])
    shift = np.array([r["params"] for r in rows]) - base
    fitted = "$J, K_P, K_D, d$" if with_delay else "$J, K_P, K_D$"
    out = []
    for k, _, unit in ROWS:
        if k >= len(base):
            out.append(None)
        else:
            theory = (np.array(predict(arm, 1e-3)) - base)[k] / unit  # per ms
            curvature = 0.0                                           # per ms^2
            if second_order and k == KD_ROW and second_order_shift(arm, 1e-3) is not None:
                curvature = second_order_shift(arm, 1e-3)[k] / unit
            out.append((shift[:, k] / unit, (theory, curvature)))
    return {"title": f"{ARM_LABEL[arm]}\nfitted: {fitted}", "h": h, "rows": out,
            "rmse": np.array([r["rmse"] for r in rows]),
            "groups": np.array([r["f_ctrl_nominal"] for r in rows])}


def main(absolute=False):
    global ROWS
    ROWS = ABSOLUTE if absolute else RELATIVE
    use_house_style()
    plain, delay = load("flow_mujoco.json"), load("flow_mujoco_delay.json")
    sources = {"A": (delay, True), "B": (plain, False), "C": (delay, True)}
    columns = [column(arm, *sources[arm], second_order=True) for arm in ARMS]
    stem = "fig1_flow_mujoco_absolute" if absolute else "fig1_flow_mujoco"
    # Show K_P on the scale the same scheme's J moves on, sign included, where it is a flat line.
    ranges = []
    for col in columns:
        y = np.asarray(col["rows"][0][0])
        lo, hi = min(0.0, y.min()), max(0.0, y.max())
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    ylim = None if absolute else {KP_ROW: ranges}
    flow_figure(columns, [label for _, label, _ in ROWS],
                [RESULTS / f"{stem}.{e}" for e in ("png", "pdf")],
                H_MAX, CAPTION, F_TICKS,
                point_colors=F_CTRL_COLOR, quadratic=True, show_rmse=False,
                no_fit=() if absolute else (KP_ROW,), row_ylim=ylim,
                legend_entries=[(rf"$f_\mathrm{{ctrl}} \approx$ {fc} Hz", c)
                                for fc, c in F_CTRL_COLOR.items()])
    print("wrote", RESULTS / f"{stem}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--absolute", action="store_true",
                    help="plot the raw shifts instead of fractions of the measured value")
    main(ap.parse_args().absolute)
