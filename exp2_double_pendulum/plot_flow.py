"""Flow of the effective parameters with the physics time step, for each integration scheme.

Arms A and C identify a command delay d as well; arm B uses (armature, K_P, K_D).
The drawing itself lives in the shared ../figures.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (ARM_LABEL, ARMS, KD0, M11_MIN, PARAM0, first_order_prediction,
                    first_order_prediction_delay)
from figures import flow_figure, use_house_style

RESULTS = Path(__file__).parent / "results"
H_MAX = 5.0  # ms; the grid is h = 0.25, 0.5, ..., 5 ms, the same as experiment 1
F_TICKS = [(4000, "4k"), (1000, "1k"), (500, "500"), (400, "400"), (200, "200")]
# Shifts as fractions of the measured value, as in experiment 1.  The joint carries no rotor
# inertia of its own, so that row is divided by the smallest joint-space inertia the
# configuration reaches, which is the scale the counterterm has to be compared against.
ROWS = [(0, r"rotor inertia $/\, M_{11}^{\min}$", M11_MIN),
        (2, r"$\Delta K_D / K_D$", KD0),
        (3, r"command delay $d^\star$ (ms)", 1e-3)]
KD_ROW = 1  # figure rows: inertia, K_D, delay
F_CTRL_COLOR = {50: "#5598e7", 100: "#2a78d6", 200: "#104281"}

CAPTION = ("Target $h$ = 0.25, 0.5, ..., 5 ms ($f$ = 200-4000 Hz), realized with integer "
           "$f_\\mathrm{ctrl}$ (Hz) and integer decimation.\n"
           "One parameter set is fitted to five drive frequencies at once, over 0.1 s, both joints "
           "in the residual.\n"
           "$K_P$ is held at its measured value.\n"
           "First-order theory (solid); fit (dashed), $c_1 h + c_2 h^2$ through the origin, so "
           "$c_1$ is the measured first-order slope.")


def load(name):
    return json.loads((RESULTS / name).read_text())


def column(arm, rows, with_delay):
    rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["h"])
    predict = first_order_prediction_delay if with_delay else first_order_prediction
    base = np.append(PARAM0, 0.0) if with_delay else PARAM0
    h = np.array([r["h"] * 1e3 for r in rows])
    shift = np.array([r["params_identified"] for r in rows]) - base
    fitted = "armature, $K_D$, $d$" if with_delay else "armature, $K_D$"
    out = []
    for k, _, unit in ROWS:
        if k >= len(base):
            out.append(None)
        else:
            out.append((shift[:, k] / unit, (np.array(predict(arm, 1e-3)) - base)[k] / unit))
    return {"title": f"{ARM_LABEL[arm]}\nfitted: {fitted}", "h": h, "rows": out,
            "groups": np.array([r["f_ctrl_nominal"] for r in rows]),
            "rmse": np.array([r["rmse_identified"] for r in rows])}


def main():
    use_house_style()
    plain, delay = load("flow_mujoco.json"), load("flow_mujoco_delay.json")
    sources = {"A": (delay, True), "B": (plain, False), "C": (delay, True)}
    columns = [column(arm, *sources[arm]) for arm in ARMS]
    ranges = []
    for col in columns:
        y = np.asarray(col["rows"][0][0])
        lo, hi = min(0.0, y.min()), max(0.0, y.max())
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    flow_figure(columns, [label for _, label, _ in ROWS],
                [RESULTS / f"fig_flow_mujoco.{e}" for e in ("png", "pdf")],
                H_MAX, CAPTION, F_TICKS, quadratic=True, show_rmse=False,
                # A and C have no predicted K_D flow and their points are pure scatter, so
                # only B's K_D row is fitted, and A's and C's are shown on the inertia's scale
                # rather than magnifying a residual of a few parts in a thousand.
                no_fit=((KD_ROW, 0), (KD_ROW, 2)),
                row_ylim={KD_ROW: [ranges[0], None, ranges[2]]},
                point_colors=F_CTRL_COLOR,
                legend_entries=[(rf"$f_\mathrm{{ctrl}} \approx$ {fc} Hz", c)
                                for fc, c in F_CTRL_COLOR.items()])
    print("wrote", RESULTS / "fig_flow_mujoco.png")


if __name__ == "__main__":
    main()
