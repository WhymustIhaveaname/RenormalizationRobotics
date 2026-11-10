"""Flow of the effective parameters with the physics time step, for each integration scheme.

Arms A and C identify a command delay d as well; arm B uses (armature, K_P, K_D).
The drawing itself lives in the shared ../figures.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import ARM_LABEL, ARMS, PARAM0, first_order_prediction, first_order_prediction_delay
from figures import flow_figure, use_house_style

RESULTS = Path(__file__).parent / "results"
H_MAX = 20.0  # ms; the grid is h = 1, 2, ..., 20 ms
F_TICKS = [(1000, "1k"), (200, "200"), (100, "100"), (67, "67"), (50, "50")]
ROWS = [(0, r"rotor inertia  ($10^{-3}$ kg m$^2$)", 1e-3),
        (2, r"$\Delta K_D = K_D^\star - K_D$  ($10^{-2}$ N m s/rad)", 1e-2),
        (1, r"$\Delta K_P = K_P^\star - K_P$  (N m/rad)", 1.0),
        (3, r"command delay $d^\star$ (ms)", 1e-3)]

CAPTION = ("Target $h$ = 1, 2, ..., 20 ms ($f$ = 50-1000 Hz), realized with integer "
           "$f_\\mathrm{ctrl}$ (Hz) and integer decimation.\n"
           "Parameter rows: first-order theory (solid); fit (dashed), linear, or quadratic "
           "where the theory slope is 0. Last row: post-fit RMSE against $h^2$.")


def load(name):
    return json.loads((RESULTS / name).read_text())


def column(arm, rows, with_delay):
    rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["h"])
    predict = first_order_prediction_delay if with_delay else first_order_prediction
    base = np.append(PARAM0, 0.0) if with_delay else PARAM0
    h = np.array([r["h"] * 1e3 for r in rows])
    shift = np.array([r["params_identified"] for r in rows]) - base
    fitted = "armature, $K_P$, $K_D$, $d$" if with_delay else "armature, $K_P$, $K_D$"
    out = []
    for k, _, unit in ROWS:
        if k >= len(base):
            out.append(None)
        else:
            out.append((shift[:, k] / unit, (np.array(predict(arm, 1e-3)) - base)[k] / unit))
    return {"title": f"{ARM_LABEL[arm]}\nfitted: {fitted}", "h": h, "rows": out,
            "rmse": np.array([r["rmse_identified"] for r in rows])}


def main():
    use_house_style()
    plain, delay = load("flow_mujoco.json"), load("flow_mujoco_delay.json")
    sources = {"A": (delay, True), "B": (plain, False), "C": (delay, True)}
    flow_figure([column(arm, *sources[arm]) for arm in ARMS],
                [label for _, label, _ in ROWS],
                [RESULTS / f"fig_flow_mujoco.{e}" for e in ("png", "pdf")],
                H_MAX, CAPTION, F_TICKS,
                rmse_label="post-fit RMSE over 1 s (rad)")
    print("wrote", RESULTS / "fig_flow_mujoco.png")


if __name__ == "__main__":
    main()
