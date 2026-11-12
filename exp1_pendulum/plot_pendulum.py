"""The pendulum result: what a finite-frequency simulator gets wrong, and what that costs.

RK4 with the PD torque held over each physics step, i.e. the paper's delay model with dt = h/2.
The drawing itself lives in the shared ../figures.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import F_REF, PARAM0, Sim, ground_truth, ref_sequence
from figures import IDENT, MEASURED, RENORM, behaviour_figure, use_house_style
from run_gap import F_CTRL, renormalized

ARM = "B"
RESULTS = Path(__file__).parent / "results"
DEMO_DEC = 4  # h = 5 ms: 200 Hz physics, the legged_gym default
DEMO_T = 2.0  # s of trajectory to show
F_MAX = 1000  # Hz; the practical range for robot simulators

ROWS = sorted((r for r in json.loads((RESULTS / "gap_mujoco.json").read_text())["main"]
               if r["arm"] == ARM), key=lambda r: r["h"])


def main():
    use_house_style()
    row = next(r for r in ROWS if r["dec"] == DEMO_DEC)
    sim = Sim(ARM)
    refs = ref_sequence(F_CTRL, F_REF)
    gt = ground_truth(refs, F_CTRL, DEMO_DEC)
    error = {"t": np.arange(1, len(gt) + 1) * row["h"], "xlim": DEMO_T,
             "ylabel": r"simulated $-$ real (rad)",
             "title": "Simulation error at $f$ = 200 Hz",
             "series": [(sim.rollout(refs, F_CTRL, DEMO_DEC, *p) - gt, c, n)
                        for p, c, n in ((PARAM0, MEASURED, "measured parameters"),
                                        (renormalized(ARM, row["h"]), RENORM,
                                         "renormalized parameters"))]}

    series = []
    for key, color, label in (("rmse_measured", MEASURED, "measured"),
                              ("rmse_renormalized", RENORM, "renormalized (no fitting)"),
                              ("rmse_identified", IDENT, "identified (fitted)")):
        ok = [r for r in ROWS if r[key] is not None and r["f"] <= F_MAX]
        series.append(([r["f"] for r in ok], [r[key] for r in ok], color, label))
    lost = [r for r in ROWS if r["rmse_measured"] is None]
    lo, hi = row["rmse_renormalized"], row["rmse_measured"]
    gap = {"series": series, "ylim": (1e-6, 1),
           "ylabel": "trajectory RMSE, all five commands (rad)",
           "diverge_f": max(r["f"] for r in lost) if lost else None,
           "annotation": (f"{hi / lo:.0f}$\\times$ smaller,\nwithout fitting", row["f"], lo, hi)}

    behaviour_figure(error, gap,
                     [RESULTS / f"fig_pendulum_mujoco.{e}" for e in ("png", "pdf")], F_MAX)
    print("wrote", RESULTS / "fig_pendulum_mujoco.png")


if __name__ == "__main__":
    main()
