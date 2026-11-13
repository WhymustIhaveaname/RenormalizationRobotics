"""Isaac behaviour figure, drawn by the same code as the MuJoCo ones (../figures.py).

Panel (a) is the trajectory error at 200 Hz for the explicit-PD scheme, panel (b) the error of
the measured, predicted and fitted parameter sets against the simulation frequency.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from figures import IDENT, MEASURED, RENORM, behaviour_figure, use_house_style  # noqa: E402

RESULTS = ROOT / "exp1_pendulum" / "results"
SCHEME, SCHEME_NAME = "A", "explicit PD + TGS"
DEMO_T = 2.0  # s shown
F_CTRL = 50
F_MAX = 1000


def main(stem="isaac_gap"):
    use_house_style()
    rows = sorted(json.loads((RESULTS / f"{stem}_{SCHEME}.json").read_text()),
                  key=lambda r: r["h"])
    z = np.load(RESULTS / "isaac_traj.npz")
    gt = z["reference"]
    t = np.arange(1, len(gt) + 1) / F_CTRL
    error = {"t": t, "xlim": DEMO_T, "ylabel": r"simulated $-$ real (rad)",
             "title": f"Isaac at $f$ = 200 Hz, {SCHEME_NAME}",
             "series": [(z[f"{SCHEME}_measured"] - gt, MEASURED, "without renormalization"),
                        (z[f"{SCHEME}_renormalized"] - gt, RENORM, "renormalized parameters")]}

    series = []
    for key, color, label in (("rmse_measured", MEASURED, "without renormalization"),
                              ("rmse_renormalized", RENORM, "renormalized (no fitting)"),
                              ("rmse_identified", IDENT, "identified (fitted)")):
        ok = [r for r in rows if r.get(key) is not None and r["f"] <= F_MAX]
        series.append(([r["f"] for r in ok], [r[key] for r in ok], color, label))
    lost = [r for r in rows if r.get("rmse_measured") is None]
    row = min(rows, key=lambda r: abs(r["f"] - 200))
    lo, hi = row["rmse_renormalized"], row["rmse_measured"]
    gap = {"series": series, "ylim": (1e-6, 1),
           "ylabel": "trajectory RMSE, all five commands (rad)",
           "diverge_f": max(r["f"] for r in lost) if lost else None,
           "annotation": (f"{hi / lo:.0f}$\\times$ smaller,\nwithout fitting", row["f"], lo, hi)
           if lo and hi else None}

    behaviour_figure(error, gap,
                     [RESULTS / f"fig_behaviour_isaac_single.{e}" for e in ("png", "pdf")], F_MAX)
    print("wrote", RESULTS / "fig_behaviour_isaac_single.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "isaac_gap")
