"""The pendulum result: what a finite-frequency simulator gets wrong, and what that costs.

RK4 with the PD torque held over each physics step, i.e. the paper's delay model with dt = h/2.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import F_REF, PARAM0, Sim, ground_truth, ref_sequence
from run_gap import F_CTRL, renormalized

ARM = "B"
RESULTS = Path(__file__).parent / "results"
DEMO_DEC = 4  # h = 5 ms: 200 Hz physics, the legged_gym default
DEMO_T = 2.0  # s of trajectory to show
F_MAX = 1000  # Hz; the practical range for robot simulators
MEASURED, RENORM, IDENT = "#eb6834", "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

ROWS = sorted((r for r in json.loads((RESULTS / "gap_mujoco.json").read_text())["main"]
               if r["arm"] == ARM), key=lambda r: r["h"])


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_error(ax):
    """What a 200 Hz simulator gets wrong on the 2 Hz command, and what renormalization fixes."""
    row = next(r for r in ROWS if r["dec"] == DEMO_DEC)
    sim = Sim(ARM)
    refs = ref_sequence(F_CTRL, F_REF)
    gt = ground_truth(refs, F_CTRL, DEMO_DEC)
    t = np.arange(1, len(gt) + 1) * row["h"]
    for params, color, name in ((PARAM0, MEASURED, "measured parameters"),
                                (renormalized(ARM, row["h"]), RENORM, "renormalized parameters")):
        ax.plot(t, sim.rollout(refs, F_CTRL, DEMO_DEC, *params) - gt, color=color, lw=1.4,
                label=name)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xlim(0, DEMO_T)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"simulated $-$ real (rad)")
    ax.set_title("Simulation error at $f$ = 200 Hz", fontsize=11.5)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    style(ax)


def panel_gap(ax):
    """The error of each parameter set across simulation frequencies."""
    for key, color, label in (("rmse_measured", MEASURED, "measured"),
                              ("rmse_renormalized", RENORM, "renormalized (no fitting)"),
                              ("rmse_identified", IDENT, "identified (fitted)")):
        ok = [r for r in ROWS if r[key] is not None and r["f"] <= F_MAX]
        ax.plot([r["f"] for r in ok], [r[key] for r in ok], "o-", ms=4, lw=1.5,
                color=color, label=label)
    lost = [r for r in ROWS if r["rmse_measured"] is None]
    if lost:
        f0 = max(r["f"] for r in lost)
        ax.plot([f0], [0.1], "x", ms=9, mew=2, color=MEASURED)
        ax.annotate("diverges", (f0, 0.1), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=MEASURED)

    row = next(r for r in ROWS if r["dec"] == DEMO_DEC)
    lo, hi = row["rmse_renormalized"], row["rmse_measured"]
    ax.annotate("", (row["f"], lo), (row["f"], hi),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.annotate(f"{hi / lo:.0f}$\\times$ smaller,\nwithout fitting", (row["f"], np.sqrt(lo * hi)),
                xytext=(10, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlim(0, F_MAX * 1.02)
    ax.set_xlabel("simulation frequency $f$ (Hz)")
    ax.set_ylabel("trajectory RMSE, all five commands (rad)")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    style(ax)


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panel_error(axes[0])
    panel_gap(axes[1])
    for ax, letter in zip(axes, "ab"):
        ax.annotate(f"({letter})", (0, 1), xycoords="axes fraction", xytext=(-48, 14),
                    textcoords="offset points", fontsize=12, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_pendulum_mujoco.{ext}", dpi=200)
    print("wrote", RESULTS / "fig_pendulum_mujoco.png")


if __name__ == "__main__":
    main()
