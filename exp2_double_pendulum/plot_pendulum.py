"""The double pendulum result: how long the simulator keeps the real trajectory, and at what error."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import F_CTRL, PARAM0, T_GAP, Sim, ref_sequence, renormalized

RESULTS = Path(__file__).parent / "results"
DEMO_DEC = 4  # 200 Hz physics, the legged_gym default
DEMO_T = 2.5  # s
F_MAX = 1000  # Hz
MEASURED, RENORM, IDENT = "#eb6834", "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

ROWS = json.loads((RESULTS / "gap_mujoco.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_error(ax):
    """How the simulation error on the free joint grows once chaos amplifies it."""
    row = next(r for r in ROWS if r["dec"] == DEMO_DEC)
    sim = Sim()
    refs = ref_sequence(DEMO_T)
    gt = sim.ground_truth(refs)
    t = np.arange(1, len(refs) + 1) / F_CTRL
    for params, color, name in ((PARAM0, MEASURED, "measured parameters"),
                                (renormalized(row["h"]), RENORM, "renormalized parameters")):
        theta = sim.rollout(refs, DEMO_DEC, *params, strict=False)
        ax.plot(t, theta[:, 1] - gt[:, 1], color=color, lw=1.4, label=name)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xlim(0, DEMO_T)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\theta_2$ simulated $-$ real (rad)")
    ax.set_title("Free joint at $f$ = 200 Hz", fontsize=11.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    style(ax)


def panel_gap(ax):
    rows = [r for r in ROWS if r["f"] <= F_MAX]
    for key, color, label in (("rmse_measured", MEASURED, "measured"),
                              ("rmse_renormalized", RENORM, "renormalized (no fitting)"),
                              ("rmse_identified", IDENT, "identified (fitted)")):
        ok = [r for r in rows if r[key] is not None]   # a diverged run has no error to plot
        ax.plot([r["f"] for r in ok], [r[key] for r in ok], "o-", ms=4, lw=1.5,
                color=color, label=label)
    row = next(r for r in rows if r["dec"] == DEMO_DEC)
    lo, hi = row["rmse_renormalized"], row["rmse_measured"]
    ax.annotate("", (row["f"], lo), (row["f"], hi),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.annotate(f"{hi / lo:.0f}$\\times$ smaller,\nwithout fitting", (row["f"], np.sqrt(lo * hi)),
                xytext=(10, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
    ax.set_yscale("log")
    ax.set_xlim(0, F_MAX * 1.02)
    ax.set_xlabel("simulation frequency $f$ (Hz)")
    ax.set_ylabel(f"joint-angle RMSE over {T_GAP:g} s (rad)")
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
        fig.savefig(RESULTS / f"fig_double_pendulum_mujoco.{ext}", dpi=200)
    print("wrote", RESULTS / "fig_double_pendulum_mujoco.png")


if __name__ == "__main__":
    main()
