"""Figure 1: parameter shifts and post-fit error versus physics time step h = 1/f.

Arms A and C identify a command delay d as well; arm B uses (J, K_P, K_D).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ARM_LABEL, ARMS, PARAM0, first_order_prediction, first_order_prediction_delay

RESULTS = Path(__file__).parent / "results"
H_MAX = 5.0  # ms; the grid is h = 0.25, 0.5, ..., 5 ms
F_CTRL = (10, 20, 50, 100, 200)
F_CTRL_COLOR = {10: "#86b6ef", 20: "#5598e7", 50: "#2a78d6", 100: "#1c5cab", 200: "#104281"}
FIT_COLOR = "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (parameter index, y label, unit) chosen so the theoretical slopes per ms are simple numbers
ROWS = [(0, r"$\Delta J = J^\star - J$  ($10^{-3}$ kg m$^2$)", 1e-3),
        (2, r"$\Delta K_D = K_D^\star - K_D$  ($10^{-2}$ N m s/rad)", 1e-2),
        (1, r"$\Delta K_P = K_P^\star - K_P$  (N m/rad)", 1.0)]
DELAY_ROW = (3, r"command delay $d^\star$ (ms)", 1e-3)


def poly_fit(x, y, deg):
    """Least-squares polynomial; returns coefficients from highest degree down, and R^2."""
    coef = np.polyfit(x, y, deg)
    r2 = 1 - np.sum((y - np.polyval(coef, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef, r2


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(name):
    return json.loads((RESULTS / name).read_text())


def make_figure(sources, stem):
    """sources: {arm: (fit rows, whether the command delay was identified)}."""
    param_rows = ROWS + [DELAY_ROW]
    n_rows = len(param_rows) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.45 * n_rows))
    h_line = np.array([0, H_MAX])
    h_dense = np.linspace(0, H_MAX, 200)

    for j, arm in enumerate(ARMS):
        arm_rows, with_delay = sources[arm]
        arm_rows = [r for r in arm_rows if r["arm"] == arm]
        predict = first_order_prediction_delay if with_delay else first_order_prediction
        base = np.append(PARAM0, 0.0) if with_delay else PARAM0
        h = np.array([r["h"] * 1e3 for r in arm_rows])
        shift = np.array([r["params"] for r in arm_rows]) - base
        f_ctrl = np.array([r["f_ctrl_nominal"] for r in arm_rows])
        fitted = "$J, K_P, K_D, d$" if with_delay else "$J, K_P, K_D$"
        axes[0, j].set_title(f"{ARM_LABEL[arm]}\nfitted: {fitted}", fontsize=11, pad=14)

        for i, (k, ylabel, unit) in enumerate(param_rows):
            ax = axes[i, j]
            style(ax)
            ax.set_xlim(0, H_MAX * 1.04)
            ax.set_xlabel("$h = 1/f$ (ms)", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel)
            if k >= len(base):
                ax.text(0.5, 0.5, "not fitted:\nno command advance in this scheme", ha="center",
                        va="center", transform=ax.transAxes, color=INK2, fontsize=9)
                ax.set_yticks([])
                continue
            y = shift[:, k] / unit
            theory = (np.array(predict(arm, 1e-3)) - base)[k] / unit  # per ms
            ax.axhline(0, color=GRID, lw=1, zorder=0)
            ax.plot(h_line, theory * h_line, color=INK, lw=1.4, zorder=1,
                    label=f"theory: slope {theory:.2f}")
            if theory == 0:  # no first-order flow predicted: fit a quadratic to expose the h^2 term
                coef, r2 = poly_fit(h, y, 2)
                label = (f"fit: $c_0 + c_1 h + c_2 h^2$, $R^2$ = {r2:.4f}\n"
                         f"$c_0$ = {coef[2]:.4f}, $c_1$ = {coef[1]:.4f}, $c_2$ = {coef[0]:.4f}")
            else:
                coef, r2 = poly_fit(h, y, 1)
                label = f"fit: slope {coef[0]:.2f}, $R^2$ = {r2:.4f}"
            ax.plot(h_dense, np.polyval(coef, h_dense), color=FIT_COLOR, lw=2, ls="--", zorder=2,
                    label=label)
            for fc, color in F_CTRL_COLOR.items():
                m = f_ctrl == fc
                ax.plot(h[m], y[m], "o", ms=6, color=color, mec="white", mew=0.7, zorder=3)
            ax.legend(loc="best", frameon=False, fontsize=9, handlelength=2.2)

        top = axes[0, j].secondary_xaxis("top")
        top.set_xticks([1e3 / f for f in (4000, 1000, 500, 400, 200)])
        top.set_xticklabels(["4k", "1k", "500", "400", "200"], fontsize=8, color=INK2)
        top.set_xlabel("$f$ (Hz)", fontsize=8, color=INK2, labelpad=2)

        # Post-fit error: all first-order effects are absorbed, so RMSE is predicted to scale as h^2.
        ax = axes[-1, j]
        rmse = np.array([r["rmse"] for r in arm_rows])
        for fc, color in F_CTRL_COLOR.items():
            m = f_ctrl == fc
            ax.plot(h[m] ** 2, rmse[m], "o", ms=6, color=color, mec="white", mew=0.7, zorder=3)
        ax.text(0.03, 0.95, r"theory: RMSE $\propto h^2$", transform=ax.transAxes, va="top",
                fontsize=9, color=INK)
        ax.set_xlim(0, H_MAX**2 * 1.04)
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        ax.set_xlabel("$h^2$ (ms$^2$)", fontsize=9)
        style(ax)
        if j == 0:
            ax.set_ylabel("post-fit RMSE (rad)")

    dots = [plt.Line2D([], [], ls="none", marker="o", ms=6, color=c, mec="white")
            for c in F_CTRL_COLOR.values()]
    fig.legend(dots, [rf"$f_\mathrm{{ctrl}} \approx$ {fc} Hz" for fc in F_CTRL_COLOR], loc="upper center",
               ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.004,
             "Target $h$ = 0.25, 0.5, ..., 5 ms ($f$ = 200-4000 Hz), realized with integer $f_\\mathrm{ctrl}$ (Hz) "
             "and integer decimation, within 2.3% of the target.\n"
             "Parameter rows: first-order theory (solid); fit (dashed), linear, or quadratic where the theory slope is 0. "
             "Last row: post-fit RMSE against $h^2$.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.018, 1, 0.975))
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{stem}.{ext}", dpi=200)
    print("wrote", RESULTS / f"{stem}.png")


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})
    plain, delay = load("flow_mujoco.json"), load("flow_mujoco_delay.json")
    make_figure({"A": (delay, True), "B": (plain, False), "C": (delay, True)}, "fig1_flow_mujoco")


if __name__ == "__main__":
    main()
