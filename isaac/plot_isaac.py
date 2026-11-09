"""Figures for the Isaac Sim (PhysX) runs, laid out like the MuJoCo ones.

Run in the project venv:  python plot_isaac.py --exp single|double
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KP0, KD0 = 80.0, 2.0
ARMS = ("A", "B", "C")
ARM_LABEL = {"A": "A: TGS + explicit PD",
             "B": "B: TGS + PhysX joint drive",
             "C": "C: PGS + PhysX joint drive"}
MEASURED, RENORM, IDENT, FIT_COLOR = "#eb6834", "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
F_MAX = 1000


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def poly_fit(x, y, deg):
    coef = np.polyfit(x, y, deg)
    r2 = 1 - np.sum((y - np.polyval(coef, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef, r2


def load(exp):
    d = ROOT / ("exp1_pendulum" if exp == "single" else "exp2_double_pendulum") / "results"
    out = {}
    for arm in ARMS:
        f = d / f"isaac_{exp}_{arm}.json"
        if f.exists():
            out[arm] = sorted(json.loads(f.read_text()), key=lambda r: r["h"])
    return d, out


def flow_figure(exp, results, outdir):
    """Rotor inertia, gain shifts and post-fit error against the physics time step."""
    rows = [(0, r"rotor inertia  ($10^{-3}$ kg m$^2$)", 1e-3, KD0 / 2),
            (2, r"$\Delta K_D$  ($10^{-2}$ N m s/rad)", 1e-2, KP0 / 2 * 1e-1),
            (1, r"$\Delta K_P$  (N m/rad)", 1.0, 0.0)]
    h_max = max(r["h"] for rs in results.values() for r in rs) * 1e3
    fig, axes = plt.subplots(4, 3, figsize=(13, 13.5))
    h_dense = np.linspace(0, h_max, 200)

    for j, arm in enumerate(ARMS):
        rs = results.get(arm, [])
        axes[0, j].set_title(ARM_LABEL[arm], fontsize=11, pad=10)
        if not rs:
            for i in range(4):
                axes[i, j].text(0.5, 0.5, "not available", ha="center", va="center",
                                transform=axes[i, j].transAxes, color=INK2)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            continue
        h = np.array([r["h"] for r in rs]) * 1e3
        P = np.array([r["params_identified"] for r in rs]) - np.array([0.0, KP0, KD0])
        for i, (k, ylabel, unit, theory) in enumerate(rows):
            ax = axes[i, j]
            y = P[:, k] / unit
            ax.axhline(0, color=GRID, lw=1, zorder=0)
            ax.plot([0, h_max], [0, theory * h_max], color=INK, lw=1.4, zorder=1,
                    label=f"theory: slope {theory:.2f}")
            deg = 2 if theory == 0 else 1
            coef, r2 = poly_fit(h, y, deg)
            label = (f"fit: slope {coef[0]:.2f}, $R^2$ = {r2:.4f}" if deg == 1 else
                     f"fit: quadratic, $R^2$ = {r2:.4f}")
            ax.plot(h_dense, np.polyval(coef, h_dense), color=FIT_COLOR, lw=2, ls="--",
                    zorder=2, label=label)
            ax.plot(h, y, "o", ms=6, color=IDENT, mec="white", mew=0.7, zorder=3)
            ax.set_xlim(0, h_max * 1.04)
            ax.set_xlabel("$h = 1/f$ (ms)", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.legend(loc="best", frameon=False, fontsize=9)
            style(ax)

        ax = axes[3, j]
        e = np.array([r["rmse_identified"] for r in rs])
        ax.plot(h**2, e, "o", ms=6, color=IDENT, mec="white", mew=0.7)
        ax.text(0.04, 0.95, r"theory: RMSE $\propto h^2$", transform=ax.transAxes, va="top",
                fontsize=9, color=INK)
        ax.set_xlim(0, h_max**2 * 1.04)
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        ax.set_xlabel("$h^2$ (ms$^2$)", fontsize=9)
        if j == 0:
            ax.set_ylabel("post-fit RMSE (rad)")
        style(ax)

    fig.text(0.5, 0.004, "Isaac Sim 5.1 / PhysX. Parameter rows: first-order theory (solid), "
             "fit (dashed). Last row: post-fit RMSE against $h^2$.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_flow_isaac_{exp}.{ext}", dpi=200)
    print("wrote", outdir / f"fig_flow_isaac_{exp}.png")


def behaviour_figure(exp, results, outdir):
    """Trajectory error at 200 Hz, and the error of each parameter set across frequencies."""
    traj_file = outdir / "isaac_traj.npz"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    if traj_file.exists():
        z = np.load(traj_file)
        ref = z["reference"]
        t = np.arange(1, len(ref) + 1) / 50
        col = 1 if ref.ndim > 1 else None
        for name, color in (("measured", MEASURED), ("renormalized", RENORM)):
            key = f"A_{name}"
            if key not in z:
                continue
            err = z[key] - ref
            ax.plot(t, err[:, col] if col is not None else err, color=color, lw=1.4,
                    label=f"{name} parameters")
        ax.axhline(0, color=INK, lw=1.2)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("time (s)")
        ax.set_ylabel((r"$\theta_2$ " if col is not None else "") + "simulated $-$ real (rad)")
        ax.set_title("Isaac at $f$ = 200 Hz, explicit PD", fontsize=11.5)
        ax.legend(loc="best", frameon=False, fontsize=9.5)
    else:
        ax.text(0.5, 0.5, "run dump_traj.py", ha="center", va="center", transform=ax.transAxes)
    style(ax)

    ax = axes[1]
    rs = [r for r in results.get("A", []) if r["f"] <= F_MAX]
    for key, color, label in (("rmse_measured", MEASURED, "measured"),
                              ("rmse_renormalized", RENORM, "renormalized (no fitting)"),
                              ("rmse_identified", IDENT, "identified (fitted)")):
        ok = [r for r in rs if r.get(key) is not None]
        ax.plot([r["f"] for r in ok], [r[key] for r in ok], "o-", ms=4, lw=1.5, color=color,
                label=label)
    lost = [r for r in rs if r.get("rmse_measured") is None]
    if lost:
        f0 = max(r["f"] for r in lost)
        ax.plot([f0], [0.3], "x", ms=9, mew=2, color=MEASURED)
        ax.annotate("diverges", (f0, 0.3), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=MEASURED)
    ax.set_yscale("log")
    ax.set_xlim(0, F_MAX * 1.02)
    ax.set_xlabel("simulation frequency $f$ (Hz)")
    ax.set_ylabel("trajectory RMSE (rad)")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    style(ax)

    for ax, letter in zip(axes, "ab"):
        ax.annotate(f"({letter})", (0, 1), xycoords="axes fraction", xytext=(-48, 14),
                    textcoords="offset points", fontsize=12, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_behaviour_isaac_{exp}.{ext}", dpi=200)
    print("wrote", outdir / f"fig_behaviour_isaac_{exp}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="single", choices=("single", "double"))
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})
    outdir, res = load(a.exp)
    flow_figure(a.exp, res, outdir)
    behaviour_figure(a.exp, res, outdir)
