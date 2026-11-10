"""The two figure layouts, shared by every experiment and every engine.

Both MuJoCo and Isaac call these, so the figures cannot drift apart: a caller supplies the
numbers, this module decides what the figure looks like.

flow_figure:       one column per scheme, one row per parameter, last row the post-fit error
                   against h^2.
behaviour_figure:  two panels, the trajectory error against time and the RMSE of each
                   parameter set against simulation frequency.
"""

import matplotlib.pyplot as plt
import numpy as np

FIT_COLOR = "#eb6834"
MEASURED, RENORM, IDENT = "#eb6834", "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SINGLE_POINT = "#1baf7a"


def use_house_style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def poly_fit(x, y, deg):
    """Least-squares polynomial; returns coefficients from highest degree down, and R^2."""
    coef = np.polyfit(x, y, deg)
    r2 = 1 - np.sum((y - np.polyval(coef, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef, r2


def poly_fit_origin(x, y, deg):
    """Least-squares polynomial with no constant term, since no shift survives at h = 0.
    Returns coefficients from highest degree down (the constant is 0), and R^2."""
    coef = np.linalg.lstsq(np.stack([x ** k for k in range(deg, 0, -1)], 1), y, rcond=None)[0]
    coef = np.append(coef, 0.0)
    r2 = 1 - np.sum((y - np.polyval(coef, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef, r2


def num(x):
    """Legend number: two decimals for ordinary values, three significant digits for small ones,
    so that relative shifts (~0.07) stay as readable as absolute ones (~1.04)."""
    return "0" if x == 0 else f"{x:.2f}" if abs(x) >= 0.1 else f"{x:.3g}"


def num1(x):
    """One significant digit, for a coefficient the theory says should vanish: it only has to be
    seen to be small, and the short form leaves room for the coefficient that does matter."""
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def flow_figure(columns, row_labels, out_paths, h_max, caption, f_ticks,
                point_colors=None, legend_entries=None, rmse_label="post-fit RMSE (rad)",
                quadratic=False, show_rmse=True):
    """Parameter flow against the physics time step.

    `columns` is one dict per scheme:
        title   two-line column heading
        h       (n,) time steps in ms
        rows    list, one per parameter row, of either None (not fitted in this scheme) or
                (values in display units, theory), where theory is the slope per ms or a pair
                (slope per ms, curvature per ms^2) when a second-order prediction is known
        rmse    (n,) post-fit RMSE
        groups  optional (n,) keys into `point_colors` for colouring the points
    `row_labels` is the y label of each parameter row; with `show_rmse` a post-fit error row is
    appended below them.
    With `quadratic`, rows that do have a predicted slope are fitted as c_1 h + c_2 h^2 instead
    of a straight line, so that c_1 can be read against the first-order theory on its own.
    """
    n_rows = len(row_labels) + bool(show_rmse)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.45 * n_rows))
    h_line = np.array([0, h_max])
    h_dense = np.linspace(0, h_max, 200)

    def draw_points(ax, x, y, groups):
        if point_colors is None or groups is None:
            ax.plot(x, y, "o", ms=6, color=SINGLE_POINT, mec="white", mew=0.7, zorder=3)
        else:
            for key, color in point_colors.items():
                m = np.asarray(groups) == key
                ax.plot(np.asarray(x)[m], np.asarray(y)[m], "o", ms=6, color=color, mec="white",
                        mew=0.7, zorder=3)

    for j, col in enumerate(columns):
        h = np.asarray(col["h"])
        groups = col.get("groups")
        axes[0, j].set_title(col["title"], fontsize=11, pad=14)

        for i, label in enumerate(row_labels):
            ax = axes[i, j]
            style(ax)
            ax.set_xlim(0, h_max * 1.04)
            ax.set_xlabel("$h = 1/f$ (ms)", fontsize=9)
            if j == 0:
                ax.set_ylabel(label)
            entry = col["rows"][i]
            if entry is None:
                ax.text(0.5, 0.5, "not fitted:\nno command advance in this scheme", ha="center",
                        va="center", transform=ax.transAxes, color=INK2, fontsize=9)
                ax.set_yticks([])
                continue
            y = np.asarray(entry[0])
            t1, t2 = (entry[1], 0.0) if np.isscalar(entry[1]) else entry[1]
            ax.axhline(0, color=GRID, lw=1, zorder=0)
            ax.plot(h_dense if t2 else h_line, np.polyval([t2, t1, 0], h_dense if t2 else h_line),
                    color=INK, lw=1.4, zorder=1,
                    label=(f"theory: $c_1$ = {num(t1)}, $c_2$ = {num(t2)}" if t2
                           else f"theory: slope {num(t1)}"))
            theory = t1
            if t1 == 0 and t2 == 0:  # nothing predicted: a free quadratic exposes the h^2 term
                coef, r2 = poly_fit(h, y, 2)
                text = (f"fit: $c_0 + c_1 h + c_2 h^2$, $R^2$ = {r2:.4f}\n"
                        f"$c_0$ = {num1(coef[2])}, $c_1$ = {num1(coef[1])}, "
                        f"$c_2$ = {num(coef[0])}")
            elif quadratic:  # separate the predicted slope from the leading correction to it
                coef, r2 = poly_fit_origin(h, y, 2)
                text = (f"fit: $c_1 h + c_2 h^2$, $R^2$ = {r2:.4f}\n"
                        f"$c_1$ = {num(coef[1])}, $c_2$ = {num(coef[0])}")
            else:
                coef, r2 = poly_fit(h, y, 1)
                text = f"fit: slope {num(coef[0])}, $R^2$ = {r2:.4f}"
            ax.plot(h_dense, np.polyval(coef, h_dense), color=FIT_COLOR, lw=2, ls="--", zorder=2,
                    label=text)
            draw_points(ax, h, y, groups)
            ax.legend(loc="best", frameon=False, fontsize=9, handlelength=2.2)

        top = axes[0, j].secondary_xaxis("top")
        top.set_xticks([1e3 / f for f, _ in f_ticks])
        top.set_xticklabels([t for _, t in f_ticks], fontsize=8, color=INK2)
        top.set_xlabel("$f$ (Hz)", fontsize=8, color=INK2, labelpad=2)

        if not show_rmse:
            continue
        # Post-fit error: all first-order effects absorbed means RMSE should scale as h^2.
        ax = axes[-1, j]
        draw_points(ax, h**2, np.asarray(col["rmse"]), groups)
        ax.text(0.03, 0.95, r"theory: RMSE $\propto h^2$", transform=ax.transAxes, va="top",
                fontsize=9, color=INK)
        ax.set_xlim(0, h_max**2 * 1.04)
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        ax.set_xlabel("$h^2$ (ms$^2$)", fontsize=9)
        style(ax)
        if j == 0:
            ax.set_ylabel(rmse_label)

    rect_top = 1.0
    if legend_entries:
        dots = [plt.Line2D([], [], ls="none", marker="o", ms=6, color=c, mec="white")
                for _, c in legend_entries]
        fig.legend(dots, [t for t, _ in legend_entries], loc="upper center", ncol=len(dots),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
        rect_top = 0.975
    fig.text(0.5, 0.004, caption, ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.018, 1, rect_top))
    for p in out_paths:
        fig.savefig(p, dpi=200)
    plt.close(fig)


def behaviour_figure(error_panel, gap_panel, out_paths, f_max):
    """Two panels: trajectory error against time, and RMSE of each parameter set against f.

    `error_panel`: dict with t, series [(values, color, label)], ylabel, title
    `gap_panel`:   dict with series [(f array, rmse array, color, label)], ylabel,
                   optional diverge_f and annotation (text, f, lo, hi)
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    for values, color, label in error_panel["series"]:
        ax.plot(error_panel["t"], values, color=color, lw=1.4, label=label)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xlim(0, error_panel.get("xlim", error_panel["t"][-1]))
    ax.set_xlabel("time (s)")
    ax.set_ylabel(error_panel["ylabel"])
    ax.set_title(error_panel["title"], fontsize=11.5)
    ax.legend(loc=error_panel.get("legend_loc", "lower left"), frameon=False, fontsize=9.5)
    style(ax)

    ax = axes[1]
    for f, rmse, color, label in gap_panel["series"]:
        ax.plot(f, rmse, "o-", ms=4, lw=1.5, color=color, label=label)
    if gap_panel.get("diverge_f"):
        f0 = gap_panel["diverge_f"]
        ax.plot([f0], [0.1], "x", ms=9, mew=2, color=MEASURED)
        ax.annotate("diverges", (f0, 0.1), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=MEASURED)
    ann = gap_panel.get("annotation")
    if ann:
        text, f0, lo, hi = ann
        ax.annotate("", (f0, lo), (f0, hi), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
        ax.annotate(text, (f0, np.sqrt(lo * hi)), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_yscale("log")
    if gap_panel.get("ylim"):
        ax.set_ylim(*gap_panel["ylim"])
    ax.set_xlim(0, f_max * 1.02)
    ax.set_xlabel("simulation frequency $f$ (Hz)")
    ax.set_ylabel(gap_panel["ylabel"])
    ax.legend(loc=gap_panel.get("legend_loc", "lower left"), frameon=False, fontsize=9.5)
    style(ax)

    for ax, letter in zip(axes, "ab"):
        ax.annotate(f"({letter})", (0, 1), xycoords="axes fraction", xytext=(-48, 14),
                    textcoords="offset points", fontsize=12, color=INK)
    fig.tight_layout()
    for p in out_paths:
        fig.savefig(p, dpi=200)
    plt.close(fig)
