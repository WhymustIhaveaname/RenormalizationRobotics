"""Error of measured, renormalized, and identified parameters across simulation frequencies.

Two numbers are recorded at each resolution: the joint-angle RMSE over the first T_GAP seconds,
and the time at which the simulated free link leaves the real one by more than half a radian —
the horizon over which the simulator can still be trusted in a chaotic system.

The identification is run_flow's: T_FIT seconds, five drive frequencies at once, both joints in
the residual, K_P held at its measured value.  It is then scored on a single drive over ten
times that window, so the identified curve is an out-of-sample error like the other two.
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import (DRIVES, F_CTRL, PARAM0, T_FIT, T_GAP, Sim, identify, ref_sequence,
                    renormalized, rmse, start_points)

DECIMATION = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40)
T_HORIZON = 3.0  # s over which the peel-off time is measured
PEEL_THRESHOLD = np.pi  # rad on the free joint: half a turn away from the real motion
OUT = Path(__file__).parent / "results" / "gap_mujoco.json"


def peel_time(theta, gt, t):
    """When the free joint first leaves the real trajectory by PEEL_THRESHOLD."""
    if theta is None:
        return 0.0
    late = np.flatnonzero(np.abs(theta[:, 1] - gt[:, 1]) > PEEL_THRESHOLD)
    return float(t[late[0]]) if len(late) else float("inf")


def run(dec):
    sim = Sim()
    h = 1.0 / (F_CTRL * dec)
    # The identification window: the same five drives run_flow fits, both joints in the residual.
    fit_data = [(r, np.asarray(sim.ground_truth(r)))
                for r in (ref_sequence(T_FIT, f_drive=f) for f in DRIVES)]
    gt_fit = np.concatenate([g for _, g in fit_data])
    # The scoring window: one drive, ten times longer, never seen by the fit.
    refs_gap, refs_long = ref_sequence(T_GAP), ref_sequence(T_HORIZON)
    gt_gap, gt_long = sim.ground_truth(refs_gap), sim.ground_truth(refs_long)
    t_long = np.arange(1, len(refs_long) + 1) / F_CTRL

    t0 = time.perf_counter()
    pred = renormalized(h)

    def rollout(a, kp, kd, delay=0.0):
        out = [sim.rollout(r, dec, a, kp, kd, delay) for r, _ in fit_data]
        return None if any(t is None for t in out) else np.concatenate(out)

    rng = np.random.default_rng(dec)
    fit = identify(rollout, gt_fit, h, start_points(h, pred, None, rng, False))
    row = {"dec": dec, "h": h, "f": F_CTRL * dec,
           "params_renormalized": pred.tolist(), "params_identified": fit["params"],
           "rmse_fit": fit["rmse"], "seconds": time.perf_counter() - t0}
    for name, p in (("measured", PARAM0), ("renormalized", pred),
                    ("identified", np.array(fit["params"]))):
        row[f"rmse_{name}"] = rmse(sim, refs_gap, dec, p, gt_gap)
        row[f"peel_{name}"] = peel_time(sim.rollout(refs_long, dec, *p), gt_long, t_long)
    e = lambda v: "diverged" if v is None else f"{v:.3e}"
    print(f"f={row['f']:5d} Hz: RMSE {e(row['rmse_measured'])} / {e(row['rmse_renormalized'])} / "
          f"{e(row['rmse_identified'])}   peel {row['peel_measured']:.2f} / "
          f"{row['peel_renormalized']:.2f} / {row['peel_identified']:.2f} s "
          f"({row['seconds']:.1f}s)", flush=True)
    return row


if __name__ == "__main__":
    with Pool(len(DECIMATION)) as pool:
        rows = pool.map(run, DECIMATION)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(sorted(rows, key=lambda r: r["f"]), indent=1, default=float))
    print(f"wrote {OUT}")
