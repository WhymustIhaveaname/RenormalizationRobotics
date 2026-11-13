"""Effective parameters on a nearly uniform grid of physics time steps, for each scheme.

For each target h the decimation is the integer nearest to 1 / (50 Hz * h) and the control
frequency the integer (Hz) nearest to 1 / (decimation * h), so both stay integers while h
tracks the grid.
With --delay, a command delay is identified as well (results/flow_mujoco_delay.json).
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import (ARMS, DRIVES, F_FLOW, PARAM0, T_FIT, Sim, first_order_prediction,
                    first_order_prediction_delay, identify, ref_sequence, start_points)

H_TARGET = np.arange(1, 21) * 0.25e-3  # 0.25 to 5 ms, the same grid as experiment 1
# Nominal command rates.  The upper end is pinned by the grid, since the decimation has to stay
# at least one: f_ctrl <= 1 / h_max = 200 Hz.  The lower end is pinned by the window, which only
# holds five command steps at 50 Hz, so the slower rates excite the parameters poorly.
F_CTRL = (50, 100, F_FLOW)
WITH_DELAY = "--delay" in sys.argv
OUT = Path(__file__).parent / "results" / (
    "flow_mujoco_delay.json" if WITH_DELAY else "flow_mujoco.json")


def schedule(f_nominal, h_target):
    f = 1.0 / h_target
    dec = max(1, round(f / f_nominal))
    return round(f / dec), dec


def chain(args):
    """Continuation from fine to coarse steps at fixed scheme and command rate."""
    arm, f_nominal = args
    sim = Sim(arm)
    predict = first_order_prediction_delay if WITH_DELAY else first_order_prediction
    rows, prev = [], None
    for h_target in sorted(H_TARGET):
        f_ctrl, dec = schedule(f_nominal, h_target)
        h = 1.0 / (f_ctrl * dec)
        # One parameter set for every drive, both joints in the residual.
        data = [ref_sequence(T_FIT, f_ctrl, f_drive=f) for f in DRIVES]
        data = [(r, np.asarray(sim.ground_truth(r, f_ctrl=f_ctrl))) for r in data]
        gt = np.concatenate([g for _, g in data])

        def rollout(a, kp, kd, delay=0.0, _data=data):
            out = [sim.rollout(r, dec, a, kp, kd, delay, f_ctrl=f_ctrl) for r, _ in _data]
            return None if any(t is None for t in out) else np.concatenate(out)

        rng = np.random.default_rng(1000 * ARMS.index(arm) + f_nominal + int(h * 1e6))
        starts = start_points(h, predict(arm, h), prev, rng, WITH_DELAY)
        t0 = time.perf_counter()
        fit = identify(rollout, gt, h, starts, with_delay=WITH_DELAY)
        prev = np.array(fit["params"])
        rows.append({"arm": arm, "h": h, "f": 1.0 / h, "f_ctrl_nominal": f_nominal,
                     "f_ctrl": f_ctrl, "dec": dec,
                     "params_identified": fit["params"], "rmse_identified": fit["rmse"],
                     "seconds": time.perf_counter() - t0})
        d = f" d/h={prev[3] / h:6.3f}" if WITH_DELAY else ""
        print(f"{arm} {f_ctrl:3d}x{dec:<3d} h={h * 1e3:6.3f} ms: armature/(h/2 K_D) "
              f"{fit['params'][0] / (h / 2 * 2.0):7.3f}{d}  rmse {fit['rmse']:.2e} "
              f"({rows[-1]['seconds']:.1f}s)", flush=True)
    return rows


if __name__ == "__main__":
    jobs = [(arm, fc) for arm in ARMS for fc in F_CTRL]
    with Pool(len(jobs)) as pool:
        rows = [r for rs in pool.map(chain, jobs) for r in rs]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {OUT}")
