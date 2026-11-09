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

from common import (ARMS, PARAM0, T_FIT, Sim, first_order_prediction,
                    first_order_prediction_delay, identify, ref_sequence)

H_TARGET = np.arange(1, 21) * 1e-3  # 1 to 20 ms
F_NOMINAL = 50  # Hz
WITH_DELAY = "--delay" in sys.argv
OUT = Path(__file__).parent / "results" / (
    "flow_mujoco_delay.json" if WITH_DELAY else "flow_mujoco.json")


def schedule(h_target):
    f = 1.0 / h_target
    dec = max(1, round(f / F_NOMINAL))
    return round(f / dec), dec


def chain(arm):
    """Continuation from fine to coarse steps at fixed scheme."""
    sim = Sim(arm)
    predict = first_order_prediction_delay if WITH_DELAY else first_order_prediction
    rows, prev = [], None
    for h_target in sorted(H_TARGET):
        f_ctrl, dec = schedule(h_target)
        h = 1.0 / (f_ctrl * dec)
        refs = ref_sequence(T_FIT, f_ctrl)
        gt = sim.ground_truth(refs, f_ctrl=f_ctrl)
        starts = [np.append(PARAM0, 0.0) if WITH_DELAY else PARAM0, predict(arm, h)]
        if prev is not None:
            starts.append(prev)
        t0 = time.perf_counter()
        fit = identify(sim, refs, dec, gt, starts, f_ctrl=f_ctrl, with_delay=WITH_DELAY)
        prev = np.array(fit["params"])
        rows.append({"arm": arm, "h": h, "f": 1.0 / h, "f_ctrl": f_ctrl, "dec": dec,
                     "params_identified": fit["params"], "rmse_identified": fit["rmse"],
                     "seconds": time.perf_counter() - t0})
        d = f" d/h={prev[3] / h:6.3f}" if WITH_DELAY else ""
        print(f"{arm} h={h * 1e3:6.3f} ms ({f_ctrl}x{dec}): armature/(h/2 K_D) "
              f"{fit['params'][0] / (h / 2 * 2.0):7.3f}{d}  rmse {fit['rmse']:.2e} "
              f"({rows[-1]['seconds']:.1f}s)", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(len(ARMS)) as pool:
        rows = [r for rs in pool.map(chain, ARMS) for r in rs]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {OUT}")
