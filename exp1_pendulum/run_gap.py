"""Trajectory error of measured, analytically renormalized, and identified parameters.

At each physics time step h, the same pendulum is simulated with
  measured      (J, K_P, K_D) copied from the physical rod,
  renormalized  the first-order prediction for that scheme, with no fitting at all,
  identified    the best fit to the real trajectory,
and compared against the continuous-time ground truth.

One parameter set has to serve five drive frequencies at once, so identification cannot
specialize to a single command. The control frequency is fixed at 50 Hz and only the decimation
varies, so every run sees the same commands and the same real trajectories; h = 1 / (50 * dec).
A fine grid of small h is also run with measured parameters only, so that the step size at
which measured parameters reach the renormalized accuracy is measured rather than extrapolated.
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import (ARMS, DRIVES, PARAM0, Sim, first_order_prediction,
                    first_order_prediction_delay, ground_truth, identify, pooled_rmse,
                    ref_sequence)

F_CTRL = 50  # Hz, the legged_gym value; held fixed so the command never changes
# h = 1 / (50 * decimation): 20 ms down to 0.2 ms, i.e. h K_D / J from 2.85 down to 0.03.
DECIMATION = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 80, 100)
DECIMATION_FINE = (125, 160, 200, 250, 320, 400, 500)  # measured parameters only, up to 25 kHz
OUT = Path(__file__).parent / "results" / "gap_mujoco.json"


def renormalized(arm, h):
    """First-order parameters for this scheme, or None where they are unphysical."""
    p = first_order_prediction("B", h) if arm == "B" else first_order_prediction_delay(arm, h)
    return None if p[0] <= 0 else p  # arm C drives the effective inertia negative at h > 2J/K_D


def datasets(dec):
    """One (command, ground truth) pair per drive frequency, all at the same resolution."""
    out = []
    for f_ref in DRIVES:
        refs = ref_sequence(F_CTRL, f_ref)
        out.append((refs, ground_truth(refs, F_CTRL, dec)))
    return out


def run_arm(arm):
    sim = Sim(arm)
    with_delay = arm != "B"
    rows, prev = [], None
    for dec in sorted(DECIMATION, reverse=True):  # continuation from fine to coarse steps
        h = 1.0 / (F_CTRL * dec)
        data = datasets(dec)
        t0 = time.perf_counter()
        ren = renormalized(arm, h)
        starts = [np.append(PARAM0, 0.0) if with_delay else PARAM0]
        if ren is not None:
            starts.append(ren)
        if prev is not None:
            starts.append(prev)
        fit = identify(sim, data, F_CTRL, dec, starts, with_delay=with_delay)
        prev = np.array(fit["params"])
        rows.append({
            "arm": arm, "h": h, "f": 1.0 / h, "f_ctrl": F_CTRL, "dec": dec,
            "rmse_measured": pooled_rmse(sim, data, F_CTRL, dec, PARAM0),
            "rmse_renormalized": pooled_rmse(sim, data, F_CTRL, dec, ren) if ren else None,
            "rmse_identified": fit["rmse"],
            "params_renormalized": list(ren) if ren else None,
            "params_identified": fit["params"],
            "seconds": time.perf_counter() - t0,
        })
        r = rows[-1]
        print(f"{arm} dec={dec:<4d} h={h * 1e3:6.3f}ms: measured {r['rmse_measured']} "
              f"renorm {r['rmse_renormalized']} identified {r['rmse_identified']:.2e} "
              f"({r['seconds']:.1f}s)", flush=True)

    fine = []
    for dec in DECIMATION_FINE:
        h = 1.0 / (F_CTRL * dec)
        fine.append({"arm": arm, "h": h, "f": 1.0 / h, "f_ctrl": F_CTRL, "dec": dec,
                     "rmse_measured": pooled_rmse(sim, datasets(dec), F_CTRL, dec, PARAM0)})
        print(f"{arm} fine dec={dec}: measured {fine[-1]['rmse_measured']:.2e}", flush=True)
    return rows, fine


if __name__ == "__main__":
    with Pool(len(ARMS)) as pool:
        out = pool.map(run_arm, ARMS)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"main": [r for rows, _ in out for r in rows],
                               "fine": [r for _, fine in out for r in fine]}, indent=1, default=float))
    print(f"wrote {OUT}")
