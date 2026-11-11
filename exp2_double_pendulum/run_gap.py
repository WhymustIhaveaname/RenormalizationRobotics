"""Error of measured, renormalized, and identified parameters across simulation frequencies.

Two numbers are recorded at each resolution: the joint-angle RMSE over the first second, and
the time at which the simulated free link leaves the real one by more than half a radian — the
horizon over which the simulator can still be trusted in a chaotic system.
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import (F_CTRL, PARAM0, T_FIT, Sim, identify, ref_sequence, renormalized, rmse)

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
    refs_fit, refs_long = ref_sequence(T_FIT), ref_sequence(T_HORIZON)
    gt_fit, gt_long = sim.ground_truth(refs_fit), sim.ground_truth(refs_long)
    t_long = np.arange(1, len(refs_long) + 1) / F_CTRL

    t0 = time.perf_counter()
    pred = renormalized(h)
    fit = identify(lambda a, kp, kd, delay=0.0: sim.rollout(refs_fit, dec, a, kp, kd, delay),
                   gt_fit, 1.0 / (F_CTRL * dec), [PARAM0, pred])
    row = {"dec": dec, "h": h, "f": F_CTRL * dec,
           "params_renormalized": pred.tolist(), "params_identified": fit["params"],
           "rmse_identified": fit["rmse"], "seconds": time.perf_counter() - t0}
    for name, p in (("measured", PARAM0), ("renormalized", pred),
                    ("identified", np.array(fit["params"]))):
        row[f"rmse_{name}"] = rmse(sim, refs_fit, dec, p, gt_fit) if name != "identified" \
            else fit["rmse"]
        row[f"peel_{name}"] = peel_time(sim.rollout(refs_long, dec, *p), gt_long, t_long)
    print(f"f={row['f']:5d} Hz: RMSE {row['rmse_measured']} / {row['rmse_renormalized']:.3e} / "
          f"{row['rmse_identified']:.3e}   peel {row['peel_measured']:.2f} / "
          f"{row['peel_renormalized']:.2f} / {row['peel_identified']:.2f} s "
          f"({row['seconds']:.1f}s)", flush=True)
    return row


if __name__ == "__main__":
    with Pool(len(DECIMATION)) as pool:
        rows = pool.map(run, DECIMATION)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(sorted(rows, key=lambda r: r["f"]), indent=1, default=float))
    print(f"wrote {OUT}")
