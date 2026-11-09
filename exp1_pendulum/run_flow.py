"""Identify (J, K_P, K_D) for every arm on a nearly uniform grid of physics time steps h.

For each target h and nominal control frequency, the integer decimation closest to
1 / (f_ctrl_nominal * h) is used with the integer f_ctrl (Hz) closest to 1 / (decimation * h).
With --delay, also identify a command delay d (results/flow_mujoco_delay.json).
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import (ARMS, J0, KD0, KP0, PARAM0, Sim, first_order_prediction,
                    first_order_prediction_delay, ground_truth, identify, ref_sequence)

F_CTRL = (10, 20, 50, 100, 200)  # nominal control frequencies (Hz)
H_TARGET = np.arange(1, 21) * 0.25e-3  # target physics time step h = 0.25, 0.5, ..., 5 ms
N_RANDOM_STARTS = 2


def schedule(f_nominal, h_target):
    """Integer (f_ctrl, decimation) whose physics step 1 / (f_ctrl * decimation) is closest to h_target."""
    f = 1.0 / h_target
    dec = max(1, round(f / f_nominal))
    return round(f / dec), dec
WITH_DELAY = "--delay" in sys.argv
OUT = Path(__file__).parent / "results" / ("flow_mujoco_delay.json" if WITH_DELAY else "flow_mujoco.json")


def starts_for(arm, h, prev, rng):
    if not WITH_DELAY:
        starts = [PARAM0, first_order_prediction("B", h)]  # the paper's formula with delta t = h/2
        # Inflated inertia keeps explicit PD stable when h K_D / J is no longer small.
        starts += [np.array([J0 + c * h * KD0, KP0, KD0]) for c in (1, 2, 4)]
        starts += [PARAM0 * np.exp(rng.normal(0, 0.5, 3)) for _ in range(N_RANDOM_STARTS)]
    else:
        starts = [np.append(PARAM0, 0.0), first_order_prediction_delay(arm, h)]
        starts += [np.array([J0 + c * h * KD0, KP0, KD0, h / 2]) for c in (1, 2, 4)]
        starts += [np.append(PARAM0 * np.exp(rng.normal(0, 0.5, 3)), rng.uniform(0, h))
                   for _ in range(N_RANDOM_STARTS)]
    return starts + ([prev] if prev is not None else [])


def chain(args):
    """Continuation from small to large h at fixed arm and nominal control frequency."""
    arm, f_nominal = args
    rng = np.random.default_rng(1000 * ARMS.index(arm) + f_nominal)
    sim = Sim(arm)
    rows, prev = [], None
    for h_target in H_TARGET:
        f_ctrl, dec = schedule(f_nominal, h_target)
        h = 1.0 / (f_ctrl * dec)
        refs = ref_sequence(f_ctrl)
        gt = ground_truth(refs, f_ctrl, dec)
        t0 = time.perf_counter()
        fit = identify(sim, refs, f_ctrl, dec, gt, starts_for(arm, h, prev, rng), with_delay=WITH_DELAY)
        meas = sim.rollout(refs, f_ctrl, dec, *PARAM0)
        fit["rmse_measured"] = float(np.sqrt(np.mean((meas - gt) ** 2))) if meas is not None else None
        fit.update(arm=arm, f_ctrl_nominal=f_nominal, f_ctrl=f_ctrl, dec=dec, h=h, f=1.0 / h,
                   seconds=time.perf_counter() - t0)
        rows.append(fit)
        prev = np.array(fit["params"])
        p = prev[:3] / PARAM0
        d = f" d/h={prev[3] / h:.4f}" if WITH_DELAY else ""
        print(f"{arm} f_ctrl={f_ctrl:3d}x{dec:<3d} h={h * 1e3:.3f}ms: J*/J={p[0]:.4f} KP*/KP={p[1]:.4f} "
              f"KD*/KD={p[2]:.4f}{d} rmse={fit['rmse']:.2e} ({fit['seconds']:.1f}s)", flush=True)
    return rows


if __name__ == "__main__":
    jobs = [(arm, fc) for arm in ARMS for fc in F_CTRL]
    with Pool(len(jobs)) as pool:
        rows = [r for rs in pool.map(chain, jobs) for r in rs]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {len(rows)} fits to {OUT}")
