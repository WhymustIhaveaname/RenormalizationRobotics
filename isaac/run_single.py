"""Isaac Sim version of experiment 1: effective parameters of a PD-controlled pendulum.

Usage (inside the Isaac venv):  python run_single.py --arm A
Writes results/isaac_single_<arm>.json next to the MuJoCo results.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="A", choices=("A", "B", "C"))
parser.add_argument("--t-fit", type=float, default=1.0, help="seconds per drive frequency")
parser.add_argument("--delay", action="store_true", help="also identify a command delay")
parser.add_argument("--points", type=int, default=10)
parser.add_argument("--h-min", type=float, default=1.0, help="ms")
parser.add_argument("--h-max", type=float, default=19.0, help="ms")
parser.add_argument("--single-drive", action="store_true",
                    help="one 2 Hz command, as MuJoCo's flow sweep uses")
args = parser.parse_args()
app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

ROOT = Path("/home/youran/RenormalizationRobotics")
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

from fit import fit  # noqa: E402
from pendulum_isaac import F_CTRL, J0, KD0, KP0, IsaacPendulum  # noqa: E402
import common as mj  # noqa: E402  (the MuJoCo module supplies the shared reference solution)

H_TARGET = np.linspace(args.h_min * 1e-3, args.h_max * 1e-3, args.points)
OUT = ROOT / "exp1_pendulum" / "results" / (
    f"isaac_single_smallh_{args.arm}.json" if (args.delay and args.h_max <= 6)
    else f"isaac_single_delay_{args.arm}.json" if args.delay else f"isaac_single_{args.arm}.json")


def schedule(h_target):
    f = 1.0 / h_target
    dec = max(1, round(f / F_CTRL))
    return round(f / dec), dec


def main():
    from isaacsim.core.api import World

    world = World(physics_dt=1 / 200, stage_units_in_meters=1.0, backend="numpy")
    ndim = 4 if args.delay else 3
    n_envs = 1 + 2 * ndim
    sim = IsaacPendulum(world, args.arm, n_envs)

    rows, prev = [], None
    for h_target in sorted(H_TARGET):
        f_ctrl, dec = schedule(h_target)
        h = 1.0 / (f_ctrl * dec)
        sim.set_dt(h)
        drives = (mj.F_REF,) if args.single_drive else mj.DRIVES
        data = [(mj.ref_sequence(f_ctrl, f_ref, t_end=args.t_fit), f_ref) for f_ref in drives]
        data = [(refs, mj.ground_truth(refs, f_ctrl, dec)[dec - 1::dec]) for refs, _ in data]
        gt = np.concatenate([g for _, g in data])

        def rollout(P):
            return np.concatenate([sim.rollout(refs, dec, P, f_ctrl=f_ctrl) for refs, _ in data],
                                  axis=1)

        if args.delay:
            measured = np.array([0.0, KP0, KD0, 0.0])
            predicted = np.array([h / 2 * KD0, KP0, KD0, h / 2])
            scales = np.array([h / 2 * KD0, KP0, KD0, h])
            lo = np.array([-0.9 * J0 / scales[0], -2.0, -2.0, -5.0])
            hi = np.array([20.0, 2.0, 2.0, 5.0])
        else:
            measured = np.array([0.0, KP0, KD0])
            predicted = np.array([h / 2 * KD0, KP0, KD0 + h / 2 * KP0])
            scales = np.array([h / 2 * KD0, KP0, KD0])
            lo = np.array([-0.9 * J0 / scales[0], -2.0, -2.0])
            hi = np.array([20.0, 2.0, 2.0])
        starts = [np.zeros(len(scales)), (predicted - measured) / scales]
        if prev is not None:
            starts.append((prev - measured) / scales)
        # The landscape has poor local optima; restart around the prediction as well.
        rng = np.random.default_rng(int(h * 1e6))
        base = (predicted - measured) / scales
        starts += [base * rng.uniform(0.3, 2.0, len(scales)) for _ in range(2)]
        t0 = time.perf_counter()
        out = fit(rollout, gt, starts, scales, measured, bounds=(lo, hi))
        prev = np.array(out["params"])

        meas_traj = rollout(measured[None, :])[0]
        pred_traj = rollout(predicted[None, :])[0]
        rmse = lambda t: float(np.sqrt(np.mean((t - gt) ** 2))) if np.all(np.isfinite(t)) else None
        rows.append({"arm": args.arm, "h": h, "f": 1.0 / h, "f_ctrl": f_ctrl, "dec": dec,
                     "params_identified": out["params"], "rmse_identified": out["rmse"],
                     "params_renormalized": predicted.tolist(),
                     "rmse_measured": rmse(meas_traj), "rmse_renormalized": rmse(pred_traj),
                     "seconds": time.perf_counter() - t0})
        r = rows[-1]
        print(f"{args.arm} h={h * 1e3:6.3f} ms ({f_ctrl}x{dec}): armature/(h/2 K_D) "
              f"{r['params_identified'][0] / (h / 2 * KD0):7.3f}  rmse meas {r['rmse_measured']} "
              f"renorm {r['rmse_renormalized']} fit {r['rmse_identified']:.3e} "
              f"({r['seconds']:.0f}s)", flush=True)
        OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {OUT}")


main()
app.close()
