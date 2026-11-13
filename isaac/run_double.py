"""Isaac Sim version of experiment 2: effective parameters of the chaotic double pendulum.

Usage (inside the Isaac venv):  python run_double.py --arm A
The window is 0.3 s: beyond that PhysX decorrelates from the reference and the trajectory
error saturates for every parameter set.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="A", choices=("A", "B", "C"))
parser.add_argument("--t-fit", type=float, default=0.3)
args = parser.parse_args()
app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repo, wherever it is checked out
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "exp2_double_pendulum"))

from double_isaac import F_CTRL, KD0, KP0, IsaacDouble  # noqa: E402
import common as mj  # noqa: E402

H_TARGET = np.arange(1, 21, 2) * 1e-3  # 1, 3, ..., 19 ms
M11_MIN = 0.05169  # kg m^2, smallest joint-space inertia of the driven joint
OUT = ROOT / "exp2_double_pendulum" / "results" / f"isaac_double_{args.arm}.json"


def schedule(h_target):
    f = 1.0 / h_target
    dec = max(1, round(f / F_CTRL))
    return round(f / dec), dec


def main():
    from isaacsim.core.api import World

    world = World(physics_dt=1 / 200, stage_units_in_meters=1.0, backend="numpy")
    sim = IsaacDouble(world, args.arm, 1 + 2 * 3)
    ref_sim = mj.Sim()

    rows, prev = [], None
    for h_target in sorted(H_TARGET):
        f_ctrl, dec = schedule(h_target)
        h = 1.0 / (f_ctrl * dec)
        sim.set_dt(h)
        refs = mj.ref_sequence(args.t_fit, f_ctrl)
        gt = ref_sim.ground_truth(refs, f_ctrl=f_ctrl)

        def run(rows, _fc=f_ctrl, _dec=dec, _refs=refs):
            return sim.rollout(_refs, _dec, np.atleast_2d(np.asarray(rows, float)), f_ctrl=_fc)

        measured = np.array([0.0, KP0, KD0])
        predicted = np.array([h / 2 * KD0, KP0, KD0 + h / 2 * KP0])
        starts = mj.start_points(h, predicted, prev, np.random.default_rng(int(h * 1e6)), False)
        t0 = time.perf_counter()
        out = mj.identify(lambda a, kp, kd: run([(a, kp, kd)])[0], gt, h, starts,
                          batch=lambda rows: list(run(rows)))
        prev = np.array(out["params"])

        rmse = lambda p: float(np.sqrt(np.mean((run([p])[0] - gt) ** 2)))
        rows.append({"arm": args.arm, "h": h, "f": 1.0 / h, "f_ctrl": f_ctrl, "dec": dec,
                     "params_identified": out["params"], "rmse_identified": out["rmse"],
                     "params_renormalized": predicted.tolist(),
                     "rmse_measured": rmse(measured), "rmse_renormalized": rmse(predicted),
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
