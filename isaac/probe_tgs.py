"""What does TGS actually do to the joint drive?

The PhysX drive is implicit, so it should renormalize like MuJoCo's implicitfast: dJ = -h KD/2
and d* = h/2.  PGS does exactly that; TGS gives almost nothing.  TGS advances time inside its
position iterations, so the natural guess is that the drive sees h / nPosIters rather than h.
This probe reports the iteration counts in use and then sweeps them.
"""
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

ROOT = Path("/home/youran/RenormalizationRobotics")
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

from pendulum_isaac import F_CTRL, J0, KD0, KP0, IsaacPendulum  # noqa: E402
import common as mj  # noqa: E402

# One arm per process: a second build_scene would collide with the first arm's USD prims.
ARMS = (sys.argv[1],) if len(sys.argv) > 1 else ("A",)
NS = (1, 4, 32)                # TGS position iterations; 32 is the Isaac default
H_MS = (0.25, 0.9, 1.6, 2.25, 3.0, 3.65, 4.35, 5.0)


def schedule(h):
    dec = max(1, round(1.0 / h / F_CTRL))
    return round(1.0 / h / dec), dec


def origin_quadratic(x, y):
    """Least-squares (c1, c2) in y = c1 x + c2 x^2, the same fit the figure draws."""
    A = np.stack([x, x**2], 1)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def main():
    from isaacsim.core.api import World

    world = World(physics_dt=5e-3, stage_units_in_meters=1.0, backend="numpy")
    for arm in ARMS:
        sweep(world, arm)


def sweep(world, arm):
    sim = IsaacPendulum(world, arm, 8)   # one env per central-difference probe
    print(f"\n=== arm {arm}, default position iterations "
          f"{sim.view.get_solver_position_iteration_counts()[0]} ===", flush=True)
    print(f"{'posIters':>9} {'armature/J c1':>14} {'c1 x N':>9} {'c2':>11} {'c2 x N^2':>11} "
          f"{'d* c1':>9} {'c1 x N':>9}", flush=True)
    for n in NS:
        sim.view.set_solver_position_iteration_counts(np.full(sim.n, n, dtype=np.int32))
        hs, arm_, del_ = [], [], []
        for h_ms in H_MS:
            fc, dec = schedule(h_ms * 1e-3)
            h = 1.0 / (fc * dec)
            sim.set_dt(h)
            refs = mj.ref_sequence(fc, mj.F_REF, t_end=1.0)
            gt = mj.ground_truth(refs, fc, dec)[dec - 1::dec]
            m = 1 if arm == "A" else n          # the drive, not a held torque, sees h / N
            sgn = +1 if arm == "A" else -1
            pred = np.array([J0 + sgn * h / 2 * KD0 / m, KP0, KD0, h / 2 / m])

            def run(rows, _fc=fc, _dec=dec, _refs=refs):
                P = np.atleast_2d(np.asarray(rows, float))
                P = np.column_stack([P[:, 0] - J0, P[:, 1], P[:, 2], P[:, 3]])
                return sim.rollout(_refs, _dec, P, f_ctrl=_fc)

            out = mj.identify(lambda J, kp, kd, d=0.0: run([(J, kp, kd, d)])[0], gt, h,
                              mj.start_points(h, pred, None, np.random.default_rng(n), True),
                              with_delay=True,
                              batch=lambda rows: [(t - gt) / np.sqrt(gt.size) for t in run(rows)])
            p_ = np.array(out["params"])
            hs.append(h * 1e3)
            arm_.append((p_[0] - J0) / J0)
            del_.append(p_[3] * 1e3)
        hs = np.array(hs)
        a1, a2 = origin_quadratic(hs, np.array(arm_))
        d1, _ = origin_quadratic(hs, np.array(del_))
        print(f"{n:>9} {a1:14.5f} {a1 * n:9.4f} {a2:11.3e} {a2 * n**2:11.3e} "
              f"{d1:9.5f} {d1 * n:9.4f}", flush=True)


main()
app.close()
