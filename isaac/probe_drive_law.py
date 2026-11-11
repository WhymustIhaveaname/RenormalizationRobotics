"""What force does PhysX's joint drive actually apply?

The step-map probe showed that (v' - v) / h does not tend to the acceleration implied by
K_P (r - theta) - K_D theta_dot, so the assumed force law is wrong before any discretization
question arises.  This measures the law directly: extrapolate the one-step acceleration to
h -> 0 over a spread of states, then regress it on the three operators it should be built from.
"""
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

ROOT = Path("/home/youran/RenormalizationRobotics")
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

from pendulum_isaac import J0, KD0, KP0, IsaacPendulum  # noqa: E402
from common import G as GRAV, L, M  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "B"
C = M * GRAV * L / 2
R = 1.0
HS = np.array([1.0, 0.5, 0.25, 0.125, 0.0625]) * 1e-3
STATES = [(0.0, 0.0), (0.3, 0.0), (-0.5, 0.0), (0.0, 1.0), (0.0, -1.5), (0.4, 0.8), (-0.2, -0.6)]


def main():
    from isaacsim.core.api import World

    world = World(physics_dt=1e-3, stage_units_in_meters=1.0, backend="numpy")
    n = len(STATES)
    sim = IsaacPendulum(world, ARM, n)
    view = sim.view
    th0 = np.array([s[0] for s in STATES], dtype=np.float32)
    v0 = np.array([s[1] for s in STATES], dtype=np.float32)
    params = np.tile(np.array([0.0, KP0, KD0, 0.0], dtype=np.float32), (n, 1))

    acc, states = [], []
    for h in HS:
        sim.set_dt(h)
        world.reset()
        sim._apply(params)
        view.set_joint_positions(th0[:, None])
        view.set_joint_velocities(v0[:, None])
        if ARM == "A":
            view.set_joint_efforts((KP0 * (R - th0) - KD0 * v0).astype(np.float32)[:, None])
        else:
            view.set_joint_position_targets(np.full((n, 1), R, dtype=np.float32))
        # The first step after writing the state uses PhysX's cached velocity in the drive's
        # damping term, so take two steps and read the map off the second one.
        world.step(render=False)
        th1 = view.get_joint_positions().reshape(-1).copy()
        v1 = view.get_joint_velocities().reshape(-1).copy()
        world.step(render=False)
        acc.append((view.get_joint_velocities().reshape(n) - v1) / h)
        states.append((th1.copy(), v1.copy()))
    acc = np.array(acc)
    a0 = np.array([np.polyfit(HS, acc[:, i], 2)[-1] for i in range(n)])

    print(f"\n=== arm {ARM}: one-step acceleration extrapolated to h -> 0 ===", flush=True)
    print(f"{'(theta, omega)':>16} {'measured a':>12} {'expected a':>12} {'ratio':>8}")
    for i, (th, v) in enumerate(STATES):
        th, v = states[-1][0][i], states[-1][1][i]
        expect = (-C * np.sin(th) + KP0 * (R - th) - KD0 * v) / J0
        print(f"{f'({th:.3f}, {v:.3f})':>16} {a0[i]:12.2f} {expect:12.2f} "
              f"{a0[i] / expect if expect else float('nan'):8.4f}")

    # Regress J * a0 on the operators the drive is supposed to be made of.
    thr, vr = states[-1]
    A = np.stack([R - thr, -vr, -np.sin(thr)], 1)
    coef, *_ = np.linalg.lstsq(A, J0 * a0, rcond=None)
    print(f"\n  J * a  =  {coef[0]:.4f} (r - theta)  {coef[1]:+.4f} theta_dot  "
          f"{coef[2]:+.4f} sin(theta)")
    print(f"  expected  {KP0:.4f} (r - theta)  {KD0:+.4f} theta_dot  {C:+.4f} sin(theta)")
    print(f"  residual {np.max(np.abs(A @ coef - J0 * a0)):.4g} N m", flush=True)


main()
app.close()
