"""Measure PhysX's one-step map, so each scheme gets its own counterterms.

Same idea as exp1_pendulum/probe_step_map.py, which validates the method against MuJoCo: put
the articulation in a known state, take a single physics step, and extrapolate h -> 0 to read
the Taylor coefficients of the step map.  PhysX is closed source, so this is the only way to
learn which map TGS and PGS actually realize.

Only the explicitly held torque can be probed this way: PhysX's own drive reads a velocity
that set_joint_velocities does not reach, so its force law is measured by probe_drive_law.py.

Run (inside the Isaac venv):  python probe_step_map_isaac.py A
"""
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repo, wherever it is checked out
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

from pendulum_isaac import KD0, KP0, IsaacPendulum  # noqa: E402
from probe_step_map import HS, R, STATES, accel, report  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "A"
ITERS = [int(a) for a in sys.argv[2:]] or [32]    # TGS position iterations to test


def main():
    from isaacsim.core.api import World

    world = World(physics_dt=1e-3, stage_units_in_meters=1.0, backend="numpy")
    n = len(STATES)
    sim = IsaacPendulum(world, ARM, n)
    view = sim.view
    th0 = np.array([s[0] for s in STATES], dtype=np.float32)
    v0 = np.array([s[1] for s in STATES], dtype=np.float32)
    params = np.tile(np.array([0.0, KP0, KD0, 0.0], dtype=np.float32), (n, 1))

    def one_step(h):
        sim.set_dt(h)
        world.reset()
        sim._apply(params)
        view.set_joint_positions(th0[:, None])
        view.set_joint_velocities(v0[:, None])
        if ARM == "A":
            view.set_joint_efforts((KP0 * (R - th0) - KD0 * v0).astype(np.float32)[:, None])
        else:
            view.set_joint_position_targets(np.full((n, 1), R, dtype=np.float32))
        world.step(render=False)
        return (view.get_joint_positions().reshape(n),
                view.get_joint_velocities().reshape(n))

    for iters in ITERS:
        view.set_solver_position_iteration_counts(np.full(n, iters, dtype=np.int32))
        f = np.array([accel(t, v) for t, v in STATES])
        c_th, c_v = [], []
        for h in HS:
            q1, u1 = one_step(h)
            c_th.append((q1 - th0 - h * v0) / h**2)
            c_v.append((u1 - v0 - h * f) / h**2)
        c_th, c_v = np.array(c_th), np.array(c_v)
        # Diagnostic: the h^1 coefficient must converge to the exact acceleration, otherwise
        # the step map is not of the assumed form and the h^2 fit below is meaningless.
        print(f"  raw residuals for state {STATES[0]}, f = {f[0]:.1f}:", flush=True)
        for k, h in enumerate(HS):
            q1, u1 = one_step(h)
            print(f"    h={h*1e3:6.3f} ms  (v'-v)/h = {(u1[0]-v0[0])/h:12.2f}  "
                  f"(th'-th-hv)/h^2 = {c_th[k, 0]:12.2f}  (v'-v-hf)/h^2 = {c_v[k, 0]:14.2f}",
                  flush=True)
        rows = []
        for i, (th, v) in enumerate(STATES):
            m2th = np.polyfit(HS, c_th[:, i], 2)[-1]
            m2v = np.polyfit(HS, c_v[:, i], 2)[-1]
            rows.append((f"({th}, {v})", m2th, m2v, f[i], th, v))
        report(f"arm {ARM}, {iters} position iterations", rows)
        print(flush=True)


main()
app.close()
