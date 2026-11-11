"""Measure a simulator's one-step map instead of guessing it.

Backward error analysis needs only the Taylor coefficients of the step map,

    theta' = theta + h v + h^2 M2th + h^3 M3th,    v' = v + h f + h^2 M2v + h^3 M3v,

where f is the exact acceleration at the start of the step.  Those four coefficients can be
read off a simulator directly: put it in a known state, take a single step, and extrapolate
h -> 0.  No knowledge of the integrator's internals is required, which is the only way to get
at a closed-source solver.  This script does it for MuJoCo; probe_step_map_isaac.py does the
same for PhysX, so both go through the same analysis.

Run:  python probe_step_map.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import ARM_LABEL, ARMS, G, J0, KD0, KP0, L, M, Sim  # noqa: E402

C = M * G * L / 2
STATES = [(0.3, 0.7), (-0.4, 1.3), (0.8, -0.5)]   # (theta, omega) to separate the terms
R = 1.0                                            # held command
HS = np.array([2.0, 1.0, 0.5, 0.25, 0.125]) * 1e-3


def accel(th, v):
    return (-C * np.sin(th) + KP0 * (R - th) - KD0 * v) / J0


def one_step(sim, h, th, v):
    """State after a single physics step from (th, v) with the command held at R."""
    m, d = sim.model, sim.data
    m.opt.timestep = h
    m.dof_armature[0] = 0.0
    if sim.arm == "C":
        m.actuator_gainprm[0, 0] = KP0
        m.actuator_biasprm[0, 1] = -KP0
        m.actuator_biasprm[0, 2] = -KD0
    import mujoco
    mujoco.mj_resetData(m, d)
    d.qpos[0], d.qvel[0] = th, v
    mujoco.mj_forward(m, d)
    d.ctrl[0] = R if sim.arm == "C" else KP0 * (R - th) - KD0 * v
    mujoco.mj_step(m, d)
    return float(d.qpos[0]), float(d.qvel[0])


def coefficients(sim, th, v):
    """(M2th, M2v) by Richardson extrapolation of the single-step residuals."""
    f = accel(th, v)
    c_th, c_v = [], []
    for h in HS:
        th1, v1 = one_step(sim, h, th, v)
        c_th.append((th1 - th - h * v) / h**2)
        c_v.append((v1 - v - h * f) / h**2)
    # each sequence is M2 + h M3 + O(h^2); a quadratic in h extrapolates to h = 0
    return [np.polyfit(HS, np.array(c), 2)[-1] for c in (c_th, c_v)]


CANDIDATES = {
    "Euler (semi-implicit)": lambda th, v, f: (f, 0.0),
    "implicit damping": lambda th, v, f: (f, -KD0 * f / J0),
    "held torque, exact dynamics": lambda th, v, f: (f / 2, -C * np.cos(th) * v / (2 * J0)),
}

def report(label, rows):
    """rows: (state label, M2th, M2v, f, theta)."""
    print(f"{label:34s} {'state':>14s} {'M2th':>12s} {'M2v':>14s}   best match")
    for name, m2th, m2v, f, th, v in rows:
        scale = max(abs(f), abs(KD0 * f / J0))
        guesses = {n: g(th, v, f) for n, g in CANDIDATES.items()}
        best, err = min(((n, max(abs(m2th - a), abs(m2v - b)) / scale)
                         for n, (a, b) in guesses.items()), key=lambda t: t[1])
        print(f"{'':34s} {name:>14s} {m2th:12.4f} {m2v:14.4f}   {best} "
              f"(within {err * 100:.2f}%)")
        for n, (a, b) in guesses.items():
            print(f"{'':34s} {'':>14s} {a:12.4f} {b:14.4f}   = {n}")


if __name__ == "__main__":
    for arm in ARMS:
        sim = Sim(arm)
        rows = []
        for th, v in STATES:
            f = accel(th, v)
            m2th, m2v = coefficients(sim, th, v)
            rows.append((f"({th}, {v})", m2th, m2v, f, th, v))
        report(ARM_LABEL[arm], rows)
        print()
