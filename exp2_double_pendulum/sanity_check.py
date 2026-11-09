"""Checks that the model is the intended mechanism, that the reference is reliable, and that the
motion is genuinely chaotic over the window used."""

import mujoco
import numpy as np

from common import (G, HINGE_FRAC, KD0, L1, L2, M1, M2, PARAM0, F_CTRL, Sim, ref_sequence,
                    renormalized)

sim = Sim()
m, d = sim.model, sim.data

# Link 2 is hinged near its own mass centre, so gravity barely restores it.
com = m.body_ipos[m.body("link2").id, 2]
print(f"link 2 mass centre {abs(com) * 100:.2f} cm from its hinge "
      f"(expected {abs(HINGE_FRAC - 0.5) * L2 * 100:.2f} cm); "
      f"max gravity torque on it {M2 * G * abs(com):.4f} N m, "
      f"against {M2 * G * L2 / 2:.4f} N m if it hung from its end")

# Joint-space inertia at rest, against the analytic value for the two rods.
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
Mfull = np.zeros((2, 2))
mujoco.mj_fullM(m, d, Mfull)
d2 = L1 - com  # link 2 mass centre below the first hinge, hanging straight down (com < 0)
analytic = M1 * L1**2 / 3 + M2 * d2**2 + M2 * L2**2 / 12
print(f"M11 at rest: mujoco {Mfull[0, 0]:.10f}  analytic {analytic:.10f}")

# Armature is exactly a constant added to the joint-space inertia of joint 1.
m.dof_armature[0] = 0.01
mujoco.mj_forward(m, d)
mujoco.mj_fullM(m, d, Mfull)
print(f"M11 with armature 0.01: {Mfull[0, 0]:.10f}  expected {analytic + 0.01:.10f}")
m.dof_armature[0] = 0.0

# The reference integration is converged: tightening the tolerance barely moves it.
refs = ref_sequence(3.0)
gt = sim.ground_truth(refs)
gt_tight = sim.ground_truth(refs, rtol=1e-13)
print(f"reference at rtol 1e-12 vs 1e-13: max difference {np.max(np.abs(gt - gt_tight)):.2e} rad "
      f"over 3 s")

# Sensitivity to initial conditions: the signature of chaos.
base = sim.ground_truth(refs)
sep = np.max(np.abs(sim.ground_truth(refs, y0=[1e-9, 0, 0, 0]) - base), axis=1)
t = np.arange(1, len(refs) + 1) / F_CTRL
print("separation from a 1e-9 rad perturbation:")
for tt in (0.5, 1.0, 2.0, 3.0):
    print(f"   t = {tt:4.2f} s: {sep[np.argmin(np.abs(t - tt))]:.2e} rad")
sel = (sep > 1e-8) & (sep < 1e-1)
lam = np.polyfit(t[sel], np.log(sep[sel]), 1)[0]
print(f"Lyapunov exponent {lam:.1f} /s, so the reference's own 1e-12 error reaches "
      f"{1e-12 * np.exp(lam * 3.0):.1e} rad at 3 s")
print(f"link 2 turns in 3 s: {(base[:, 1].max() - base[:, 1].min()) / 2 / np.pi:.1f}")

# What the renormalized parameters are worth at the legged_gym default.
dec = 4
h = 1 / (F_CTRL * dec)
refs1 = ref_sequence(1.0)
gt1 = sim.ground_truth(refs1)
for name, p in (("measured", PARAM0), ("renormalized", renormalized(h))):
    theta = sim.rollout(refs1, dec, *p)
    print(f"{name:13s} at 200 Hz: armature {p[0]:.3e} kg m^2, "
          f"RMSE over 1 s {np.sqrt(np.mean((theta - gt1) ** 2)):.3e} rad")
