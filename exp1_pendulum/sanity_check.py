"""Checks that the MuJoCo pendulum and the ground-truth ODE describe the same system."""

import time

import mujoco
import numpy as np

from common import (ARMS, J0, KD0, KP0, MGL2, Sim, first_order_prediction, ground_truth,
                    model_xml, ref_sequence)

# Inertia about the hinge and gravity torque.
m = mujoco.MjModel.from_xml_string(model_xml("A"))
d = mujoco.MjData(m)
d.qpos[0] = np.pi / 2
mujoco.mj_forward(m, d)
Mfull = np.zeros((1, 1))
mujoco.mj_fullM(m, d, Mfull)
print(f"J: mujoco {Mfull[0, 0]:.10f}  analytic {J0:.10f}")
print(f"gravity torque at pi/2: mujoco {d.qfrc_bias[0]:.10f}  analytic {MGL2:.10f}")

# Negative armature lowers the total inertia.
m.dof_armature[0] = -0.5 * J0
mujoco.mj_forward(m, d)
mujoco.mj_fullM(m, d, Mfull)
print(f"J with armature -J0/2: {Mfull[0, 0]:.10f}  expected {0.5 * J0:.10f}")

# Native PD integrated by RK4 at a high rate approximates continuous PD: should match ground truth.
f_ctrl, dec = 50, 400
refs = ref_sequence(f_ctrl)
gt = ground_truth(refs, f_ctrl, dec)
sim = Sim("C")
sim.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
th = sim.rollout(refs, f_ctrl, dec, J0, KP0, KD0)
print(f"RK4 native PD at 20 kHz vs ground truth: max |err| = {np.max(np.abs(th - gt)):.2e} rad")

# Each arm at the legged_gym default, measured vs first-order renormalized parameters.
f_ctrl, dec = 50, 4
refs = ref_sequence(f_ctrl)
gt = ground_truth(refs, f_ctrl, dec)
h = 1 / (f_ctrl * dec)
for arm in ARMS:
    sim = Sim(arm)
    t0 = time.perf_counter()
    th_meas = sim.rollout(refs, f_ctrl, dec, J0, KP0, KD0)
    dt = time.perf_counter() - t0
    th_ren = sim.rollout(refs, f_ctrl, dec, *first_order_prediction(arm, h))
    rmse = lambda th: np.sqrt(np.mean((th - gt) ** 2)) if th is not None else np.inf
    print(f"arm {arm} @200 Hz: RMSE measured {rmse(th_meas):.3e}, renormalized {rmse(th_ren):.3e} "
          f"(rollout {dt * 1e3:.1f} ms)")
