"""Driven double pendulum: a chaotic two-link system with one PD-controlled joint.

Link 1 is driven by the same PD controller as experiment 1. Link 2 hangs from the end of link 1
and is free; it is hinged near its own middle, so gravity barely restores it and it tumbles, as in
a chaotic-pendulum desk toy.

The paper's multi-body prediction (its appendix) is a constant shift of the joint-space inertia,
M_sim = M_real + dt * K_D. In MuJoCo that is exactly a rotor inertia (`armature`) on the driven
joint, which the physical joint does not have.
"""

import mujoco
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Two uniform rods, same mass and length as the single pendulum (SI units).
M1 = M2 = 0.45359237  # 1 lb
L1 = L2 = 0.3048  # 1 ft
HINGE_FRAC = 0.40  # link 2 is hinged this far along its own length, so its mass centre is close
G = 9.81

# ANYmal-C gains from legged_gym, on joint 1 only; joint 2 is unactuated.
KP0 = 80.0
KD0 = 2.0

# Command: theta_ref(t) = A sin(2 pi f_drive t), sampled at F_CTRL and held.
# 4 Hz with amplitude pi/4 gives the same 19.7 rad/s peak command velocity as experiment 1,
# and drives the free link into a tumbling, sensitive regime.
F_CTRL = 50
A_REF = np.pi / 4
F_DRIVE = 4.0
T_FIT = 1.0  # s of trajectory used for the error metric and for identification
T_SHOW = 2.0  # s shown in the trajectory panel

PARAM0 = np.array([0.0, KP0, KD0])  # measured: no rotor inertia on the joint
M11_MIN = 0.05169  # kg m^2, the smallest joint-space inertia of joint 1 over the free angle
DIVERGED_RESIDUAL = 10.0  # rad, per sample

ARMS = ("A", "B", "C")
ARM_INTEGRATOR = {
    "A": mujoco.mjtIntegrator.mjINT_EULER,
    "B": mujoco.mjtIntegrator.mjINT_RK4,
    "C": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
ARM_LABEL = {
    "A": "A: Euler + explicit PD",
    "B": "B: RK4 + explicit PD (torque held)",
    "C": "C: implicitfast + native PD",
}


def model_xml(arm="B"):
    a, b = HINGE_FRAC * L2, (1 - HINGE_FRAC) * L2  # link 2 reaches a above and b below its hinge
    actuator = ('<position name="pd1" joint="j1" kp="80" kv="2"/>' if arm == "C"
                else '<motor name="tau1" joint="j1"/>')
    return f"""
<mujoco model="double_pendulum">
  <compiler inertiafromgeom="false"/>
  <option timestep="0.005" gravity="0 0 -{G}" integrator="RK4"/>
  <visual><global offwidth="900" offheight="900"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".86 .88 .91" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0 -1 1" dir="0 1 -1"/>
    <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.012" rgba=".3 .3 .3 1"
          contype="0" conaffinity="0"/>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {-L1 / 2}" mass="{M1}" diaginertia="{M1 * L1**2 / 12} {M1 * L1**2 / 12} 1e-9"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0 0 {-L1}" size="0.009" rgba=".25 .25 .25 1"
            contype="0" conaffinity="0"/>
      <body name="link2" pos="0 0 {-L1}">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 {(a - b) / 2}" mass="{M2}"
                  diaginertia="{M2 * L2**2 / 12} {M2 * L2**2 / 12} 1e-9"/>
        <geom name="g2" type="capsule" fromto="0 0 {a} 0 0 {-b}" size="0.007" rgba=".25 .25 .25 1"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>{actuator}</actuator>
</mujoco>"""


def ref_sequence(t_end, f_ctrl=F_CTRL):
    n = int(round(t_end * f_ctrl))
    return A_REF * np.sin(2 * np.pi * F_DRIVE * np.arange(n) / f_ctrl)


class Sim:
    """One MuJoCo double pendulum; parameters are set per rollout."""

    def __init__(self, arm="B"):
        self.arm = arm
        self.model = mujoco.MjModel.from_xml_string(model_xml(arm))
        self.model.opt.integrator = ARM_INTEGRATOR[arm]
        self.data = mujoco.MjData(self.model)
        # The reference uses its own torque-driven copy, so it does not depend on how the
        # simulator under test applies the PD force.
        self._gt_model = mujoco.MjModel.from_xml_string(model_xml("B"))
        self._gt_data = mujoco.MjData(self._gt_model)

    def rollout(self, refs, dec, armature, kp, kd, delay=0.0, strict=True, f_ctrl=F_CTRL):
        """Joint angles after each control step; None if the rollout diverged.

        `delay` (s, may be negative) shifts the held command; each physics step uses the command
        averaged over the step, so delay = 0 reproduces the held command exactly.
        """
        m, d = self.model, self.data
        m.opt.timestep = 1.0 / (f_ctrl * dec)
        m.dof_armature[0] = armature  # may be negative; total joint inertia must stay positive
        if self.arm == "C":
            m.actuator_gainprm[0, 0] = kp
            m.actuator_biasprm[0, 1] = -kp
            m.actuator_biasprm[0, 2] = -kd
        mujoco.mj_resetData(m, d)
        cmd = (np.repeat(refs, dec) if delay == 0
               else delayed_command(refs, f_ctrl, dec, delay))
        out = np.empty((len(refs), 2))
        for k in range(len(refs)):
            for j in range(dec):
                r = cmd[k * dec + j]
                d.ctrl[0] = r if self.arm == "C" else kp * (r - d.qpos[0]) - kd * d.qvel[0]
                mujoco.mj_step(m, d)
            out[k] = d.qpos
        diverged = (d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                    or not np.all(np.abs(out) < 1e3))
        return None if (diverged and strict) else out

    def rollout_dense(self, refs, dec, armature, kp, kd, f_ctrl=F_CTRL):
        """Joint angles after every physics step, for rendering."""
        m, d = self.model, self.data
        m.opt.timestep = 1.0 / (f_ctrl * dec)
        m.dof_armature[0] = armature
        mujoco.mj_resetData(m, d)
        out = np.empty((len(refs) * dec, 2))
        i = 0
        for r in refs:
            for _ in range(dec):
                d.ctrl[0] = kp * (r - d.qpos[0]) - kd * d.qvel[0]
                mujoco.mj_step(m, d)
                out[i] = d.qpos
                i += 1
        return out

    def ground_truth(self, refs, rtol=1e-12, y0=None, sub=1, f_ctrl=F_CTRL):
        """Exact continuous-time motion under the held command, sampled at each control step.

        The right-hand side is MuJoCo's own rigid-body dynamics, so the simulator and the
        reference describe the same mechanism; only the time discretization differs.
        """
        m, d = self._gt_model, self._gt_data
        m.dof_armature[0] = 0.0

        def rhs(t, s, r):
            d.qpos[:], d.qvel[:] = s[:2], s[2:]
            d.ctrl[0] = KP0 * (r - s[0]) - KD0 * s[2]
            mujoco.mj_forward(m, d)
            return np.concatenate([s[2:], d.qacc])

        y = np.zeros(4) if y0 is None else np.array(y0, dtype=float)
        out = np.empty((len(refs) * sub, 2))
        for k, r in enumerate(refs):
            t0, t1 = k / f_ctrl, (k + 1) / f_ctrl
            t_eval = np.linspace(t0, t1, sub + 1)[1:]
            sol = solve_ivp(rhs, (t0, t1), y, args=(r,), method="DOP853", rtol=rtol, atol=rtol,
                            t_eval=t_eval)
            y = sol.y[:, -1]
            out[k * sub:(k + 1) * sub] = sol.y[:2].T
        return out


def delayed_command(refs, f_ctrl, dec, delay):
    """Average of the held command r(t - delay) over each physics step."""
    T = 1.0 / f_ctrl
    h = T / dec
    cum = np.concatenate([[0.0], np.cumsum(refs) * T])

    def integral(t):
        k = np.clip(np.floor(t / T).astype(int), 0, len(refs) - 1)
        return np.where(t < 0, refs[0] * t, cum[k] + refs[k] * (t - k * T))

    t0 = np.arange(len(refs) * dec) * h - delay
    return (integral(t0 + h) - integral(t0)) / h


def first_order_prediction(arm, h):
    """(armature, K_P, K_D) to first order, with no command-delay parameter.

    The measured joint has no rotor inertia, so these shifts are the whole prediction; they
    mirror the single pendulum with J_measured replaced by zero.
    """
    if arm == "A":
        return np.array([h * KD0, KP0, KD0 + h / 2 * KP0])
    if arm == "B":
        return np.array([h / 2 * KD0, KP0, KD0 + h / 2 * KP0])
    return np.array([0.0, KP0, KD0 + h / 2 * KP0])


def first_order_prediction_delay(arm, h):
    """(armature, K_P, K_D, d) when a command delay is identified as well."""
    if arm == "A":
        return np.array([h / 2 * KD0, KP0, KD0, h / 2])
    if arm == "B":
        return np.array([h / 2 * KD0, KP0, KD0 + h / 2 * KP0, 0.0])
    return np.array([-h / 2 * KD0, KP0, KD0, h / 2])


def renormalized(h):
    """The RK4 prediction, used for the behaviour figure and the video."""
    return first_order_prediction("B", h)


def rmse(sim, refs, dec, params, gt, f_ctrl=F_CTRL):
    theta = sim.rollout(refs, dec, *params, f_ctrl=f_ctrl)
    return None if theta is None else float(np.sqrt(np.mean((theta - gt) ** 2)))


def identify(sim, refs, dec, gt, starts, f_ctrl=F_CTRL, with_delay=False):
    """Fit (armature, K_P, K_D[, delay]); the linear parameters may go negative."""
    h = 1.0 / (f_ctrl * dec)
    unit = h / 2 * KD0  # the predicted rotor inertia, the scale for the linear parameter
    n = gt.size

    def unpack(x):
        p = [x[0] * unit, KP0 * np.exp(x[1]), KD0 * np.exp(x[2])]
        return np.array(p + [x[3] * h] if with_delay else p)

    def residual(x):
        theta = sim.rollout(refs, dec, *unpack(x), f_ctrl=f_ctrl)
        if theta is None:
            return np.full(n, DIVERGED_RESIDUAL / np.sqrt(n))
        return (theta - gt).ravel() / np.sqrt(n)

    lo = [-0.9 * M11_MIN / unit, -2.0, -2.0] + ([-5.0] if with_delay else [])
    hi = [0.9 * M11_MIN / unit, 2.0, 2.0] + ([5.0] if with_delay else [])
    best = None
    for p in starts:
        p = np.asarray(p, dtype=float)
        x0 = [p[0] / unit, np.log(p[1] / KP0), np.log(p[2] / KD0)]
        if with_delay:
            x0.append((p[3] if len(p) > 3 else 0.0) / h)
        x0 = np.clip(x0, np.array(lo) * 0.99, np.array(hi) * 0.99)
        res = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-12, ftol=1e-14, gtol=1e-14,
                            max_nfev=400)
        if best is None or np.sqrt(2 * res.cost) < best[0]:
            best = (np.sqrt(2 * res.cost), unpack(res.x))
    return {"rmse": best[0], "params": best[1].tolist()}
