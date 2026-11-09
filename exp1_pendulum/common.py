"""PD-controlled uniform-rod pendulum: ground truth, MuJoCo arms, and system identification."""

import numpy as np
import mujoco
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Uniform rod, hinge at one end (SI units).
M = 0.45359237  # 1 lb
L = 0.3048  # 1 ft
G = 9.81
J0 = M * L**2 / 3
MGL2 = M * G * L / 2  # max gravity torque

# ANYmal-C gains from legged_gym.
KP0 = 80.0
KD0 = 2.0

# Command: theta_ref(t) = A sin(2 pi f_ref t), sampled at f_ctrl and held.
A_REF = np.pi / 2
F_REF = 2.0
T_END = 4.0

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


def ref_sequence(f_ctrl, amp=A_REF, f_ref=F_REF, t_end=T_END):
    n = int(round(t_end * f_ctrl))
    return amp * np.sin(2 * np.pi * f_ref * np.arange(n) / f_ctrl)


def ground_truth(refs, f_ctrl, dec, J=J0, kp=KP0, kd=KD0):
    """Continuous-time PD under the held command, sampled at the simulator's physics steps.

    Returns theta at t = h, 2h, ..., len(refs) * dec * h with h = 1 / (f_ctrl * dec).
    """
    h = 1.0 / (f_ctrl * dec)
    out = np.empty(len(refs) * dec)
    y = np.zeros(2)
    for k, r in enumerate(refs):
        t0 = k / f_ctrl
        t_eval = t0 + h * np.arange(1, dec + 1)
        t_eval[-1] = (k + 1) / f_ctrl

        def rhs(t, s):
            th, om = s
            return [om, (-MGL2 * np.sin(th) + kp * (r - th) - kd * om) / J]

        sol = solve_ivp(rhs, (t0, (k + 1) / f_ctrl), y, method="DOP853",
                        t_eval=t_eval, rtol=1e-12, atol=1e-12)
        out[k * dec:(k + 1) * dec] = sol.y[0]
        y = sol.y[:, -1]
    return out


def model_xml(arm):
    actuator = ('<position name="pd" joint="hinge" kp="80" kv="2"/>' if arm == "C"
                else '<motor name="tau" joint="hinge"/>')
    return f"""
<mujoco model="pendulum">
  <compiler inertiafromgeom="false"/>
  <option timestep="0.005" gravity="0 0 -{G}"/>
  <visual><global offwidth="720" offheight="720"/></visual>
  <worldbody>
    <light pos="0 -1 1" dir="0 1 -1"/>
    <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.012" rgba=".3 .3 .3 1"
          contype="0" conaffinity="0"/>
    <body name="rod">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {-L / 2}" mass="{M}" diaginertia="{M * L**2 / 12} {M * L**2 / 12} 1e-8"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-L}" size="0.008" rgba=".85 .45 .15 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>{actuator}</actuator>
</mujoco>"""


class Sim:
    """One MuJoCo pendulum for a given arm; parameters are set per rollout."""

    def __init__(self, arm):
        self.arm = arm
        self.model = mujoco.MjModel.from_xml_string(model_xml(arm))
        self.model.opt.integrator = ARM_INTEGRATOR[arm]
        self.data = mujoco.MjData(self.model)

    def rollout(self, refs, f_ctrl, dec, J, kp, kd, delay=0.0):
        """Theta after each physics step; None if the rollout diverged.

        `delay` (s, may be negative) shifts the held command; each physics step uses the
        command averaged over the step, so delay = 0 reproduces the held command exactly.
        """
        m, d = self.model, self.data
        m.opt.timestep = 1.0 / (f_ctrl * dec)
        m.dof_armature[0] = J - J0  # may be negative; total inertia J must stay positive
        if self.arm == "C":
            m.actuator_gainprm[0, 0] = kp
            m.actuator_biasprm[0, 1] = -kp
            m.actuator_biasprm[0, 2] = -kd
        mujoco.mj_resetData(m, d)
        cmd = np.repeat(refs, dec) if delay == 0 else delayed_command(refs, f_ctrl, dec, delay)
        out = np.empty(len(cmd))
        if self.arm == "C":
            for i, r in enumerate(cmd):
                d.ctrl[0] = r
                mujoco.mj_step(m, d)
                out[i] = d.qpos[0]
        else:
            for i, r in enumerate(cmd):
                d.ctrl[0] = kp * (r - d.qpos[0]) - kd * d.qvel[0]
                mujoco.mj_step(m, d)
                out[i] = d.qpos[0]
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.all(np.abs(out) < 1e3):
            return None
        return out


def delayed_command(refs, f_ctrl, dec, delay):
    """Average of the held command r(t - delay) over each physics step [t_n, t_n + h).

    r(t) = refs[floor(t f_ctrl)], extended by the first value before 0 and the last after the end.
    """
    T = 1.0 / f_ctrl
    h = T / dec
    cum = np.concatenate([[0.0], np.cumsum(refs) * T])

    def integral(t):  # integral of r from 0 to t
        k = np.clip(np.floor(t / T).astype(int), 0, len(refs) - 1)
        return np.where(t < 0, refs[0] * t, cum[k] + refs[k] * (t - k * T))

    t0 = np.arange(len(refs) * dec) * h - delay
    return (integral(t0 + h) - integral(t0)) / h


def first_order_prediction(arm, h):
    """(J*, K_P*, K_D*) to first order in h.

    B: the torque is held over a step, a delay of h/2 on both feedback terms (the paper's model).
    A: semi-implicit Euler staggers velocity, so velocity feedback lags h/2 and a held command
       acts h/2 early; absorbing the command advance into (J, K_P, K_D) adds h/2 K_P to K_D and
       another h/2 K_D to J.
    C: implicit damping equals arm A with J -> J + h K_D exactly.
    """
    if arm == "A":
        return J0 + h * KD0, KP0, KD0 + h / 2 * KP0
    if arm == "B":
        return J0 + h / 2 * KD0, KP0, KD0 + h / 2 * KP0
    return J0, KP0, KD0 + h / 2 * KP0


def first_order_prediction_delay(arm, h):
    """(J*, K_P*, K_D*, d*) to first order in h when a command delay d is also identified.

    The delay cancels the half-step command advance of semi-implicit Euler (A, C), leaving
    only the velocity-feedback lag of h/2 in J; arm B has no advance, so d* = 0.
    """
    if arm == "A":
        return J0 + h / 2 * KD0, KP0, KD0, h / 2
    if arm == "B":
        return J0 + h / 2 * KD0, KP0, KD0 + h / 2 * KP0, 0.0
    return J0 - h / 2 * KD0, KP0, KD0, h / 2


PARAM0 = np.array([J0, KP0, KD0])
DIVERGED_RESIDUAL = 10.0  # rad, per sample


def identify(sim, refs, f_ctrl, dec, gt, starts, with_delay=False):
    """Least-squares fit of (J, K_P, K_D) in log space, plus the command delay in units of h.

    `starts` are parameter vectors (J, K_P, K_D) or (J, K_P, K_D, delay); returns the best fit.
    """
    n = len(gt)
    h = 1.0 / (f_ctrl * dec)

    def unpack(x):
        return (*(PARAM0 * np.exp(x[:3])), *(x[3:] * h))

    def pack(p):
        p = np.asarray(p, dtype=float)
        x = np.log(p[:3] / PARAM0)
        return np.append(x, (p[3] if len(p) > 3 else 0.0) / h) if with_delay else x

    def residual(x):
        th = sim.rollout(refs, f_ctrl, dec, *unpack(x))
        if th is None:
            return np.full(n, DIVERGED_RESIDUAL / np.sqrt(n))
        return (th - gt) / np.sqrt(n)

    fits = []
    for p in starts:
        x0 = pack(p)
        r0 = residual(x0)
        if np.all(r0 == r0[0]) and r0[0] > 0:  # diverged start
            fits.append((np.inf, np.array(unpack(x0)), x0, None))
            continue
        res = least_squares(residual, x0, method="trf", xtol=1e-12, ftol=1e-14, gtol=1e-14,
                            max_nfev=400)
        fits.append((np.sqrt(2 * res.cost), np.array(unpack(res.x)), res.x, res.jac))
    rmse, params, x, jac = min(fits, key=lambda f: f[0])
    cond = np.linalg.cond(jac) if jac is not None else np.inf
    spread = [f[1].tolist() for f in fits if np.isfinite(f[0])]
    return {"rmse": rmse, "params": params.tolist(), "cond": cond,
            "start_rmse": [f[0] for f in fits], "start_params": spread}
