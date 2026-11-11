"""System identification shared by every experiment and both engines.

Having one optimizer is the point: a fit that behaves differently in MuJoCo and in Isaac cannot
be compared, and the difference is then an artifact of the fitter rather than of the physics.
The only thing an engine supplies is how to roll out a trajectory, plus optionally a way to
evaluate several parameter vectors at once.
"""

import numpy as np
from scipy.optimize import least_squares

DIVERGED_RESIDUAL = 10.0  # residual reported for a rollout that blew up


N_RANDOM_STARTS = 2


def start_points(h, predicted, prev, rng, with_delay, param0):
    """The starting points every fit uses, whichever engine it runs in.

    `predicted` is the scheme's own first-order prediction.  Inflated inertia keeps explicit PD
    stable when h K_D / J is no longer small, and the random restarts cover the poor local
    optima the landscape has at coarse control rates.
    """
    j0, kp0, kd0 = param0
    starts = [np.append(param0, 0.0) if with_delay else np.asarray(param0),
              np.asarray(predicted, float)]
    starts += [np.array([j0 + k * h * kd0, kp0, kd0, h / 2][:3 + with_delay])
               for k in (1, 2, 4)]
    for _ in range(N_RANDOM_STARTS):
        jitter = np.asarray(param0) * np.exp(rng.normal(0, 0.5, 3))
        starts.append(np.append(jitter, rng.uniform(0, h)) if with_delay else jitter)
    return starts + ([np.asarray(prev, float)] if prev is not None else [])


def identify(rollout, gt, h, starts, param0, with_delay=False, batch=None,
             inertia_floor=None):
    """Least-squares fit of (J, K_P, K_D) in log space, plus the command delay in units of h.

    `rollout(J, kp, kd, delay)` returns the simulated angles, already concatenated over every
    command, so that one parameter set is fitted to all of them at once; None means it diverged.
    `batch(X)` is optional: given several parameter vectors in the scaled space it returns one
    residual row each, which lets a simulator that steps many copies at once evaluate the
    Jacobian in a single rollout.  It changes what a Jacobian costs, not what the fit is.
    `inertia_floor` switches the first parameter from log space to a linear coordinate in units
    of h K_D / 2, bounded so the total inertia stays above zero.  A joint that carries no rotor
    inertia of its own has nothing to take a logarithm of, and its counterterm may be negative.
    `starts` are parameter vectors (J, K_P, K_D) or (J, K_P, K_D, delay); returns the best fit.
    """
    gt = np.asarray(gt)
    n = gt.size          # a multi-joint ground truth is 2-D
    unit = h / 2 * param0[2]                  # the predicted rotor inertia
    linear = inertia_floor is not None
    bounds = (-np.inf, np.inf)
    if linear:
        lo = [-0.9 * inertia_floor / unit, -2.0, -2.0] + ([-5.0] if with_delay else [])
        hi = [0.9 * inertia_floor / unit, 2.0, 2.0] + ([5.0] if with_delay else [])
        bounds = (np.array(lo), np.array(hi))

    def unpack(x):
        head = ((x[0] * unit,) if linear else (param0[0] * np.exp(x[0]),))
        return (*head, *(np.asarray(param0[1:3]) * np.exp(x[1:3])), *(x[3:] * h))

    def pack(p):
        p = np.asarray(p, dtype=float)
        first = p[0] / unit if linear else np.log(p[0] / param0[0])
        x = np.append(first, np.log(p[1:3] / np.asarray(param0[1:3])))
        x = np.append(x, (p[3] if len(p) > 3 else 0.0) / h) if with_delay else x
        return np.clip(x, bounds[0] * 0.99, bounds[1] * 0.99) if linear else x

    def residual(x):
        theta = rollout(*unpack(x))
        if theta is None:
            return np.full(n, DIVERGED_RESIDUAL)
        return (np.asarray(theta) - gt).ravel() / np.sqrt(n)

    jac = None
    if batch is not None:
        eps = 1e-4   # large enough to stay above the float32 noise of a batched simulator
        def jac(x):  # noqa: E306
            steps = np.eye(len(x)) * eps
            rows = batch([unpack(x + d) for d in steps] + [unpack(x - d) for d in steps])
            half = len(steps)
            return ((np.asarray(rows[:half]) - np.asarray(rows[half:])) / (2 * eps)).T

    fits = []
    for p in starts:
        x0 = pack(p)
        r0 = residual(x0)
        if np.all(r0 == r0[0]) and r0[0] > 0:  # diverged start
            fits.append((np.inf, np.array(unpack(x0)), x0, None))
            continue
        res = least_squares(residual, x0, method="trf", jac=jac or "2-point", bounds=bounds,
                            xtol=1e-12, ftol=1e-14, gtol=1e-14, max_nfev=400)
        fits.append((np.sqrt(2 * res.cost), np.array(unpack(res.x)), res.x, res.jac))
    rmse, params, x, jac_out = min(fits, key=lambda f: f[0])
    cond = np.linalg.cond(jac_out) if jac_out is not None else np.inf
    spread = [f[1].tolist() for f in fits if np.isfinite(f[0])]
    return {"rmse": rmse, "params": params.tolist(), "cond": cond,
            "start_rmse": [f[0] for f in fits], "start_params": spread}
