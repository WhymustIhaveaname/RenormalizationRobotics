"""Batched least-squares fitting for Isaac rollouts.

An Isaac step costs the same whether one environment or sixteen are stepped, so the finite
difference Jacobian is evaluated in a single batched rollout: environment 0 carries the current
parameters and the rest carry the perturbations.
"""

import numpy as np
from scipy.optimize import least_squares

EPS = 1e-4  # relative step for the central differences, in the scaled parameter space


def fit(rollout, gt, x0_list, scales, base, bounds=None, max_nfev=60):
    """Fit scaled parameters x so that rollout(base + x * scales) matches gt.

    `rollout(P)` takes (n, ndim) physical parameter rows and returns (n, ...) trajectories.
    Returns the best of the starting points in `x0_list`.
    """
    ndim = len(scales)
    scales = np.asarray(scales, float)
    base = np.asarray(base, float)
    n = gt.size

    def physical(X):
        return base + np.asarray(X, float).reshape(-1, ndim) * scales

    def residual(x):
        theta = rollout(physical(x[None, :]))[0]
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e3:
            return np.full(n, 10.0 / np.sqrt(n))
        return (theta - gt).ravel() / np.sqrt(n)

    def jac(x):
        steps = np.eye(ndim) * EPS
        X = np.concatenate([x + steps, x - steps])
        out = rollout(physical(X))
        J = np.empty((n, ndim))
        for k in range(ndim):
            plus, minus = out[k], out[ndim + k]
            bad = [t for t in (plus, minus)
                   if not np.all(np.isfinite(t)) or np.max(np.abs(t)) > 1e3]
            if bad:
                J[:, k] = 0.0
            else:
                J[:, k] = ((plus - minus).ravel() / (2 * EPS)) / np.sqrt(n)
        return J

    best = None
    for x0 in x0_list:
        x0 = np.asarray(x0, float)
        if bounds is not None:
            x0 = np.clip(x0, bounds[0] + 1e-9, bounds[1] - 1e-9)
        res = least_squares(residual, x0, jac=jac, bounds=bounds or (-np.inf, np.inf),
                            xtol=1e-10, ftol=1e-12, gtol=1e-12, max_nfev=max_nfev)
        rmse = np.sqrt(2 * res.cost)
        if best is None or rmse < best[0]:
            best = (float(rmse), physical(res.x[None, :])[0])
    return {"rmse": best[0], "params": best[1].tolist()}
