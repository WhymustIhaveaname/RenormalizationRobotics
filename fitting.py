"""System identification shared by every experiment and both engines.

Having one optimizer is the point: a fit that behaves differently in MuJoCo and in Isaac cannot
be compared, and the difference is then an artifact of the fitter rather than of the physics.
The only thing an engine supplies is how to roll out a trajectory, plus optionally a way to
evaluate several parameter vectors at once.
"""

import threading

import numpy as np
from scipy.optimize import least_squares

DIVERGED_RESIDUAL = 10.0  # residual reported for a rollout that blew up


N_RANDOM_STARTS = 2


class _Gang:
    """Lets several fits share one batched rollout.

    Each start runs in its own thread and calls scipy exactly as it would on its own; this
    only collects the rollout requests that are outstanding at the same moment and issues
    them as a single call.  The optimizer, its tolerances and its iterates are untouched, so
    the answer is the serial answer - what changes is that a simulator which costs the same
    for eight worlds as for sixty-four is asked once instead of seven times.
    """

    def __init__(self, batch, n_fits):
        self.batch, self.active = batch, n_fits
        self.cv = threading.Condition()
        self.pending, self.results, self.round = {}, {}, 0

    def submit(self, rows):
        me = threading.get_ident()
        with self.cv:
            self.pending[me] = rows
            mine = self.round
            if len(self.pending) >= self.active:
                self._dispatch()
            while self.round == mine:
                self.cv.wait()
            return self.results.pop(me)

    def retire(self):
        with self.cv:
            self.active -= 1
            if self.pending and len(self.pending) >= self.active:
                self._dispatch()

    def _dispatch(self):
        order = list(self.pending)
        out = self.batch([r for t in order for r in self.pending[t]])
        i = 0
        for t in order:
            k = len(self.pending[t])
            self.results[t], i = out[i:i + k], i + k
        self.pending.clear()
        self.round += 1
        self.cv.notify_all()


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
    `batch(params)` is optional: given several parameter vectors it returns one trajectory each,
    which lets a simulator that steps many copies at once evaluate the Jacobian in a single
    rollout.  It changes what a Jacobian costs, not what the fit is.
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

    def clean(theta):
        """A rollout that blew up is reported as a large residual, never as NaN."""
        if theta is None:
            return np.full(n, DIVERGED_RESIDUAL), False
        theta = np.asarray(theta)
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e3:
            return np.full(n, DIVERGED_RESIDUAL), False
        return (theta - gt).ravel() / np.sqrt(n), True

    eps = 1e-4   # large enough to stay above the float32 noise of a batched simulator

    def solve(p, roll):
        """One start, with `roll(params) -> trajectories` however they get evaluated."""
        def residual(x):
            return clean(roll([unpack(x)])[0])[0]

        def jac(x):
            steps = np.eye(len(x)) * eps
            half = len(steps)
            raw = roll([unpack(x + d) for d in steps] + [unpack(x - d) for d in steps])
            rows, ok = zip(*(clean(t) for t in raw))
            J = (np.asarray(rows[:half]) - np.asarray(rows[half:])) / (2 * eps)
            J[[k for k in range(half) if not (ok[k] and ok[half + k])]] = 0.0
            return J.T

        x0 = pack(p)
        r0 = residual(x0)
        if np.all(r0 == r0[0]) and r0[0] > 0:  # diverged start
            return (np.inf, np.array(unpack(x0)), x0, None)
        res = least_squares(residual, x0, method="trf", jac=jac if batch else "2-point",
                            bounds=bounds, xtol=1e-12, ftol=1e-14, gtol=1e-14, max_nfev=400)
        return (np.sqrt(2 * res.cost), np.array(unpack(res.x)), res.x, res.jac)

    fits = []
    if batch is None:
        serial = lambda rows: [rollout(*row) for row in rows]
        fits = [solve(p, serial) for p in starts]
    else:
        # Every start advances at once, so a simulator that charges the same for eight worlds
        # as for sixty-four is asked once per round instead of once per start.
        gang = _Gang(batch, len(starts))
        fits = [None] * len(starts)

        def worker(k, p):
            try:
                fits[k] = solve(p, gang.submit)
            finally:
                gang.retire()

        threads = [threading.Thread(target=worker, args=(k, p), daemon=True)
                   for k, p in enumerate(starts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    rmse, params, x, jac_out = min(fits, key=lambda f: f[0])
    cond = np.linalg.cond(jac_out) if jac_out is not None else np.inf
    spread = [f[1].tolist() for f in fits if np.isfinite(f[0])]
    return {"rmse": rmse, "params": params.tolist(), "cond": cond,
            "start_rmse": [f[0] for f in fits], "start_params": spread}
