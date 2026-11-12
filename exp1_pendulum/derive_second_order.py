"""Second-order counterterms for the pendulum, derived and then checked numerically.

Backward error analysis asks which continuous system a step map is really solving.  Writing
the map as exp(h D_g) for a vector field g = (v, f) + h(a1, b1) + h^2(a2, b2) and solving order
by order gives g; converting back to Newtonian form gives theta'' as a function of (theta,
theta'), and projecting the discrepancy onto the span of the three tunable operators
{sin theta, r - theta, theta'} gives the effective parameters.  Whatever falls outside that span
is a genuine obstruction to renormalizability.

Three step maps are covered:
  Euler     semi-implicit Euler with an explicit PD torque (MuJoCo arm A).
  implicit  the same map with J -> J + h KD, which is what implicit damping amounts to
            (MuJoCo's implicitfast, arm C; PhysX's own joint drive).
  held      the PD torque frozen over the step while the dynamics are integrated exactly
            (MuJoCo's RK4 arm B).
  drive     PhysX's own joint drive, read off DyCpuGpuArticulation.h: an implicit spring with
            step delta, iterated to its fixed point, after which the position is integrated with
            the new velocity.  PGS uses delta = h.
  tgs       the same drive under TGS, which advances time inside its position iterations:
            delta = h / N with N = solverPositionIterationCount.  Gravity is NOT substepped -
            PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS is off by default, so
            DyTGSDynamics.cpp integrates external forces once over the full step before the
            iterations start.  That single fact is first order in h: substepping gravity as
            well gives the wrong counterterms for every parameter.  At N = 1 this reduces to
            the PGS expressions exactly.

Isaac's explicit PD is not the held-torque map: its measured step map (probe_step_map.py and
isaac/probe_step_map_isaac.py) is semi-implicit Euler, whatever the iteration count.

Run:  python derive_second_order.py
"""

import sympy as sp

th, v, u, h, J, KP, KD, c, r = sp.symbols('theta v u h J K_P K_D c r', real=True)
G = -c * sp.sin(th) + KP * (r - th)
f = (G - KD * v) / J
D0 = lambda e: v * sp.diff(e, th) + f * sp.diff(e, v)


def modified_field(M2th, M2v, M3th, M3v):
    a1 = M2th - f / 2
    b1 = M2v - D0(f) / 2
    D1 = lambda e: a1 * sp.diff(e, th) + b1 * sp.diff(e, v)
    a2 = M3th - (D0(a1) + b1) / 2 - D0(f) / 6
    b2 = M3v - (D0(b1) + D1(f)) / 2 - D0(D0(f)) / 6
    return a1, b1, a2, b2


def acceleration(a1, b1, a2, b2):
    """theta'' as a function of (theta, u = theta'), expanded to O(h^2)."""
    g_th, g_v = v + h * a1 + h**2 * a2, f + h * b1 + h**2 * b2
    acc = sp.diff(g_th, th) * g_th + sp.diff(g_th, v) * g_v
    a1u, a2u = a1.subs(v, u), a2.subs(v, u)
    w1 = -a1u
    w2 = sp.diff(a1u, u) * a1u - a2u
    return sp.series(sp.expand(acc.subs(v, u + h * w1 + h**2 * w2)), h, 0, 3).removeO()


S, C, T = sp.symbols('S C T', real=True)          # sin(theta), cos(theta), theta as independents
poly = lambda e: sp.expand(sp.expand_trig(sp.expand(e)).subs(
    {sp.sin(th): S, sp.cos(th): C, th: T}))


def match(acc):
    """Return (j1, k1, d1), (j2, k2, d2) and the unrepresentable O(h^2) remainder."""
    a0 = poly(f.subs(v, u))                        # leading acceleration, in S/C/T form
    one = {S: 0, C: 0, T: 0, u: 0, r: 0}

    def coeffs(e):
        """(coeff of theta, of r, of sin theta, of theta', remainder) of a force-like expression."""
        e = sp.expand(e)
        parts = {k: sp.expand(e.coeff(k, 1)).subs(one) for k in (T, r, S, u)}
        rest = sp.expand(e - sum(parts[k] * k for k in parts))
        return parts, sp.simplify(rest)

    out = {}
    for order in (1, 2):
        e = poly(acc).coeff(h, order)
        for o in range(1, order):                  # subtract what the lower orders already explain
            pass
        out[order] = e
    # order h: solve j1, k1, d1 from  e1 = [k1(r-T) - d1 u]/J - a0 j1/J
    lam1, k1, d1 = sp.symbols('lam1 k1 d1')
    p, rest1 = coeffs(sp.expand(out[1] - ((k1 * (r - T) - d1 * u) / J - a0 * lam1)))
    s1 = sp.solve([p[S], p[r], p[u]], [lam1, k1, d1], dict=True)[0]
    j1 = sp.simplify(s1[lam1] * J)
    check1 = sp.simplify(p[T].subs(s1))
    rest1 = sp.simplify(rest1.subs(s1))
    # order h^2: e2 = [k2(r-T) - d2 u]/J + a0 (j1^2/J^2 - j2/J) - (k1(r-T)-d1 u) j1/J^2
    lam2, k2, d2 = sp.symbols('lam2 k2 d2')
    corr = (k1 * (r - T) - d1 * u).subs(s1) * j1 / J**2
    p2, rest2 = coeffs(sp.expand(out[2] - ((k2 * (r - T) - d2 * u) / J + a0 * lam2 - corr)))
    s2 = sp.solve([p2[S], p2[r], p2[u]], [lam2, k2, d2], dict=True)[0]
    j2 = sp.simplify(j1**2 / J - s2[lam2] * J)
    check2 = sp.simplify(p2[T].subs(s2))
    rest2 = sp.simplify(rest2.subs(s2))
    return ((j1, sp.simplify(s1[k1]), sp.simplify(s1[d1])),
            (j2, sp.simplify(s2[k2]), sp.simplify(s2[d2])),
            (check1, rest1), (check2, rest2))


# Taylor coefficients of each step map: theta' = theta + h v + h^2 M2th + h^3 M3th, and
# v' = v + h f + h^2 M2v + h^3 M3v.  For the held torque the only time dependence inside the
# step is gravity, so the derivatives of the acceleration keep only the c sin(theta) part.
A1 = -c * sp.cos(th) / J * v                 # d/dt of the acceleration, torque frozen
A2 = c * sp.sin(th) / J * v**2 - c * sp.cos(th) / J * f
def physx_drive_map():
    """Taylor coefficients of PhysX's force drive, solved to its fixed point.

    Per position iteration the solver forms, with a = delta (delta K_P + K_D) and x = 1/(1 + a/J),
        F <- F (1 - x) - x a v + K_P x delta (r - theta),
    which converges to F = x delta [K_P (r - theta) - (delta K_P + K_D) v].  The damping the
    joint really feels is therefore K_D + delta K_P, not K_D.
    """
    v_ext = v + h * (-c * sp.sin(th)) / J          # gravity is integrated before the solve
    v1 = v_ext + h * (KP * (r - th) - (h * KP + KD) * v_ext) / (J + h * (h * KP + KD))
    th1 = th + h * v1                              # position integrated with the new velocity
    sv = sp.series(sp.expand(v1 - v), h, 0, 4).removeO()
    st = sp.series(sp.expand(th1 - th - h * v), h, 0, 4).removeO()
    return tuple(sp.simplify(e.coeff(h, k)) for e, k in ((st, 2), (sv, 2), (st, 3), (sv, 3)))


def tgs_drive_map(n):
    """Taylor coefficients of a TGS step: gravity once over h, then n drive substeps of h / n."""
    d = h / n
    grav = -c * sp.sin(th) / J
    fd = (KP * (r - th) - KD * v) / J                  # the drive alone
    e = sp.Symbol('delta', positive=True)
    v1 = v + e * (KP * (r - th) - (e * KP + KD) * v) / (J + e * (e * KP + KD))
    sv = sp.series(sp.expand(v1 - v), e, 0, 4).removeO()
    st = sp.series(sp.expand(th + e * v1 - th - e * v), e, 0, 4).removeO()
    D0d = lambda x: v * sp.diff(x, th) + fd * sp.diff(x, v)
    a1 = st.coeff(e, 2) - fd / 2
    b1 = sv.coeff(e, 2) - D0d(fd) / 2
    D1d = lambda x: a1 * sp.diff(x, th) + b1 * sp.diff(x, v)
    a2 = st.coeff(e, 3) - (D0d(a1) + b1) / 2 - D0d(fd) / 6
    b2 = sv.coeff(e, 3) - (D0d(b1) + D1d(fd)) / 2 - D0d(D0d(fd)) / 6
    # the drive runs for a time h as the flow of its own modified field, evaluated at delta
    gx, gv = v + d * a1 + d**2 * a2, fd + d * b1 + d**2 * b2
    Dg = lambda x: gx * sp.diff(x, th) + gv * sp.diff(x, v)
    after = [y + h * g + h**2 / 2 * Dg(g) + h**3 / 6 * Dg(Dg(g))
             for y, g in ((th, gx), (v, gv))]
    after = [sp.expand(x.subs(v, v + h * grav)) for x in after]   # gravity applied first
    sT = sp.series(sp.expand(after[0] - th - h * v), h, 0, 4).removeO()
    sV = sp.series(sp.expand(after[1] - v - h * f), h, 0, 4).removeO()
    return tuple(sp.simplify(x.coeff(h, k)) for x, k in ((sT, 2), (sV, 2), (sT, 3), (sV, 3)))


TGS_ITERS = 32          # Isaac's default solverPositionIterationCount
ARMS = (("Euler: semi-implicit, explicit PD (MuJoCo A)",
         (f, sp.Integer(0), sp.Integer(0), sp.Integer(0))),
        ("implicit damping: J -> J + h K_D (MuJoCo C, PhysX drive)",
         (f, -KD*f/J, -KD*f/J, KD**2*f/J**2)),
        ("held torque: exact dynamics (MuJoCo B)",
         (f / 2, A1 / 2, A1 / 6, A2 / 6)),
        ("drive: PhysX force drive under PGS (Isaac C)", physx_drive_map()),
        (f"tgs: the same drive under TGS, {TGS_ITERS} substeps (Isaac B)",
         tgs_drive_map(TGS_ITERS)))


def derive():
    for name, M in ARMS:
        first, second, chk1, chk2 = match(acceleration(*modified_field(*M)))
        print("=" * 76)
        print(name)
        for lab, s1, s2 in zip(("J", "K_P", "K_D"), first, second):
            print(f"  {lab}e = {lab} + h*({s1}) + h^2*({sp.simplify(s2)})")
        d1, d2 = sp.Matrix(first), sp.Matrix(second)
        p1, p2 = -d1, d1.jacobian((J, KP, KD)) * d1 - d2
        for lab, x1, x2 in zip(("J", "K_P", "K_D"), p1, p2):
            print(f"  {lab}* - {lab} = h*({sp.simplify(x1)}) + h^2*({sp.simplify(sp.expand(x2))})")
        print(f"  consistency of the theta and r coefficients: {chk1[0]}, {chk2[0]}")
        print(f"  unrepresentable O(h^2) remainder: {chk2[1]}")


def step_map(kind, hh, th0, v0, tend, num):
    """Advance the discrete map from (th0, v0) to time tend."""
    import numpy as np
    from scipy.integrate import solve_ivp
    Jv, KPv, KDv, cv, r0 = num
    t_, v_ = th0, v0
    for _ in range(int(round(tend / hh))):
        if kind.startswith("held"):   # torque frozen, dynamics integrated to machine precision
            tau = KPv * (r0 - t_) - KDv * v_
            sol = solve_ivp(lambda _t, y: [y[1], (-cv * np.sin(y[0]) + tau) / Jv],
                            (0, hh), [t_, v_], rtol=1e-13, atol=1e-14, method="DOP853")
            t_, v_ = sol.y[0, -1], sol.y[1, -1]
        elif kind.startswith("drive") or kind.startswith("tgs"):
            n = TGS_ITERS if kind.startswith("tgs") else 1
            v_ += hh * (-cv * np.sin(t_)) / Jv          # gravity once over the whole step
            dd = hh / n
            for _ in range(n):                          # then the drive, substep by substep
                v_ += dd * (KPv * (r0 - t_) - (dd * KPv + KDv) * v_) / (Jv + dd * (dd * KPv + KDv))
                t_ += dd * v_
        else:
            denom = Jv + hh * KDv if kind.startswith("implicit") else Jv
            v_ += hh * (-cv * np.sin(t_) + KPv * (r0 - t_) - KDv * v_) / denom
            t_ += hh * v_
    return t_


def numerical_check():
    """Each extra order should drop the error by one power of h.

    Everything here is lambdified straight out of the symbolic solution, including the initial
    condition: the modified system's state is not the discrete state, since its velocity is
    theta' = v + h a1 + h^2 a2 rather than v.
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    from common import G as GRAV, J0, KD0, KP0, M, L

    num = (J0, KP0, KD0, M * GRAV * L / 2, 1.0)
    subs = dict(zip((J, KP, KD, c, r), num))
    tend, th0, v0 = 0.01, 0.3, 0.0

    def flow(Je, KPe, KDe, thd0):
        Jv, KPv, KDv, cv, r0 = num
        rhs = lambda t, y: [y[1], (-cv * np.sin(y[0]) + KPe * (r0 - y[0]) - KDe * y[1]) / Je]
        return solve_ivp(rhs, (0, tend), [th0, thd0], rtol=1e-13, atol=1e-14,
                         method="DOP853").y[0, -1]

    for name, M_ in ARMS:
        a1, b1, a2, b2 = modified_field(*M_)
        first, second, _, _ = match(acceleration(a1, b1, a2, b2))
        at = [float(e.subs(subs).subs({th: th0, v: v0})) for e in (a1, a2)]
        s1 = [float(e.subs(subs)) for e in first]
        s2 = [float(e.subs(subs)) for e in second]
        Jv, KPv, KDv = num[:3]
        print("=" * 76)
        print(name)
        print(f"  a1 = {at[0]:+.4g}, a2 = {at[1]:+.4g} rad/s per power of h")
        prev = None
        for hh in (5e-4, 2.5e-4, 1.25e-4, 6.25e-5):
            sim = step_map(name, hh, th0, v0, tend, num)
            row = (sim - flow(Jv, KPv, KDv, v0),
                   sim - flow(Jv + hh * s1[0], KPv + hh * s1[1], KDv + hh * s1[2],
                              v0 + hh * at[0]),
                   sim - flow(Jv + hh * s1[0] + hh**2 * s2[0], KPv + hh * s1[1] + hh**2 * s2[1],
                              KDv + hh * s1[2] + hh**2 * s2[2],
                              v0 + hh * at[0] + hh**2 * at[1]))
            o = ["   "] * 3 if prev is None else [f"{np.log2(abs(a / b)):4.1f}"
                                                  for a, b in zip(prev, row)]
            print(f"  h={hh*1e3:6.4f} ms  none {row[0]:+.2e} [{o[0]}]  first {row[1]:+.2e} "
                  f"[{o[1]}]  second {row[2]:+.2e} [{o[2]}]")
            prev = row
        print("  (bracket = measured order of convergence against the previous row)")


if __name__ == "__main__":
    derive()
    numerical_check()
