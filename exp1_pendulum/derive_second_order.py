"""Second-order counterterms for the pendulum, derived and then checked numerically.

Backward error analysis asks which continuous system a step map is really solving.  Writing
the map as exp(h D_g) for a vector field g = (v, f) + h(a1, b1) + h^2(a2, b2) and solving order
by order gives g; converting back to Newtonian form gives theta'' as a function of (theta,
theta'), and projecting the discrepancy onto the span of the three tunable operators
{sin theta, r - theta, theta'} gives the effective parameters.  Whatever falls outside that span
is a genuine obstruction to renormalizability.

Arm A is semi-implicit Euler with an explicit PD torque; arm C is the same map with
J -> J + h KD, which is what implicit damping amounts to.  RK4 (arm B) is not covered.

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


ARMS = (("A: semi-implicit Euler, explicit PD", (f, sp.Integer(0), sp.Integer(0), sp.Integer(0))),
        ("C: implicit damping (J -> J + h K_D)", (f, -KD*f/J, -KD*f/J, KD**2*f/J**2)))


def derive():
    for name, M in ARMS:
        first, second, chk1, chk2 = match(acceleration(*modified_field(*M)))
        print("=" * 76)
        print(name)
        for lab, s1, s2 in zip(("J", "K_P", "K_D"), first, second):
            print(f"  {lab}e = {lab} + h*({s1}) + h^2*({sp.simplify(s2)})")
        print(f"  consistency of the theta and r coefficients: {chk1[0]}, {chk2[0]}")
        print(f"  unrepresentable O(h^2) remainder: {chk2[1]}")


def numerical_check():
    """Each extra order should drop the one-step error by one power of h.

    The modified system's state is not the discrete state: theta matches, but its velocity is
    theta' = v + h a1 + h^2 a2, so the initial condition has to be transformed too.
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    from common import G as GRAV, J0, KD0, KP0, M, L

    Jv, KPv, KDv, cv, r0 = J0, KP0, KD0, M * GRAV * L / 2, 1.0
    tend, th0, v0 = 0.01, 0.3, 0.0
    acc = lambda t_, v_: (-cv * np.sin(t_) + KPv * (r0 - t_) - KDv * v_) / Jv
    d0f = lambda t_, v_: v_ * (-cv * np.cos(t_) - KPv) / Jv + acc(t_, v_) * (-KDv / Jv)

    def step_map(hh, implicit):
        t_, v_ = th0, v0
        for _ in range(int(round(tend / hh))):
            v_ += hh * (-cv * np.sin(t_) + KPv * (r0 - t_) - KDv * v_) / (
                Jv + hh * KDv if implicit else Jv)
            t_ += hh * v_
        return t_

    def flow(Je, KPe, KDe, thd0):
        rhs = lambda t, y: [y[1], (-cv * np.sin(y[0]) + KPe * (r0 - y[0]) - KDe * y[1]) / Je]
        return solve_ivp(rhs, (0, tend), [th0, thd0], rtol=1e-13, atol=1e-14,
                         method="DOP853").y[0, -1]

    for implicit in (False, True):
        print("=" * 76)
        print("C: implicit damping" if implicit else "A: semi-implicit Euler")
        sign = +1 if implicit else -1
        a1 = acc(th0, v0) / 2
        a2 = (-KDv * acc(th0, v0) / (2 * Jv) if implicit else 0.0) - d0f(th0, v0) / 6
        prev = None
        for hh in (5e-4, 2.5e-4, 1.25e-4, 6.25e-5):
            sim = step_map(hh, implicit)
            row = (sim - flow(Jv, KPv, KDv, v0),
                   sim - flow(Jv + sign * hh * KDv / 2, KPv, KDv, v0 + hh * a1),
                   sim - flow(Jv + sign * hh * KDv / 2 - hh**2 * (Jv * KPv + KDv**2) / (12 * Jv),
                              KPv, KDv - hh**2 * KPv * KDv / (12 * Jv),
                              v0 + hh * a1 + hh**2 * a2))
            o = ["   "] * 3 if prev is None else [f"{np.log2(abs(a / b)):4.1f}"
                                                  for a, b in zip(prev, row)]
            print(f"  h={hh*1e3:6.4f} ms  none {row[0]:+.2e} [{o[0]}]  first {row[1]:+.2e} "
                  f"[{o[1]}]  second {row[2]:+.2e} [{o[2]}]")
            prev = row
        print("  (bracket = measured order of convergence against the previous row)")


if __name__ == "__main__":
    derive()
    numerical_check()
