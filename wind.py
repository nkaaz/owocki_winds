"""Minimal, robust solver for the Owocki, Townsend & Quataert (2017)
photon-tired wind BVP, valid arbitrarily close to the tiring limit
m -> m_max = 1 - 1/Gamma_0.

Equations (paper App. A, with the eq A1 sign typo fixed):
    dw/dx = Gamma_0 [1 - m(w+x) - 4 q sqrt(w)] - 1
    dq/dx = -(m Gamma_0 / sqrt(w)) [1 - m(w+x)] + 2q/(1-x) + 4 m Gamma_0 q
with q = p/(1-x)^2, BCs w(0) = w_sg, q(1) = 0 (via the outer series).

Method — one idea per regime:
  * moderate m: direct two-segment matched shoot, unknowns (q0, w1).
  * near threshold: w1 -> 0 makes (q0, w1) root-finding singular, so swap
    roles — prescribe w1 and let m float: collocation (solve_bvp) with m as
    an unknown parameter, in the stretched coordinate xi = -log(1-x), walked
    geometrically downward in w1 from a moderate-m anchor. m(w1) is smooth
    and monotone; the target m is located by Illinois secant on log w1.

Every candidate is validated against the global ODE residual (rejects
spurious matched-only roots — these do occur).

KNOWN LIMIT: collocation node demand grows ~3x per w1 step near the
threshold (empirically m_max - m ~ sqrt(w1), and the profile becomes
uniformly harder to resolve as it hugs the marginal Gamma*(tired L) = 1
state). Practical reach is m/m_max ~ 0.93-0.95, mildly Gamma_0-dependent
(~0.93 at Gamma_0 = 1.5; ~0.95 at Gamma_0 = 2). solve() raises past its
reach; closing the last few percent to the tiring limit needs an asymptotic
(matched-expansion) treatment, not more mesh.

Public API:
    solve(m, Gamma_0, w_sg=0.0, verbose=False) -> Solution
    solve_at_w1(w1, Gamma_0, w_sg, guess)      -> Solution (m is an output)
    m_max(Gamma_0), rhs, inner_state, outer_state
"""
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import root

EPS_IN, EPS_OUT, X_MATCH = 1e-7, 1e-6, 0.5


def m_max(Gamma_0):
    return 1.0 - 1.0/Gamma_0


def rhs(x, y, m, Gamma_0):
    w = np.maximum(y[0], 1e-30)
    q = y[1]
    a = 1.0 - m*(w + x)
    dw = Gamma_0*(a - 4*q*np.sqrt(w)) - 1
    dq = -(m*Gamma_0/np.sqrt(w))*a + 2*q/np.maximum(1.0 - x, 1e-14) + 4*m*Gamma_0*q
    return np.stack([dw, dq])


def inner_state(q0, m, Gamma_0, eps, w_sg=0.0):
    """Series state at x = eps (Frobenius for w_sg = 0, Taylor otherwise)."""
    if w_sg < 1e-10:
        sg = np.sqrt(Gamma_0 - 1)
        return ((Gamma_0 - 1)*eps - (8/3)*Gamma_0*q0*sg*eps**1.5,
                q0 - 2*m*Gamma_0/sg*np.sqrt(eps))
    sw = np.sqrt(w_sg)
    dw = Gamma_0*(1 - m*w_sg - 4*q0*sw) - 1
    dq = -(m*Gamma_0/sw)*(1 - m*w_sg) + (2 + 4*m*Gamma_0)*q0
    return (w_sg + dw*eps, q0 + dq*eps)


def outer_state(w1, m, Gamma_0, eps):
    """Series state at x = 1 - eps (paper eqs A7, A8)."""
    return (w1 + (1 + Gamma_0*(m*w1 + m - 1))*eps,
            (m*Gamma_0/(3*np.sqrt(w1)))*(1 - m*(w1 + 1))*eps)


def _segment(rhs_fn, xa, xb, y0):
    return solve_ivp(rhs_fn, [xa, xb], y0, method='Radau',
                     atol=1e-12, rtol=1e-10, max_step=0.05, dense_output=True)


class Solution:
    """BVP solution: two matched shooting segments ('matched' mode) or a
    collocation result in xi = -log(1-x) ('bvp_xi' mode).
    Exposes .sol(x), .t, .y, .w1, .q0, .m, .junctions."""

    def __init__(self, *, m, Gamma_0, w_sg, w1, q0, sol_in=None, sol_out=None,
                 x_match=X_MATCH, bvp_xi=None):
        self.m, self.Gamma_0, self.w_sg = m, Gamma_0, w_sg
        self.w1, self.q0 = w1, q0
        if bvp_xi is not None:
            self._bvp_xi = bvp_xi
            self._in = self._out = None
            self.junctions = []
            self.x_match = None
            self.t = 1.0 - np.exp(-bvp_xi.x)
            self.y = bvp_xi.y.copy()
            self.success = bool(bvp_xi.success)
        else:
            self._bvp_xi = None
            self._in, self._out, self.x_match = sol_in, sol_out, x_match
            self.junctions = [x_match]
            t = np.concatenate([sol_in.t, sol_out.t])
            y = np.hstack([sol_in.y, sol_out.y])
            order = np.argsort(t)
            self.t = t[order]
            self.y = y[:, order]
            self.success = bool(sol_in.success and sol_out.success)

    def sol(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        xc = np.clip(x, EPS_IN, 1 - EPS_OUT)
        if self._bvp_xi is not None:
            return self._bvp_xi.sol(-np.log(1.0 - xc))
        out = np.empty((2, x.size))
        inner = xc <= self.x_match
        if inner.any():
            out[:, inner] = self._in.sol(xc[inner])
        if (~inner).any():
            out[:, ~inner] = self._out.sol(xc[~inner])
        return out


def _pair(q0, w1, m, Gamma_0, w_sg):
    rhs_fn = lambda x, y: rhs(x, y, m, Gamma_0)
    sA = _segment(rhs_fn, EPS_IN, X_MATCH, inner_state(q0, m, Gamma_0, EPS_IN, w_sg))
    sB = _segment(rhs_fn, 1 - EPS_OUT, X_MATCH, outer_state(w1, m, Gamma_0, EPS_OUT))
    return sA, sB


def _mismatch(sA, sB):
    if not (sA.success and sB.success):
        return None
    return sA.y[:, -1] - sB.y[:, -1]


def ode_residual(sol, n=1500, h=1e-7):
    """Max term-normalized ODE residual of a Solution on a fine grid.

    Normalized per-equation by the SUM of term magnitudes. (|rhs|-relative
    norms false-alarm wherever a derivative crosses zero — near-threshold
    profiles have a flat w peak, and that false alarm is what actually broke
    the old solver at m/m_max ~ 0.9.) The grid stays in [1e-4, 1-1e-4]: the
    sqrt(x) Frobenius behavior at the inner edge wrecks the check, and the
    endpoint zones are the boundary series' responsibility anyway.

    Each solution mode is checked in the coordinates its solver controlled.
    Matched mode: central differences of the Radau dense output, x-space
    equations. Collocation mode: the spline's analytic derivative against the
    xi-space equations — the same ODEs, but (a) finite-differencing a
    tol~1e-6 interpolant manufactures 1/h-amplified fake residuals, and
    (b) converting the controlled xi-residual to x-space multiplies it by
    1/(1-x) (up to 1e4 at the outer edge) while the w-equation's term scale
    stays O(1), false-rejecting perfectly good solutions."""
    G = sol.Gamma_0
    m = sol.m
    if sol._bvp_xi is not None:
        xi = np.linspace(max(-np.log(1.0 - max(sol.t.min(), 1e-4)), 1e-4),
                         -np.log(1e-4), n)              # x in [1e-4, 1-1e-4]
        d = sol._bvp_xi.sol(xi, 1)
        y = sol._bvp_xi.sol(xi)
        r = _rhs_xi(xi, y, m, G)
        omx = np.exp(-xi)
        x = 1.0 - omx
        w = np.maximum(y[0], 1e-30)
        q = y[1]
        a = 1.0 - m*(w + x)
        scale_w = (G*(np.abs(a) + np.abs(4*q*np.sqrt(w))) + 1.0)*omx + 1e-30
        scale_q = ((np.abs(m*G/np.sqrt(w)*a) + np.abs(4*m*G*q))*omx
                   + np.abs(2*q) + 1e-30)
    else:
        x = np.linspace(max(sol.t.min() + 10*h, 1e-4),
                        min(sol.t.max() - 10*h, 1 - 1e-4), n)
        for xj in sol.junctions:
            x = x[np.abs(x - xj) > 10*h]
        d = (sol.sol(x + h) - sol.sol(x - h))/(2*h)
        y = sol.sol(x)
        r = rhs(x, y, m, G)
        w = np.maximum(y[0], 1e-30)
        q = y[1]
        a = 1.0 - m*(w + x)
        scale_w = G*(np.abs(a) + np.abs(4*q*np.sqrt(w))) + 1.0
        scale_q = (np.abs(m*G/np.sqrt(w)*a) + np.abs(2*q/(1 - x))
                   + np.abs(4*m*G*q) + 1e-30)
    return max(float(np.max(np.abs(d[0] - r[0])/scale_w)),
               float(np.max(np.abs(d[1] - r[1])/scale_q)))


def residual_ok(sol, tol=1e-5):
    """Gate on the global ODE residual; rejects spurious matched-only roots.
    True solutions score ~1e-8 (matched) / ~1e-6 (collocation)."""
    return ode_residual(sol) < tol


def _build(q0, w1, m, Gamma_0, w_sg):
    sA, sB = _pair(q0, w1, m, Gamma_0, w_sg)
    s = Solution(m=m, Gamma_0=Gamma_0, w_sg=w_sg, w1=w1, q0=q0,
                 sol_in=sA, sol_out=sB)
    return s if (s.success and residual_ok(s)) else None


def solve_at_m(m, Gamma_0, w_sg=0.0, guess=None):
    """Direct matched shoot, unknowns (q0, w1). Reliable for m/m_max <~ 0.8."""
    bridge_w1 = max(Gamma_0/(1 + m*Gamma_0) - 1, 1e-4)
    starts = ([guess] if guess is not None else []) + [
        (0.3*m*Gamma_0, bridge_w1), (m*Gamma_0, bridge_w1), (1.0, 1e-2)]

    def f(p):
        q0, w1 = p
        if q0 < -1e-3 or w1 <= 1e-12:
            return [1e3, 1e3]
        r = _mismatch(*_pair(q0, w1, m, Gamma_0, w_sg))
        return list(r) if r is not None else [1e3, 1e3]

    for g in starts:
        s = root(f, g, method='hybr', options={'xtol': 1e-12, 'maxfev': 100})
        if s.success and s.x[1] > 0 and np.linalg.norm(s.fun) < 1e-6:
            cand = _build(s.x[0], s.x[1], m, Gamma_0, w_sg)
            if cand is not None:
                return cand
    return None


XI_MAX = -np.log(EPS_OUT)      # stretched coordinate xi = -log(1-x)


def _rhs_xi(xi, y, m, Gamma_0):
    """ODEs in xi = -log(1-x). Regularizes 2q/(1-x) -> 2q and log-stretches
    the outer boundary layer that forms as w1 -> 0."""
    omx = np.exp(-xi)                     # = 1 - x
    x = 1.0 - omx
    w = np.maximum(y[0], 1e-30)
    q = y[1]
    a = 1.0 - m*(w + x)
    dw = (Gamma_0*(a - 4*q*np.sqrt(w)) - 1)*omx
    dq = (-(m*Gamma_0/np.sqrt(w))*a + 4*m*Gamma_0*q)*omx + 2*q
    return np.stack([dw, dq])


def solve_at_w1(w1, Gamma_0, w_sg=0.0, guess=None, tol=1e-6, max_nodes=60000):
    """Prescribed-w1 collocation with m as an unknown parameter.

    Regular through w1 -> 0 (no root-find touches the singular q ~ 1/sqrt(w1)
    direction), and collocation's global linearization avoids the shared-m
    conditioning wall that stalls shooting-type methods here. Three BCs close
    the 2-ODE + 1-parameter system: w at the base plus the (w, q) outer series
    at x = 1 - EPS_OUT, both slaved to the floating m.

    `guess`: a nearby Solution (mesh + profile + m warm start) — required;
    this is a continuation workhorse, not a cold-start method."""
    if guess is None:
        return None
    inner_w = w_sg if w_sg > 1e-10 else (Gamma_0 - 1)*EPS_IN

    def bc(ya, yb, p):
        wb, qb = outer_state(w1, p[0], Gamma_0, EPS_OUT)
        return np.array([ya[0] - inner_w, yb[0] - wb, yb[1] - qb])

    # Warm-start mesh: reuse the guess's own xi-mesh when it has one, else a
    # uniform xi grid through the guess profile. Thin inherited meshes:
    # solve_bvp only ever refines, so without this the mesh bloats
    # monotonically along a continuation until every start exceeds max_nodes.
    if getattr(guess, '_bvp_xi', None) is not None:
        xi = guess._bvp_xi.x
        y0 = guess._bvp_xi.y
        if xi.size > 6000:
            keep = np.unique(np.r_[0, np.arange(0, xi.size,
                                                xi.size // 4000), xi.size - 1])
            xi = xi[keep]
            y0 = y0[:, keep]
        xi, y0 = xi.copy(), y0.copy()
    else:
        xi = np.linspace(EPS_IN, XI_MAX, 600)
        y0 = guess.sol(1.0 - np.exp(-xi))
    # First-order profile shift: the outer profile rides down with w1.
    x_mesh = 1.0 - np.exp(-xi)
    y0 = y0.copy()
    y0[0] = np.maximum(y0[0] + (w1 - guess.w1)*np.clip((x_mesh - 0.5)/0.5, 0, 1)**2,
                       0.3*w1)

    fun = lambda xi_, y_, p: _rhs_xi(xi_, y_, p[0], Gamma_0)
    b = solve_bvp(fun, bc, xi, y0, p=[guess.m], tol=tol,
                  max_nodes=max_nodes, verbose=0)
    if not b.success:
        return None

    def wrap(b_):
        # Recover q0 = q(x=0), inverting the inner series from q(EPS_IN) so
        # the stored q0 means the same thing in both solution modes.
        m_out = float(b_.p[0])
        q_eps = float(b_.y[1, 0])
        if w_sg < 1e-10:
            q0 = q_eps + 2*m_out*Gamma_0/np.sqrt(Gamma_0 - 1)*np.sqrt(EPS_IN)
        else:
            sw = np.sqrt(w_sg)
            q0 = ((q_eps + (m_out*Gamma_0/sw)*(1 - m_out*w_sg)*EPS_IN)
                  / (1 + (2 + 4*m_out*Gamma_0)*EPS_IN))
        return Solution(m=m_out, Gamma_0=Gamma_0, w_sg=w_sg, w1=w1,
                        q0=q0, bvp_xi=b_)

    cand = wrap(b)
    if residual_ok(cand):
        return cand
    # solve_bvp's residual estimate can accept a marginally sloppy solution
    # (seen on w1-increasing warm starts); one pass at much tighter tol from
    # the candidate's own mesh forces genuine refinement.
    b = solve_bvp(fun, bc, b.x, b.y, p=[float(b.p[0])], tol=tol*1e-2,
                  max_nodes=max_nodes, verbose=0)
    if not b.success:
        return None
    cand = wrap(b)
    return cand if residual_ok(cand) else None


def solve(m, Gamma_0, w_sg=0.0, verbose=False):
    """Solve the BVP at (m, Gamma_0, w_sg), any m up to the tiring limit."""
    mm = m_max(Gamma_0)
    if m >= mm:
        raise ValueError(f"m = {m} >= m_max = {mm:.6f}: no escaping wind exists.")

    s = solve_at_m(m, Gamma_0, w_sg)
    if s is not None:
        if verbose:
            print(f"  direct: m={m:.6f}, w1={s.w1:.4e}")
        return s

    # Near-threshold: continuation downward in w1, m as output.
    s0 = None
    for frac in (0.75, 0.6, 0.45):
        m0 = frac*mm
        if m0 < m:
            s0 = solve_at_m(m0, Gamma_0, w_sg)
            if s0 is not None:
                break
    if s0 is None:
        raise RuntimeError(f"direct solve failed even at moderate m (Gamma_0={Gamma_0}).")

    pts = [s0]                       # samples of the m(w1) curve, w1 decreasing
    shrink = 0.55
    for _ in range(200):
        cur = pts[-1]
        if cur.m >= m:
            break
        w1_next = cur.w1*shrink
        if w1_next < 1e-13:
            raise RuntimeError(
                f"w1-continuation reached w1 < 1e-13 with m = {cur.m:.8f} < {m}: "
                f"target is numerically at the tiring limit.")
        s_next = solve_at_w1(w1_next, Gamma_0, w_sg, guess=cur)
        if s_next is None:
            shrink = np.sqrt(shrink)         # smaller step, retry
            if shrink > 0.97:
                raise RuntimeError(
                    f"w1-continuation stalled at w1={cur.w1:.3e}, "
                    f"m={cur.m:.6f} (m/m_max={cur.m/mm:.4f}) short of target "
                    f"m={m:.6f} (m/m_max={m/mm:.4f}). This solver's practical "
                    f"reach is m/m_max ~ 0.93-0.95; closer to the tiring limit "
                    f"needs an asymptotic treatment (see module docstring).")
            continue
        pts.append(s_next)
        if verbose:
            print(f"  cont: w1={s_next.w1:.4e} -> m={s_next.m:.8f}")
    else:
        raise RuntimeError("w1-continuation did not bracket the target m.")

    # Bracketed secant (Illinois) on u = log w1 for m(u) = m; each evaluation
    # is a warm-started solve, so exit on |m - target| rather than on u.
    tol_m = 1e-9*max(m, 1e-9)
    if abs(pts[-1].m - m) < tol_m:
        return pts[-1]
    ua, fa = np.log(pts[-2].w1), pts[-2].m - m     # fa < 0
    ub, fb = np.log(pts[-1].w1), pts[-1].m - m     # fb > 0
    best = pts[-1]
    for _ in range(30):
        u = ub - fb*(ub - ua)/(fb - fa)
        if not (min(ua, ub) < u < max(ua, ub)):
            u = 0.5*(ua + ub)
        # Warm-start only from ABOVE in w1: downward steps are the proven
        # continuation direction — w1-increasing warm starts empirically
        # converge to marginally sloppy solutions the residual gate rejects.
        def nearest_above(u_):
            above = [p for p in pts if np.log(p.w1) > u_ + 1e-14]
            return min(above, key=lambda p: np.log(p.w1) - u_) if above else None
        near = nearest_above(u)
        s_ = solve_at_w1(np.exp(u), Gamma_0, w_sg, guess=near) if near else None
        # If the jump from the warm start is too large, walk u toward the
        # nearest converged point until a step succeeds.
        retries = 0
        while s_ is None and near is not None and retries < 6:
            u = 0.5*(u + np.log(near.w1))
            if abs(u - np.log(near.w1)) < 1e-12:
                break
            s_ = solve_at_w1(np.exp(u), Gamma_0, w_sg, guess=near)
            retries += 1
        if s_ is None:
            raise RuntimeError(
                f"prescribed-w1 solve failed near w1={np.exp(u):.3e} during root polish.")
        pts.append(s_)
        f = s_.m - m
        if abs(f) < abs(best.m - m):
            best = s_
        if abs(f) < tol_m:
            break
        if f > 0:
            ub, fb = u, f
        else:
            ua, fa = u, f
            fb *= 0.5                              # Illinois anti-stall
    if verbose:
        print(f"  final: w1={best.w1:.6e}, m={best.m:.10f} (target {m})")
    if abs(best.m - m) > 1e-8*max(m, 1e-8):
        raise RuntimeError(f"converged m={best.m} misses target {m}.")
    return best
