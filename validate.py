"""Self-contained validation of wind.py against the governing equations.

No external references (old solver, run libraries): each solution is checked
directly against what defines the BVP —

  1. ODE residual:  |dy/dx - rhs(x, y)|, with dy/dx from central differences
     of the dense output, normalized per-equation by the SUM of the magnitudes
     of its terms (a backward-error measure that stays meaningful where
     dw/dx crosses zero, unlike |rhs|-relative norms).
  2. Boundary conditions: endpoint states agree with the inner/outer series
     at the solved (q0, w1).
  3. Closure: continuity of (w, q) across the two-segment match point.

Grid: Gamma_0 x (m/m_max) spanning moderate to near-tiring-limit, plus one
finite-w_sg point. Exits nonzero if any check fails.
"""
import sys
import time
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from wind import (solve, rhs, inner_state, outer_state, m_max, ode_residual,
                  EPS_IN, EPS_OUT)

TOL_ODE = 1e-4      # max normalized ODE residual
TOL_BC = 1e-8       # endpoint state vs series
TOL_MATCH = 1e-7    # segment continuity at the match point


# ODE residual: shared implementation from wind (term-magnitude normalized;
# uses the collocation spline's analytic derivative for bvp-mode solutions —
# finite-differencing that interpolant manufactures fake ~1e-4 residuals).
# Sampled here at a different n than the in-solver gate so the two grids
# don't share artifact locations.
def _ode_residual(s):
    return ode_residual(s, n=2000)


def bc_error(s):
    """Endpoint states vs the boundary series at the solved (q0, w1)."""
    y_in = s.sol(EPS_IN)[:, 0]
    y_out = s.sol(1 - EPS_OUT)[:, 0]
    ref_in = inner_state(s.q0, s.m, s.Gamma_0, EPS_IN, s.w_sg)
    ref_out = outer_state(s.w1, s.m, s.Gamma_0, EPS_OUT)
    e_in = max(abs(y_in[0] - ref_in[0]), abs(y_in[1] - ref_in[1]))
    e_out = max(abs(y_out[0] - ref_out[0]), abs(y_out[1] - ref_out[1]))
    return max(e_in, e_out)


def match_error(s, h=1e-9):
    """Worst discontinuity of (w, q) across any segment junction.
    Collocation solutions have no junctions — their C1 dense output makes
    this check moot (returns 0)."""
    return max((float(np.max(np.abs(s.sol(xj - h) - s.sol(xj + h))))
                for xj in s.junctions), default=0.0)


def check(m, Gamma_0, w_sg=0.0):
    t0 = time.time()
    try:
        s = solve(m, Gamma_0, w_sg)
    except RuntimeError as e:
        return False, f"SOLVE FAILED — {e}"
    r_ode = _ode_residual(s)
    r_bc = bc_error(s)
    r_match = match_error(s)
    passed = r_ode < TOL_ODE and r_bc < TOL_BC and r_match < TOL_MATCH
    msg = (f"w1={s.w1:.4e}  ode={r_ode:.1e}  bc={r_bc:.1e}  "
           f"match={r_match:.1e}  ({time.time()-t0:.0f}s)")
    return passed, msg


def main():
    # Reach is m/m_max ~ 0.93-0.95 (Gamma_0-dependent); f = 0.9 is the
    # certified near-threshold level across the grid, with f = 0.94 probed at
    # Gamma_0 = 2 where the reach extends furthest. Beyond that, the profile
    # needs an asymptotic treatment (see wind.py docstring).
    grid_G = [1.5, 2.0, 5.0]
    grid_f = [0.3, 0.8, 0.9]
    failures = 0

    for G in grid_G:
        for f in grid_f:
            m = f*m_max(G)
            passed, msg = check(m, G)
            print(f"  G0={G:<4} f={f:<5} {'PASS' if passed else 'FAIL'}  {msg}",
                  flush=True)
            failures += 0 if passed else 1

    passed, msg = check(0.94*m_max(2.0), 2.0)
    print(f"  G0=2.0  f=0.94  {'PASS' if passed else 'FAIL'}  {msg}", flush=True)
    failures += 0 if passed else 1

    # finite base sonic energy (App. B extension)
    passed, msg = check(0.8*m_max(2.0), 2.0, w_sg=3e-5)
    print(f"  G0=2.0  f=0.8   w_sg=3e-5 {'PASS' if passed else 'FAIL'}  {msg}",
          flush=True)
    failures += 0 if passed else 1

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return failures


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
