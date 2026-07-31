# tiring_solver

> **Note:** This code and documentation were written by an AI assistant
> (Claude Opus 4.8) in collaboration with N. Kaaz. The solver
> has been verified against the governing equations via `validate.py`, but
> review the implementation before relying on it for published results.

Solves the Owocki, Townsend & Quataert (2017) super-Eddington wind equations
(their Appendix A), including mass-loss rates approaching the photon-tiring
limit.

## The problem

The steady, spherically symmetric, radiation-driven wind is described by two
coupled ODEs for the scaled wind kinetic energy `w(x) = v²/v_esc²` and the
scaled radiation pressure `q(x)` on the domain `x = 1 − R/r ∈ (0, 1)`:

    dw/dx = Γ₀ [1 − m(w + x) − 4q√w] − 1
    dq/dx = −(mΓ₀/√w) [1 − m(w + x)] + 2q/(1−x) + 4mΓ₀ q

with boundary conditions `w(0) = w_sg` (base sonic energy; 0 in the paper's
idealized limit) and `q → 0` as `x → 1`. Here `Γ₀ = L₀/L_Edd` and
`m = Ṁ/Ṁ_max` is the photon-tiring number. (The paper's eq. A1 prints
`+4q√w`; consistency with their variable change requires the minus sign used
here.)

A wind that reaches infinity exists only for `m < m_max = 1 − 1/Γ₀`: the
luminosity spent lifting the wind out of the potential well must leave at
least the Eddington luminosity available at large radius. As `m → m_max` the
terminal energy `w₁ ≡ w(1) → 0`.

## Method

Both endpoints of the domain are singular, so integrations start a small
distance inside them using series expansions (paper eqs. A7–A8 for the outer
boundary; a Frobenius expansion, or a Taylor expansion when `w_sg > 0`, for
the inner one). Two regimes:

- **`m/m_max ≲ 0.8` — matched shooting.** Integrate from both boundaries to
  `x = 0.5` and root-find on the two unknowns `(q₀, w₁)` until the two
  half-solutions agree there. (`solve_at_m`)

- **Larger `m` — collocation with `w₁` prescribed.** As `w₁ → 0`,
  root-finding on `w₁` becomes ill-conditioned (the outer boundary series
  has `q ∝ 1/√w₁`), and shooting methods fail outright because the solution
  becomes extremely sensitive to the starting values. Instead, the roles are
  swapped: fix `w₁`, and solve for the profile *and the value of `m` that
  admits it*, using `scipy.integrate.solve_bvp` with `m` as an unknown
  parameter. This is done in the stretched coordinate `ξ = −log(1−x)`, which
  removes the `1/(1−x)` term and resolves the outer region where the
  profile varies rapidly. Since `m(w₁)` is smooth and monotonic, a target
  `m` is reached by stepping `w₁` downward geometrically (each solution
  warm-starting the next) and then refining with a bracketed secant
  iteration on `log w₁`. (`solve_at_w1`)

Every candidate solution is verified against the governing equations before
being accepted: the residual of both ODEs is evaluated on a fine grid and
normalized by the sum of the magnitudes of the terms in each equation
(`ode_residual`). Genuine solutions score ~1e-8 to 1e-6 on this measure;
false solutions — which the root-finders can and do produce — score orders
of magnitude higher and are rejected.

Two implementation details matter for reliability:

1. Each collocation solve is started from the solution at a nearby `w₁`
   (its profile serves as the initial guess). This works well when the new
   `w₁` is *smaller* than the previous one, but starting from a solution
   with smaller `w₁` and solving for a larger one tends to produce
   inaccurate results that `solve_bvp` nevertheless reports as converged
   (the equation-residual check above catches them). The code therefore
   always approaches a target `w₁` from above.

2. The initial guess also supplies the initial mesh. `solve_bvp` adds mesh
   points where needed but never removes them, so passing each solution's
   mesh directly to the next solve makes the mesh grow without bound over a
   long sequence of solves. The code removes a fraction of the points from
   large meshes before reusing them.

## Range of validity

The solver works up to `m/m_max ≈ 0.94–0.95` (weakly `Γ₀`-dependent). Beyond
that, the required mesh resolution grows without bound: the profile hugs the
marginal state where the photon-tired luminosity exactly balances gravity,
and empirically `m_max − m ∝ √w₁`, so the remaining approach to the limit
compresses into an ever-harder numerical corner. Reaching `m/m_max → 1`
would require a matched asymptotic expansion about the tiring limit rather
than further mesh refinement; `solve()` raises a clear error beyond its
reach.

## Usage

```python
from wind import solve

s = solve(m=0.45, Gamma_0=2.0)   # m/m_max = 0.90
s.w1                             # terminal wind energy w(1) = v_∞²/v_esc²
s.q0                             # base radiation pressure q(0)
s.sol(x)                         # (w, q) profile at any x in (0, 1)
s.m, s.Gamma_0, s.w_sg           # parameters of this solution
```

`w_sg > 0` sets a finite base sonic energy (paper's Appendix B extension).
The solver is purely dimensionless: mapping a physical system
(M, R, Ṁ, κ, L₀) onto `(m, Γ₀, w_sg)` is the caller's responsibility.

## Validation

`python validate.py` solves a grid over `Γ₀ ∈ {1.5, 2, 5}` and
`m/m_max ∈ {0.3, 0.8, 0.9}` (plus `0.94` at `Γ₀ = 2`, near the edge of the
solver's range, and one case with finite `w_sg`) and checks every solution
directly against the governing equations:

- ODE residual (term-normalized, as above) below 1e-4,
- endpoint values consistent with the boundary series to 1e-8,
- continuity across the matching point of shooting solutions to 1e-7.

It is self-contained — no reference solutions required — and exits nonzero
on any failure.
