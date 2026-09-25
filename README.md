# HybridSolve.jl

An independent reference solver for the multi-sphere dielectric transmission
problem, callable from Julia. It exists to verify
[`LaplaceMFS.jl`](https://github.com/xuanzhaogao/LaplaceMFS.jl) on
configurations that have no analytic answer (closely spaced spheres, several
spheres, etc.).

`HybridSolve.jl` wraps the multi-sphere solver of **HybridMD** (Gan, Jiang,
Luijten & Xu; upstream commit `1f87baf`): Kelvin + line image charges, a
spherical-harmonic method of moments per sphere, GMRES, and an FMM. The
upstream code is vendored nearly verbatim under `deps/upstream/`, compiled
into a shared library by `deps/build.jl`, and called through `ccall`. Because
it vendors HybridMD, this package is licensed **GPL-3.0** (see `LICENSE`);
`LaplaceMFS.jl` itself has no such requirement.

## Install

Requires a C/C++/Fortran toolchain: `gcc`, `g++`, `gfortran`, and `patch` on
`PATH` (all part of a typical Linux dev environment; on Ubuntu/Debian,
`sudo apt-get install -y build-essential gfortran patch`).

```julia
using Pkg
Pkg.develop(path = "/path/to/HybridSolve.jl")
Pkg.build("HybridSolve")
```

`Pkg.build` compiles `deps/upstream` (after applying the patches in
`deps/patches/`) plus the C shim in `deps/shim/` into
`deps/usr/lib/libhybridsolve.so`. Nothing is downloaded; there is no
BinaryBuilder/JLL packaging (see "Out of scope" in the design doc).

## Quick start

```julia
using HybridSolve

a, eps_r, q = 1.0, 2.5, 1.0
center = zeros(1, 3)                          # ns × 3
charge_pos = reshape([0.0, 0.0, 3.0], 3, 1)    # 3 × nq
targets = reshape([0.0, 0.0, 2.0], 3, 1)       # 3 × ntrg

sol = hybrid_solve(center, [a], [eps_r], [q], charge_pos; p = 20, im = 12)
u = eval_exterior_pot(sol, targets)

ref = single_sphere_pointcharge_exterior(targets, vec(center), a, eps_r, q, vec(charge_pos))
@show u[1] ref[1]
```

This is the same configuration exercised in `test/single_sphere.jl`, checked
there against the exact Legendre series to within `1e-9` relative error.

## Conventions (identical to `LaplaceMFS.jl`)

- `centers` is `ns × 3`; `charge_pos` and `targets` are `3 × n`. Per-sphere
  `radii` and relative permittivity `eps_r`; the exterior medium always has
  permittivity 1.
- Units: `1/(4π r)` kernel (`φ_HybridSolve = φ_HybridMD / (4π)`, with the
  upstream free-space permittivity fixed at 1).
- `eval_exterior_pot(sol, targets)` returns the **scattered** (reaction)
  potential only — the incident free-charge potential is excluded, exactly as
  in `LaplaceMFS.jl`. Both packages export a function with this same name;
  qualify with `HybridSolve.eval_exterior_pot` /
  `LaplaceMFS.eval_exterior_pot` if both are loaded in the same session.
- `electrostatic_energy(sol)` is the total electrostatic energy: the free-pair
  Coulomb energy of the point charges plus `½ Σ q_k u_scat(X_k)`.
- Real `Float64` only. Charges and targets must be strictly outside every
  sphere; spheres must not overlap or touch.

## Accuracy knobs

`hybrid_solve` takes `p` (spherical-harmonic truncation per sphere) and `im`
(number of Kelvin/line images per source-sphere pair) as its main accuracy
knobs, plus `gmres_tol`, `fmm_iprec`, `source_tol`, `sph_tol` passed straight
through to HybridMD.

- **`im` is the limiting parameter** in practice: image expansions converge
  geometrically, but slowly at small sphere–sphere or charge–sphere gaps.
  Raising `p` past what `im` can resolve does not help.
- The precision table shipped with HybridMD (mapping `fmm_iprec` to a decimal
  digit count) is not used to size `p`/`im`/tolerances here; treat it as
  informational only and validate against `single_sphere_pointcharge_exterior`
  or a known configuration.
- Measured on this package's own tests: a single sphere vs. the analytic
  series agrees to about `1e-11` (`p = 20`, `im = 12`). Two spheres at a `1a`
  gap agree with converged `LaplaceMFS.jl` values to `4.3e-13` (`p = 20`,
  `im = 8`); at a much tighter `0.05a` gap, agreement is `3.9e-13` but only
  with **`p ≥ 40`** (`im = 8`).

## Limitations

- **No forces.** The upstream force-computation path has known bugs
  (`deps/patches/README.md`: a stride-4 indexing bug and a missing `√(2n+1)`
  factor) and is disabled (`force_compute = 0`); it is out of scope to fix.
- **No uniform-field excitation**, no charged spheres, no complex permittivity
  — only point charges in a dielectric exterior driving grounded/floating
  dielectric spheres.
- **Not thread-safe.** HybridMD keeps all solver state in global variables, so
  concurrent calls would corrupt each other; every call into the library is
  serialised by a single `ReentrantLock` (`HybridSolve.LIB_LOCK`). Concurrent
  Julia tasks calling `hybrid_solve`/`eval_exterior_pot` block on this lock
  rather than racing.
- **Each call leaks memory.** Upstream reallocates its per-solve global arrays
  on every call and never frees the previous ones (this package's patch 3
  removes the largest such leak, an unused `O(M²·p⁴)` initialization block,
  but the residual per-call allocation of size `O(ns·p² + N²)` remains). A
  long-running process issuing many solves will grow its memory footprint
  roughly linearly in the number of solves; restart the process periodically
  if you need many solves at large `p`/`ns`.

See `docs/plans/2026-09-25-hybridsolve-design.md` for the full design and
`deps/patches/README.md` for exactly what was changed relative to upstream
HybridMD and why.
