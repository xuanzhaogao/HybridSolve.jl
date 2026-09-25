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

### Several spheres, charges and target points

`hybrid_solve` does the expensive work once; `eval_exterior_pot` can then be
called on any number of target sets. Targets are the columns of a `3 × ntrg`
matrix and must lie strictly outside every sphere.

```julia
using HybridSolve, Printf

# Two unit spheres along z with a 1a gap, eps_r = 2.5 and 4.0,
# and two point charges outside both spheres.
centers    = [0.0 0.0 0.0; 0.0 0.0 3.0]    # ns × 3, one row per sphere
radii      = [1.0, 1.0]
eps_r      = [2.5, 4.0]
charges    = [1.0, -0.5]
charge_pos = [0.0 1.8; 0.0 0.0; -2.0 1.5]  # 3 × nq, one column per charge

sol = hybrid_solve(centers, radii, eps_r, charges, charge_pos; p = 20, im = 8)

# 6 points on the x axis plus 3 others: a 3 × 9 target matrix.
xs = range(1.2, 4.0; length = 6)
line = vcat(xs', zeros(1, 6), zeros(1, 6))
extra = [0.0 1.3 -1.4; 0.0 0.0 0.0; -1.6 0.7 4.5]
targets = hcat(line, extra)

u = eval_exterior_pot(sol, targets)  # scattered potential, length 9

# eval_exterior_pot excludes the incident field; add it for the total.
inc(t) = sum(charges[k] / (4π * sqrt(sum(abs2, t .- charge_pos[:, k]))) for k in eachindex(charges))
φ = [u[j] + inc(targets[:, j]) for j in axes(targets, 2)]

@printf("%8s %8s %8s   %14s %14s\n", "x", "y", "z", "u_scattered", "phi_total")
for j in axes(targets, 2)
    @printf("%8.3f %8.3f %8.3f   %14.6e %14.6e\n", targets[:, j]..., u[j], φ[j])
end
@printf("energy = %.12e\n", electrostatic_energy(sol))
```

Output:

```
       x        y        z      u_scattered      phi_total
   1.200    0.000    0.000     2.581630e-03   1.207158e-02
   1.760    0.000    0.000     1.159963e-03   4.513507e-03
   2.320    0.000    0.000     6.856271e-04   1.602711e-03
   2.880    0.000    0.000     4.660341e-04   1.634745e-03
   3.440    0.000    0.000     3.417817e-04   2.437891e-03
   4.000    0.000    0.000     2.624769e-04   3.113567e-03
   0.000    0.000   -1.600    -4.471257e-03   1.833728e-01
   1.300    0.000    0.700     3.334705e-03  -1.228592e-02
  -1.400    0.000    4.500    -9.393742e-05   2.803240e-03
energy = -1.189160725743e-02
```

A target on or inside a sphere, or with a non-finite coordinate, raises an
`ArgumentError`; filter such points out first when evaluating on a grid.

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

- **Leave `sph_tol` at its default `Inf`.** Upstream skips the
  multipole-to-local translation for sphere pairs whose centres are more than
  `sph_tol` radii apart, dropping their mutual polarisation. HybridMD's own
  value of 4 gives a `3.6e-3` error for an 8-sphere cube at `1a` gaps (the
  face-diagonal pairs are `4.24a` apart); `sph_tol ≥ 6` or `Inf` agrees with
  converged `LaplaceMFS.jl` to `1e-13` there.

- **`p` limits tight sphere–sphere gaps; `im` limits charges close to a
  sphere surface.** Sphere–sphere polarisation lives in the per-sphere
  spherical-harmonic expansion, while the Kelvin/line images only represent
  each charge's response near the sphere it is imaged in. Measured at a
  `0.05a` sphere–sphere gap (the reference geometry below, max abs error vs
  converged `LaplaceMFS.jl`): `p = 20` gives `2.75e-9` at both `im = 8` and
  `im = 16` (raising `im` does nothing), while `p = 40, im = 8` gives
  `3.9e-13`. Conversely, for one unit sphere with a charge at `1.05a`
  (`p = 20`), the relative error vs the analytic series is `1.7e-5` at
  `im = 4`, `3.6e-12` at `im = 8` and `2.3e-15` at `im = 12`.
- Inputs are nondimensionalised (lengths by `maximum(radii)`, charges by
  `maximum(abs, charges)`) before HybridMD is called, because upstream GMRES
  stops on an **absolute** residual: `gmres_tol` is that absolute tolerance in
  normalised units, and results scale exactly (potential `∝ q/L`, energy
  `∝ q²/L`) with the inputs. Only the largest radius sets `L`, so a
  configuration with very disparate radii may still see a loose effective
  tolerance on the small spheres.
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
  — only point charges in a dielectric exterior driving uncharged dielectric
  spheres.
- **Not thread-safe.** HybridMD keeps all solver state in global variables, so
  concurrent calls would corrupt each other; every call into the library is
  serialised by a single `ReentrantLock` (`HybridSolve.LIB_LOCK`). Concurrent
  Julia tasks calling `hybrid_solve`/`eval_exterior_pot` block on this lock
  rather than racing.
- **Each call leaks memory.** Upstream reallocates its per-solve global arrays
  on every call and never frees the previous ones (this package's patch 3
  removes the largest such leak, an unused `O(M²·p⁴)` initialization block).
  The residual per-call allocation is dominated by the per-sphere-pair
  `mpole` and `local` arrays, `ns²·(2p+1)·(p+1)` complex entries each, plus
  `O(p³)` for the `ynm` tables, `ns·p²` for the target/field grids and
  `O(ns·N·im)` image-source buffers — measured at about 15 MB per call at
  `ns = 8`, `p = 40`. A
  long-running process issuing many solves will grow its memory footprint
  roughly linearly in the number of solves; restart the process periodically
  if you need many solves at large `p`/`ns`.

See `docs/plans/2026-09-25-hybridsolve-design.md` for the full design and
`deps/patches/README.md` for exactly what was changed relative to upstream
HybridMD and why.
