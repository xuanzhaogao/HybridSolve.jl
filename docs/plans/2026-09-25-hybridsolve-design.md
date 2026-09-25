# HybridSolve.jl — Design

**Date:** 2026-09-25
**Status:** approved in conversation, 2026-09-25

## Purpose

An independent reference solver for the dielectric-sphere transmission problem,
callable from Julia, used to verify `LaplaceMFS.jl` on configurations with no
analytic answer (verification ladder rungs 2–4 in
`LaplaceMFS.jl/docs/plans/2026-09-06-mfs-verification-findings.md`).

It wraps the multi-sphere solver of `HybridMD` (Gan, Jiang, Luijten & Xu;
upstream commit `1f87baf`): Kelvin + line image charges, a spherical-harmonic
method of moments per sphere, GMRES, FMM. The upstream code is vendored nearly
verbatim, compiled into a shared library, and called through `ccall`.

**Success:** the package reproduces the exact single-sphere Legendre series,
and reproduces the converged-MFS values recorded in §3.7 of the findings doc at
the `1a` and `0.05a` sphere–sphere gaps to about 1e-10.

## Conventions (identical to LaplaceMFS.jl)

- `centers` is `ns × 3`; `charge_pos` and `targets` are `3 × n`.
- Per-sphere `radii` and relative permittivity `eps_r`; the exterior medium has
  permittivity 1.
- Units: `1/(4π r)` kernel. `φ_HybridSolve = φ_HybridMD / (4π)` (with `ε_s = 1`).
- `eval_exterior_pot` returns the **scattered** (reaction) potential only —
  the incident free-charge potential is excluded, exactly as LaplaceMFS's
  `eval_exterior_pot`.
- Real `Float64` only.

## Architecture

```
HybridSolve.jl/
  Project.toml, LICENSE (GPL-3.0), README.md, AGENTS.md, .gitignore
  .github/workflows/CI.yml
  src/HybridSolve.jl      module, exports, includes
  src/libhybrid.jl        library path, ccall wrappers, global lock
  src/solve.jl            HybridSolution, hybrid_solve, input validation
  src/evaluation.jl       eval_exterior_pot, electrostatic_energy
  src/analytic.jl         exact single-sphere point-charge Legendre series
  deps/upstream/          HybridMD@1f87baf solver files, byte-identical
  deps/upstream/SHA256SUMS
  deps/patches/           0001-…0003-*.patch + README.md
  deps/shim/hybridsolve_capi.cpp   thin C ABI (+ copied allocate_dynamic)
  deps/build.jl           copy upstream → apply patches → compile → deps/usr/lib/libhybridsolve.so
  test/                   runtests.jl + focused files
  docs/plans/             this spec and the implementation plan
```

### Vendored upstream (solver only)

`Coulomb_accelerations_Hybrid.cpp`, `MDpara.cpp`, `MDpara.h`, `ran.h`,
`allocate.cpp`, `output_force.cpp`, `header/itlin.h`, `GMRES/{gmres.c,utils.c}`,
`JacobiGaussQuad/jacobi_rule.cpp`, the `FMM3dlib/*.f` and `sht/*.f` files named in
upstream `Makefile.gnu`'s `LIBOBJECTS`, plus upstream `LICENSE`. No MD code
(Verlet, Langevin, LJ, cell list, RDF, `main.cpp`). `SHA256SUMS` records every
vendored file; a test checks them.

### Upstream patches (complete list)

1. `0001-fix-cjk-comma.patch` — `Coulomb_accelerations_Hybrid.cpp:1227`: CJK
   ideographic comma U+3001 → `\`. Hard compile error upstream.
2. `0002-per-sphere-epsilon.patch` — in the `iter_indicator==0` block that sets
   dielectric constants, replace the two-species `ei1/ei2` assignment with
   `epsi_i[i] = hs_eps_in[i]`, reading a new global `double *hs_eps_in` defined
   by the shim. `epsi_s = epsi_ion` is kept.
3. `0003-skip-dead-initialization.patch` — do not call `initialization()`.
   Every array it allocates (`tempcoef`, `tempcoefimag` of size
   `M²·imm·(p+1)³`, `powerd`, `newpowerd`, `rotangle`) is never read; the only
   readers are `generate_coeff`/`generate_power`, which are never called. At
   27 spheres, `p = 40` it is ~3 GB per call.
The image-count guard's `exit(0)` is left alone: `allocate_dynamic` sizes the
buffer as `imnum_thresh = N_col·N·im`, the maximum possible image count, so the
guard is unreachable. `deps/patches/README.md` records this.

No numerical code is changed. The known force-path bugs (stride-4 `srcDenarr`
indexing at `:692-700`, missing `√(2n+1)` at `:1925`) are documented, not fixed;
forces are out of scope.

### C shim (`hybridsolve_capi.cpp`)

- Owns `hs_eps_in` and a copy of upstream `allocate_dynamic()`
  from `main.cpp` (which is not vendored).
- `int hs_abi_version(void)` → `1`.
- `int hs_solve(int ns, const double *centers /*3×ns*/, const double *radii,
  const double *eps_r, int nq, const double *qpos /*3×nq*/, const double *q,
  int p, int im, double gmres_tol, int fmm_iprec, double source_tol,
  double sph_tol, int verbose)` — sets every global `read_para`/`set_precision`
  would set: `N = nq+ns`, `N_ion = nq`, `N_col = M = ns`, particles ordered ions
  first then spheres, sphere charges 0, `c_lj = 0`, `epsi_ion = 1`,
  `iter_indicator = 0`, `force_compute = 0`, `energy_compute = 1`, `p`, `im`,
  `imm = im`, `gmrestol`, `fmmtol`, `sourcetol`, `sphtol`; calls
  `allocate_arrays()`-equivalent allocation and `allocate_dynamic()`; calls
  `Coulomb_accelerations_Hybrid(0)`. When `verbose == 0`, fd 1 is redirected
  to `/dev/null` for the duration (with `fflush`). Returns 0 or a nonzero
  error code (2 = GMRES did not converge, read from the `ITLIN_INFO` returned by
  upstream's `gmres` call if it is reachable; 3 = invalid arguments).
- `int hs_result_sizes(int *nimages, int *p)`.
- `int hs_copy_results(double *imx, double *imy, double *imz, double *imq,
  int *imsphere, double *bknm /* 2 × (p+1) × (2p+1) × ns, complex interleaved */,
  double *energy)`. Image count = `im ×` number of `(i,j)` with `ionind[i][j]==1`.
- `void hs_ssheval(const double *ycoef, int p, const double *dir, double *out)`
  — calls upstream Fortran `ssheval_` so spherical-harmonic conventions are
  never re-derived in Julia.

### Julia API

```julia
sol = hybrid_solve(centers, radii, eps_r, charges, charge_pos;
                   p = 20, im = 6, gmres_tol = 1e-12, fmm_iprec = 5,
                   source_tol = 4.0, sph_tol = 4.0, verbose = false)::HybridSolution
eval_exterior_pot(sol, targets)::Vector{Float64}   # scattered potential
electrostatic_energy(sol)::Float64                 # HybridMD total energy / 4π
single_sphere_pointcharge_exterior(targets, a, eps_r, q, charge_pos)  # analytic
```

`HybridSolution` holds only Julia-owned arrays: geometry, charges, parameters,
image positions/strengths/owning sphere, per-sphere multipole coefficients, and
energy. It never refers back to C global state.

### Safety and errors

- One global `ReentrantLock` serialises every call into the library.
- Validation in Julia **before** any `ccall`: shapes and lengths agree; all
  values finite; `radii > 0`, `eps_r > 0`; spheres do not overlap; charges and
  targets strictly outside every sphere; `p ≥ 1`, `im ≥ 2`, `gmres_tol > 0`.
  Violations throw `ArgumentError`/`DimensionMismatch`.
- Nonzero shim error codes throw `ErrorException` with a message.
- Upstream reallocates on every call and never frees; patch 3 removes the large
  leak and the residual per-call leak is documented in the README.

## Testing

1. Analytic: one sphere + one charge, `d/a ∈ {3, 2, 1.5, 1.2}`, including a
   non-unit radius and an off-origin centre, against the Legendre series.
2. Analytic series self-check: interface conditions (continuity of potential and
   of `ε ∂ₙφ`) at surface points.
3. Units / energy: `electrostatic_energy` equals the free-pair energy plus
   `½ Σ q_k u_scat(X_k)`.
4. Per-sphere ε: two spheres with different ε mirrored through a plane.
5. Report values: `1a` and `0.05a` gaps from findings §3.7 (geometry in
   `LaplaceMFS.jl/docs/figures/precond.jl`), to ~1e-10.
6. Repeated calls with different `ns` and `p` in one session are reproducible.
7. Errors: invalid inputs throw before any `ccall`; `verbose = false` prints nothing.
8. Upstream hashes match `SHA256SUMS`.

## Out of scope

Forces, uniform-field excitation, charged spheres, complex ε, threading,
BinaryBuilder/JLL packaging. The build uses the system `gcc`/`g++`/`gfortran`.
