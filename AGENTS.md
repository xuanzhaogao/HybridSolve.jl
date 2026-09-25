# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the package code. Entry point is `src/HybridSolve.jl`, which
  declares the exports and includes the focused modules: `libhybrid.jl`
  (library path, `ccall` wrappers, the global solve lock), `solve.jl`
  (`HybridSolution`, `hybrid_solve`, input validation), `evaluation.jl`
  (`eval_exterior_pot`, `electrostatic_energy`), and `analytic.jl` (the exact
  single-sphere point-charge Legendre series used as a reference in tests).
- `deps/upstream/` is a byte-identical, **read-only** copy of HybridMD
  (`@1f87baf`); never edit it directly. `deps/upstream/SHA256SUMS` records a
  hash per vendored file, and `test/upstream.jl` checks them.
- `deps/patches/` holds the numbered patches applied to a scratch copy of
  `deps/upstream/` at build time, plus a `README.md` that documents each
  patch's line, effect, and reason, and the upstream behaviour deliberately
  left unpatched (the unreachable image-overflow `exit(0)`, the known
  force-path bugs).
- `deps/shim/hybridsolve_capi.cpp` is the thin C ABI (`hs_abi_version`,
  `hs_solve`, `hs_result_sizes`, `hs_copy_results`, `hs_ssheval`) that Julia
  calls into; it also carries a verbatim copy of upstream's
  `allocate_dynamic()` from the non-vendored `main.cpp`.
- `deps/build.jl` copies `deps/upstream/` to `deps/build/`, applies the
  patches with `patch -p1`, compiles everything (plus the shim) with the
  system `gcc`/`g++`/`gfortran`, and links `deps/usr/lib/libhybridsolve.so`.
  `deps/build/` and `deps/usr/` are build artifacts (git-ignored); never
  commit them.
- `test/` holds unit and regression tests, orchestrated by `test/runtests.jl`:
  `upstream.jl` (hash check), `analytic.jl` (Legendre series self-checks),
  `library.jl` (raw C-ABI smoke tests), `single_sphere.jl` (vs. the analytic
  series), `validation.jl` (input validation), `multisphere.jl` (multi-sphere
  regressions against converged `LaplaceMFS.jl` values).
- `docs/plans/` contains the design spec for this package.

## Build, Test, and Development Commands
- Use the `julia` on `PATH` (juliaup-managed); never `module load julia`.
- `julia --project=. -e 'using Pkg; Pkg.build("HybridSolve")'`: compile
  `libhybridsolve.so` from the vendored sources.
- `julia --project=. -e 'using Pkg; Pkg.build(); Pkg.test()'`: full clean
  build followed by the test suite. Run this after any change under `deps/`.
- `julia --project=. -e 'using Pkg; Pkg.test()'`: run the test suite only
  (reuses an existing build).
- Rebuild from scratch with
  `rm -rf deps/build deps/usr && julia --project=. -e 'using Pkg; Pkg.build(); Pkg.test()'`.
- CI runs on Julia `1.12` (`.github/workflows/CI.yml`); it installs
  `gfortran` and `patch` before building, since GitHub's `ubuntu-latest`
  runner ships `gcc`/`g++` but not those two.

## Coding Style & Naming Conventions
- Follow existing Julia style: 4-space indentation, no tabs, and concise
  function-level doc/comments only where needed.
- Use `snake_case` for functions/variables and `CamelCase` for concrete types
  (`HybridSolution`).
- Keep exported API declarations in `src/HybridSolve.jl`; place implementation
  details in the most relevant source file.

## Upstream and Patch Discipline
- **Never edit `deps/upstream/`.** Any change to upstream behaviour is a new,
  numbered patch (`NNNN-description.patch`) added to `deps/patches/`, applied
  in lexical order by `deps/build.jl`, and documented as a new row in
  `deps/patches/README.md` (line, what, why). Regenerate `SHA256SUMS` only
  when deliberately re-vendoring upstream, never to make a stale hash match.
- Patches touch as little as possible and never change solver numerics beyond
  what the row in `deps/patches/README.md` describes; if a change is purely
  numerical (e.g. a real upstream bug fix), it must be called out explicitly
  as such, not folded silently into an unrelated patch.

## Testing Guidelines
- Add tests in `test/*.jl` and include new files from `test/runtests.jl`.
- Prefer deterministic numeric checks with explicit tolerances (`atol`/`rtol`)
  and relative-error assertions; do not loosen a tolerance to make a test pass
  without recording the measured value and the reason.
- `p` (spherical-harmonic order) limits tight sphere–sphere gaps; `im` (image
  count) limits charges close to a sphere surface. At a `0.05a` gap, `p = 20`
  stalls at `2.75e-9` for both `im = 8` and `im = 16`, and `p = 40, im = 8`
  reaches `3.9e-13`; for a charge at `1.05a` from one sphere, `im = 4` gives
  `1.7e-5` and `im = 8` gives `3.6e-12`. When adding a regression, check
  convergence in both knobs before choosing values.
- Results must scale exactly with the inputs (potential `∝ q/L`, energy
  `∝ q²/L`; see `test/scaling.jl`): `hybrid_solve` nondimensionalises before
  the `ccall` because upstream GMRES uses an absolute residual.

## Commit & Pull Request Guidelines
- Short, imperative commit messages.
- For pull requests, include: purpose, key numerical/algorithmic changes,
  test updates, and docs updates if behavior or API changed.
