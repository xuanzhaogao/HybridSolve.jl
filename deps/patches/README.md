# Patches to the vendored HybridMD sources

`deps/upstream/` is a byte-identical copy of HybridMD@1f87baf and is never edited.
`deps/build.jl` copies it to `deps/build/`, then applies these patches in
lexical order with `patch -p1`. All three touch only
`Coulomb_accelerations_Hybrid.cpp`. Each patch was made against the output of
the previous one, so the line numbers below refer to the file after the
earlier patches have been applied.

| patch | line (upstream) | what | why |
|:-|-|-|-|
| `0001-fix-cjk-comma.patch` | 1227 | replaces the full-width comma `、` (U+3001) at the end of the line with `\` | The stray character is a syntax error, so the file does not compile. It sits in the `p >= 5` branch of the force kernel, and the intended character is the line-continuation `\` used on the surrounding lines. |
| `0002-per-sphere-epsilon.patch` | top of file; ≈151–164 | adds `extern double *hs_eps_in;` after the includes, and replaces the two loops `epsi_i[i]=ei1` (for `i < N_col1`) / `epsi_i[i]=ei2` (for `N_col1 ≤ i < N_col`) with `for(i=0;i<N_col;i++) epsi_i[i]=hs_eps_in[i];`. `epsi_s=epsi_ion;` is kept. | Upstream supports only two colloid species, each with its own permittivity. HybridSolve needs one permittivity per sphere, which the C shim stores in `hs_eps_in` (defined in `deps/shim/hybridsolve_capi.cpp`). |
| `0003-skip-dead-initialization.patch` | ≈94 | replaces the body of `if(iter_indicator==0) initialization();` with an empty statement `;` followed by a comment. The empty statement keeps the `if` from capturing the next statement. | `initialization()` only allocates and fills arrays that the energy path never reads: `powerd`, `newpowerd`, `tempcoef`, `tempcoefimag`, `wquad`, `xquad`, `powerxquad`, `rotangle`. It is O(M²·p⁴) in memory and time, and it would leak on every solve. |

## Upstream behaviour that is deliberately not patched

- **The image-overflow `exit(0)`** (the `imcount>imnum_thresh` check, ≈ line 282)
  can never fire. The shim's copy of `allocate_dynamic()` sets
  `imnum_thresh = N_col*N*im`, which is the most images possible: each of the
  `N` sources makes at most `im` images in each of the `N_col` spheres.
- **Bugs in the force path** are not fixed, because forces are out of scope and the
  shim sets `force_compute = 0`:
  - lines 692–700 read `srcDenarr` with stride 4, but it holds `(p+1)²` entries per source;
  - line 1925 is missing the `√(2n+1)` factor.

## GMRES convergence detection

`gmres()` (`GMRES/gmres.c`) sets `info->rcode` as follows:

| `rcode` | meaning |
|:-|-|
| 0 | converged |
| 2 | `opt->maxiter` exceeded |
| 1 | QR factorisation failed |
| 20, 21, −99, −9xx | bad input or allocation failure |

`info` is a non-static global `struct ITLIN_INFO *info` defined in
`Coulomb_accelerations_Hybrid.cpp`. The shim declares it `extern`, so it can
read the status without a patch. `hs_solve` returns 2 whenever
`info->rcode != 0`.

## Notes for callers of the shim

- Upstream multiplies `Bknm` in place by `sqrtk[n] = √(2n+1)` for its FMM step
  (around line 600, "rescale"), then divides that factor back out before the
  energy block (≈ lines 730–736, "un-rescale"). By the time `hs_solve`
  returns, `Bknm` is therefore already back in the plain `sht`/`ssheval`
  convention (no extra `√(2n+1)`), and `hs_copy_results` copies it as-is,
  stored as `[sphere][m+p][n]` with `n` fastest. Entries with `|m| > n` are
  not solution coefficients (they still hold GMRES's initial guess and are
  never written back).
- Every `hs_solve` call allocates fresh global arrays (`allocate_arrays`,
  `allocate_dynamic`, and the `iter_indicator==0` blocks upstream) and never
  frees the previous ones, so each solve leaks memory in proportion to the
  problem size.
- `hs_result_sizes` and `hs_copy_results` return 3 and read no solver globals
  unless the most recent `hs_solve` returned 0. The result globals are NULL at
  process start and stale after a failed solve.
