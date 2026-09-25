# HybridSolve.jl Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Julia package that wraps HybridMD's multi-sphere dielectric solver (vendored C++/Fortran, compiled into a shared library, called through `ccall`) as a reference for `LaplaceMFS.jl`.

**Architecture:** Upstream solver sources are vendored byte-identical under `deps/upstream/`, patched at build time by three numbered patches, and compiled with a thin C shim into `deps/usr/lib/libhybridsolve.so` by `deps/build.jl`. Julia sets up a solve through `hs_solve`, copies image charges and multipole coefficients into a Julia-owned `HybridSolution`, and evaluates the scattered potential itself, using upstream's Fortran `ssheval_` for the angular part.

**Tech Stack:** Julia ≥ 1.10 (local: juliaup `julia` 1.13; never `module load julia`), system `gcc`/`g++`/`gfortran` 11.5, GNU `patch`, stdlib `Libdl`, `LinearAlgebra`, `SHA`, `Test`.

**Spec:** `docs/plans/2026-09-25-hybridsolve-design.md` (read it first).

## Global Constraints

- Repo root: `/mnt/home/xgao1/project/laplacemfs/HybridSolve.jl` (branch `main`, remote `origin` = `git@github.com:xuanzhaogao/HybridSolve.jl.git`). **Never push.** Commit locally only.
- Upstream source of truth: `/mnt/home/xgao1/project/laplacemfs/HybridMD` at commit `1f87baf`. Read-only — never modify it.
- License: GPL-3.0 (copy upstream `LICENSE`).
- Array conventions: `centers` is `ns × 3`; `charge_pos` and `targets` are `3 × n`; `Float64` only.
- Units: `1/(4π r)` kernel, exterior permittivity 1. `φ_HybridSolve = φ_HybridMD / (4π)`.
- `eval_exterior_pot` returns the **scattered** potential only (incident free-charge potential excluded).
- Style (from `LaplaceMFS.jl/AGENTS.md`): 4-space indent, `snake_case` functions, `CamelCase` types, exports only in `src/HybridSolve.jl`, deterministic tests with explicit tolerances.
- Commit messages: short imperative, ending with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Julia commands run from the repo root with `julia --project=.`. HybridMD is single-threaded; do not set thread counts.
- File searches must stay inside `/mnt/home/xgao1/...` with `timeout 60` and `-maxdepth 7`.
- Do not loosen a numerical tolerance to make a test pass without recording the measured value and the reason in the commit message.

## Review Focus

1. **Non-unit radius and off-origin centre.** The multipole radial scaling (`1/ρ^(n+1)` vs `(a/ρ)^(n+1)`) is invisible at `a = 1, c = 0`. Task 4 pins it with an `a = 0.7`, off-centre test.
2. **Targets very close to a surface** (`ρ = 1.01a`). A user expects a finite, reasonably accurate answer; Task 4 asserts finite output and error < 1e-6.
3. **Repeated calls with a different `ns` and `p` in one session.** Globals are reallocated each call; a stale size would silently corrupt results. Task 5 checks bit-reproducibility after an interleaved different solve.
4. **Charges or targets on or inside a sphere, and overlapping spheres.** Must throw `ArgumentError` before any `ccall`, never segfault. Task 4.
5. **Many spheres at once.** A symmetric 8-sphere cube with a charge at the centre must give symmetric potentials (to discretisation error). Task 5.

---

### Task 1: Package scaffold and vendored upstream

**Files:**
- Create: `Project.toml`, `LICENSE`, `.gitignore`, `src/HybridSolve.jl`, `test/runtests.jl`, `test/upstream.jl`
- Create: `deps/upstream/` (copied files), `deps/upstream/SHA256SUMS`, `deps/upstream/UPSTREAM.md`

**Interfaces:**
- Produces: module `HybridSolve`; `deps/upstream/` tree with the layout below; `SHA256SUMS` in `sha256sum` format (`<hex>  <relative path>`).

- [ ] **Step 1: Create `Project.toml`**

Generate a UUID with `julia -e 'using UUIDs; println(uuid4())'` and use it:

```toml
name = "HybridSolve"
uuid = "<generated>"
version = "0.1.0"
authors = ["Xuanzhao Gao <xgao@flatironinstitute.org> and contributors"]

[deps]
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
julia = "1.10"

[extras]
SHA = "ea8e919c-243c-51af-8825-aaa63cd721ce"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["SHA", "Test"]
```

- [ ] **Step 2: Vendor upstream files**

```bash
U=/mnt/home/xgao1/project/laplacemfs/HybridMD
D=deps/upstream
mkdir -p $D/header $D/GMRES $D/JacobiGaussQuad $D/FMM3dlib $D/sht
cp $U/{Coulomb_accelerations_Hybrid.cpp,MDpara.cpp,MDpara.h,ran.h,allocate.cpp,output_force.cpp,LICENSE} $D/
cp $U/header/itlin.h $D/header/
cp $U/GMRES/{gmres.c,utils.c} $D/GMRES/
cp $U/JacobiGaussQuad/jacobi_rule.cpp $D/JacobiGaussQuad/
for f in l3dtrans laprouts3d rotviarecur3 cdjseval3d d3mtreeplot d3tstrcr lfmm3drouts triasymq triagauc triquadflatlib l3dtrirouts lfmm3dtria second trilib l3dterms lfmm3dpart rotproj triahquad; do cp $U/FMM3dlib/$f.f $D/FMM3dlib/; done
for f in dfft legeexps prini prinm sshexps xrecursion yrecursion; do cp $U/sht/$f.f $D/sht/; done
cp $U/LICENSE LICENSE
(cd $D && find . -type f ! -name SHA256SUMS ! -name UPSTREAM.md | sort | sed 's|^\./||' | xargs sha256sum > SHA256SUMS)
```

The list is exactly upstream `Makefile.gnu`'s `LIBOBJECTS` Fortran/C sources plus the solver's own C++ files. Write `deps/upstream/UPSTREAM.md`:

```markdown
# Vendored upstream

Files copied byte-identical from HybridMD (Gan, Jiang, Luijten & Xu), commit `1f87baf`,
GPL-3.0. Only the electrostatic solver is vendored; MD drivers are not.
Do not edit these files — changes go in `deps/patches/`. `SHA256SUMS` is checked by
`test/upstream.jl`.
```

- [ ] **Step 3: Write the failing hash test** — `test/upstream.jl`

```julia
using SHA

@testset "vendored upstream is byte-identical" begin
    root = joinpath(pkgdir(HybridSolve), "deps", "upstream")
    lines = filter(!isempty, readlines(joinpath(root, "SHA256SUMS")))
    @test length(lines) == 36
    for line in lines
        hex, rel = split(line; limit = 2)
        rel = strip(rel)
        @test bytes2hex(open(sha256, joinpath(root, rel))) == hex
    end
end
```

(36 = 7 top-level files incl. LICENSE + 1 header + 2 GMRES + 1 Jacobi + 18 FMM + 7 sht. If your count differs, recount the copy commands and fix the constant, not the copy.)

`test/runtests.jl`:

```julia
using HybridSolve
using Test

include("upstream.jl")
```

`src/HybridSolve.jl`:

```julia
module HybridSolve

end
```

`.gitignore`:

```
deps/build/
deps/usr/
deps/build.log
Manifest.toml
```

- [ ] **Step 4: Run tests** — `julia --project=. -e 'using Pkg; Pkg.test()'`. Expected: PASS (37 tests). Then corrupt one byte in a scratch copy mentally — do not edit vendored files — no further action.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "scaffold package and vendor HybridMD solver sources"` (with trailer).

---

### Task 2: Exact single-sphere Legendre series

**Files:**
- Create: `src/analytic.jl`, `test/analytic.jl`
- Modify: `src/HybridSolve.jl`, `test/runtests.jl`

**Interfaces:**
- Produces:
  - `single_sphere_pointcharge_exterior(targets::AbstractMatrix{Float64}, center::AbstractVector{Float64}, a::Float64, eps_r::Float64, q::Float64, charge_pos::AbstractVector{Float64})::Vector{Float64}` — scattered potential, valid for `|t − c| > a`.
  - `single_sphere_pointcharge_interior(...)` (same signature) — **total** potential inside, valid for `|t − c| < a`.

Formula (spec + findings §1.5), `θ` the angle between `t − c` and `X − c`, `d = |X − c|`, `ρ = |t − c|`:

- exterior scattered: `q/(4π) Σ_{n≥1} B_n a^(2n+1) / (d^(n+1) ρ^(n+1)) P_n(cosθ)`, `B_n = −n(ε_r−1)/(n(ε_r+1)+1)`
- interior total: `q/(4π) Σ_{n≥0} C_n ρ^n / d^(n+1) P_n(cosθ)`, `C_n = 1 + B_n` (`C_0 = 1`)

Truncation: fixed `nmax = min(20000, ceil(Int, log(1e-18) / log(ratio)) + 10)` with `ratio = a^2/(dρ)` (exterior) or `ρ/d` (interior). Never truncate on a relative term size — odd `P_n` vanish at `cosθ = 0`.

- [ ] **Step 1: Write the failing test** — `test/analytic.jl`

```julia
using LinearAlgebra

@testset "analytic series satisfies the interface conditions" begin
    a, eps_r, q = 0.7, 2.5, 1.3
    c = [0.2, -0.1, 0.4]
    X = c .+ [0.3, 0.2, 0.9] ./ norm([0.3, 0.2, 0.9]) .* (1.3a)
    for u in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], normalize([1.0, -2.0, 0.5]), normalize([-0.3, 0.2, 0.9]))
        h = 1e-6
        yo = reshape(c .+ (a + h) .* u, 3, 1)
        yi = reshape(c .+ (a - h) .* u, 3, 1)
        y0o = reshape(c .+ (a + 2h) .* u, 3, 1)
        y0i = reshape(c .+ (a - 2h) .* u, 3, 1)
        inc(t) = q / (4π * norm(vec(t) .- X))
        φo(t) = inc(t) + single_sphere_pointcharge_exterior(t, c, a, eps_r, q, X)[1]
        φi(t) = single_sphere_pointcharge_interior(t, c, a, eps_r, q, X)[1]
        # continuity of potential (extrapolate both sides to the surface)
        so = 2φo(yo) - φo(y0o)
        si = 2φi(yi) - φi(y0i)
        @test so ≈ si rtol = 1e-7
        # continuity of flux: ∂ₙφ⁺ = ε_r ∂ₙφ⁻ (one-sided first differences)
        dno = (φo(y0o) - φo(yo)) / h
        dni = (φi(yi) - φi(y0i)) / h
        @test dno ≈ eps_r * dni rtol = 1e-4
    end
end

@testset "analytic series limits" begin
    # n = 1 coefficient is minus the polarizability
    a, eps_r = 1.0, 4.0
    α = (eps_r - 1) / (eps_r + 2)
    # far charge: reaction ≈ induced dipole of the (nearly uniform) incident field
    d = 1e4; q = 1.0
    X = [0.0, 0.0, d]
    t = reshape([0.0, 0.0, 3.0], 3, 1)
    Ez = -q / (4π * d^2)   # LaplaceMFS convention: u_inc ≈ const − Ez z
    @test single_sphere_pointcharge_exterior(t, zeros(3), a, eps_r, q, X)[1] ≈ α * Ez * a^3 * 3.0 / 27.0 rtol = 1e-3
    # eps_r = 1 is invisible
    @test single_sphere_pointcharge_exterior(t, zeros(3), a, 1.0, q, [0.0, 0.0, 1.5])[1] == 0.0
end
```

Note the far-charge sign: with the charge on `+z`, `∂u_inc/∂z > 0` near the origin, i.e. `u_inc ≈ const + (q/4πd²) z = const − Ez z` with `Ez = −q/(4πd²)`; the exterior dipole response of `u_inc = −Ez z` is `α Ez a³ z/ρ³`. If this test fails by exactly a sign, recheck the derivation before touching the formula — the Legendre formula in the spec is authoritative.

Add `include("analytic.jl")` to `test/runtests.jl`.

- [ ] **Step 2: Run** — `julia --project=. -e 'using Pkg; Pkg.test()'`. Expected: FAIL, `UndefVarError: single_sphere_pointcharge_exterior`.

- [ ] **Step 3: Implement** — `src/analytic.jl`

```julia
_legendre_nmax(ratio) = min(20000, ceil(Int, log(1e-18) / log(ratio)) + 10)

function _series_geometry(t, center, charge_pos)
    rt = t .- center
    rx = charge_pos .- center
    ρ = sqrt(sum(abs2, rt)); d = sqrt(sum(abs2, rx))
    cθ = clamp(sum(rt .* rx) / (ρ * d), -1.0, 1.0)
    return ρ, d, cθ
end

function single_sphere_pointcharge_exterior(targets::AbstractMatrix{Float64}, center::AbstractVector{Float64},
                                            a::Float64, eps_r::Float64, q::Float64,
                                            charge_pos::AbstractVector{Float64})
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × n"))
    out = zeros(size(targets, 2))
    for j in axes(targets, 2)
        ρ, d, cθ = _series_geometry(view(targets, :, j), center, charge_pos)
        ρ > a || throw(ArgumentError("target $j is not outside the sphere"))
        d > a || throw(ArgumentError("charge is not outside the sphere"))
        base = a^2 / (d * ρ)
        nmax = _legendre_nmax(base)
        Pm, P = 1.0, cθ               # P_0, P_1
        fac = a / (d * ρ) * base      # a^(2n+1)/(d^(n+1) ρ^(n+1)) at n = 1
        s = 0.0
        for n in 1:nmax
            Bn = -n * (eps_r - 1) / (n * (eps_r + 1) + 1)
            s += Bn * fac * P
            Pm, P = P, ((2n + 1) * cθ * P - n * Pm) / (n + 1)
            fac *= base
        end
        out[j] = q / (4π) * s
    end
    return out
end

function single_sphere_pointcharge_interior(targets::AbstractMatrix{Float64}, center::AbstractVector{Float64},
                                            a::Float64, eps_r::Float64, q::Float64,
                                            charge_pos::AbstractVector{Float64})
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × n"))
    out = zeros(size(targets, 2))
    for j in axes(targets, 2)
        ρ, d, cθ = _series_geometry(view(targets, :, j), center, charge_pos)
        ρ < a || throw(ArgumentError("target $j is not inside the sphere"))
        ratio = ρ / d
        nmax = _legendre_nmax(max(ratio, 1e-300))
        Pm, P = 1.0, cθ
        fac = 1 / d                   # ρ^n / d^(n+1) at n = 0
        s = fac                       # n = 0 term, C_0 = 1, P_0 = 1
        fac *= ratio
        for n in 1:nmax
            Cn = 1 - n * (eps_r - 1) / (n * (eps_r + 1) + 1)
            s += Cn * fac * P
            Pm, P = P, ((2n + 1) * cθ * P - n * Pm) / (n + 1)
            fac *= ratio
        end
        out[j] = q / (4π) * s
    end
    return out
end
```

In `src/HybridSolve.jl` add `export single_sphere_pointcharge_exterior, single_sphere_pointcharge_interior` and `include("analytic.jl")`. (For `ρ = 0` in the interior, `cθ` is NaN — guard: if `ρ == 0` return `q/(4π d)`.)

- [ ] **Step 4: Run** — expected PASS.
- [ ] **Step 5: Commit** — `git commit -am "add exact single-sphere point-charge Legendre series"` (add new files first).

---

### Task 3: Patches, C shim and build

**Files:**
- Create: `deps/patches/0001-fix-cjk-comma.patch`, `0002-per-sphere-epsilon.patch`, `0003-skip-dead-initialization.patch`, `deps/patches/README.md`
- Create: `deps/shim/hybridsolve_capi.cpp`, `deps/build.jl`
- Create: `src/libhybrid.jl`, `test/library.jl`
- Modify: `src/HybridSolve.jl`, `test/runtests.jl`

**Interfaces:**
- Produces (C ABI, all `extern "C"`):
  - `int hs_abi_version(void)` → 1
  - `int hs_solve(int ns, const double *centers /*3×ns column-major*/, const double *radii, const double *eps_r, int nq, const double *qpos /*3×nq*/, const double *q, int p, int im, double gmres_tol, int fmm_iprec, double source_tol, double sph_tol, int verbose)` → 0 ok, 2 GMRES not converged, 3 bad args
  - `int hs_result_sizes(int *nimages, int *p, int *ns)`
  - `int hs_copy_results(double *imx, double *imy, double *imz, double *imq, int *imsphere /*0-based*/, double *bknm /*complex interleaved, (p+1)×(2p+1)×ns, n fastest*/, double *energy)`
  - `void hs_ssheval(const double *ycoef /*complex (p+1)×(2p+1)*/, int p, const double *target3, double *out_re_im)`
- Produces (Julia): `HybridSolve.libhybrid::String` (absolute path), `HybridSolve.LIB_LOCK::ReentrantLock`, `HybridSolve.abi_version()::Int`.

- [ ] **Step 1: Write patches.** For each, copy the upstream file to a scratch dir, edit, and produce a `-p1` unified diff:

```bash
S=$(mktemp -d); mkdir -p $S/a $S/b
cp deps/upstream/Coulomb_accelerations_Hybrid.cpp $S/a/; cp $S/a/*.cpp $S/b/
# edit $S/b/Coulomb_accelerations_Hybrid.cpp, then:
(cd $S && diff -u a/Coulomb_accelerations_Hybrid.cpp b/Coulomb_accelerations_Hybrid.cpp) > deps/patches/000N-name.patch
```

Make each patch separately against the output of the previous one (apply 0001 before diffing 0002, etc.).

- **0001**: line 1227 — replace the character `、` (U+3001) with `\`.
- **0002**: in the block `if(iter_indicator==0) { epsi_s=epsi_ion; for(i=0;i<N_col1;i++){epsi_i[i]=ei1;} for(i=N_col1;i<N_col;i++){epsi_i[i]=ei2;} }` (≈ lines 151–164), keep `epsi_s=epsi_ion;` and replace both loops with `for(i=0;i<N_col;i++) epsi_i[i]=hs_eps_in[i];`. Add `extern double *hs_eps_in;` near the top after the includes.
- **0003**: the call `initialization();` under `if(iter_indicator==0)` near line 94 — comment it out with `/* HybridSolve: dead allocations, see deps/patches/README.md */`.

`deps/patches/README.md` documents each patch (what, why, line), plus: the `exit(0)` image-overflow guard is unreachable because `allocate_dynamic` sizes `imnum_thresh = N_col*N*im` (the maximum); the force-path bugs (`:692-700` stride-4 `srcDenarr`, `:1925` missing `√(2n+1)`) are deliberately not fixed because forces are out of scope.

- [ ] **Step 2: Write the shim** — `deps/shim/hybridsolve_capi.cpp`

```cpp
// HybridSolve C ABI over the vendored HybridMD solver. GPL-3.0.
#include "MDpara.h"
#include "itlin.h"
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

void Coulomb_accelerations_Hybrid(int iprint);
void allocate_arrays();

double *hs_eps_in = NULL;

/* Copied verbatim from upstream main.cpp: allocate_dynamic() (main.cpp:338 to end of
   function). main.cpp is not vendored because it contains the MD driver. */
static void hs_allocate_dynamic()
{
    /* PASTE the body of allocate_dynamic() from
       /mnt/home/xgao1/project/laplacemfs/HybridMD/main.cpp:338 onwards, unchanged. */
}

extern "C" int hs_abi_version(void) { return 1; }

extern "C" int hs_solve(int ns, const double *centers, const double *radii, const double *eps_r,
                        int nq, const double *qpos, const double *qv,
                        int p_, int im_, double gmres_tol, int fmm_iprec,
                        double source_tol, double sph_tol, int verbose)
{
    if (ns < 1 || nq < 1 || p_ < 1 || im_ < 2) return 3;
    N = nq + ns; Ntype = 2;
    N_ion = nq; N_ion1 = nq; N_ion2 = 0;
    N_col = ns; M = ns; N_col1 = ns; N_col2 = 0;
    allocate_arrays();
    for (int i = 0; i < nq; i++) {
        x[i] = qpos[3*i]; y[i] = qpos[3*i+1]; z[i] = qpos[3*i+2];
        q[i] = qv[i]; r[i] = 0.0;
    }
    for (int s = 0; s < ns; s++) {
        int k = nq + s;
        x[k] = centers[3*s]; y[k] = centers[3*s+1]; z[k] = centers[3*s+2];
        q[k] = 0.0; r[k] = radii[s];
    }
    delete[] hs_eps_in; hs_eps_in = new double[ns];
    for (int s = 0; s < ns; s++) hs_eps_in[s] = eps_r[s];
    c_lj = 0.0; epsi_ion = 1.0;
    p = p_; im = im_; imm = im_;
    gmrestol = gmres_tol; fmmtol = fmm_iprec;
    sourcetol = source_tol; sphtol = sph_tol;
    iter_indicator = 0; force_compute = 0; energy_compute = 1;
    hs_allocate_dynamic();

    int saved = -1, devnull = -1;
    fflush(stdout);
    if (!verbose) {
        saved = dup(1); devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, 1);
    }
    Coulomb_accelerations_Hybrid(0);
    fflush(stdout);
    if (!verbose) { dup2(saved, 1); close(saved); close(devnull); }
    return 0;   /* Step 5 adds the GMRES convergence check */
}

static int hs_count_images()
{
    int pairs = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            if (ionind[i][j] == 1) pairs++;
    return pairs * im;
}

extern "C" int hs_result_sizes(int *nimages, int *p_out, int *ns_out)
{
    *nimages = hs_count_images(); *p_out = p; *ns_out = M;
    return 0;
}

extern "C" int hs_copy_results(double *ox_, double *oy_, double *oz_, double *oq_, int *osph,
                               double *bknm, double *energy)
{
    int n = hs_count_images();
    for (int i = 0; i < n; i++) {
        ox_[i] = imx[i]; oy_[i] = imy[i]; oz_[i] = imz[i]; oq_[i] = imq[i];
        osph[i] = imind[i] - N_ion;
    }
    memcpy(bknm, &Bknm[0][0][0], sizeof(complex) * M * (2*p+1) * (p+1));
    *energy = printed_ele_energy;
    return 0;
}

extern "C" void ssheval_(complex *, int *, double [3], complex *);

extern "C" void hs_ssheval(const double *ycoef, int p_, const double *target3, double *out)
{
    double t[3] = {target3[0], target3[1], target3[2]};
    complex f;
    ssheval_((complex *)ycoef, &p_, t, &f);
    out[0] = f.real; out[1] = f.imag;
}
```

Verify every global the shim writes is declared in `MDpara.h` (e.g. `imind`, `c_lj`, `epsi_ion`, `fmmtol`); if one is only defined in `Coulomb_accelerations_Hybrid.cpp`, add a matching `extern` declaration in the shim rather than patching upstream.

- [ ] **Step 3: Write `deps/build.jl`**

```julia
const DEPS = @__DIR__
const BUILD = joinpath(DEPS, "build")
const LIBDIR = joinpath(DEPS, "usr", "lib")

function build()
    rm(BUILD; force = true, recursive = true)
    cp(joinpath(DEPS, "upstream"), BUILD)
    for p in sort(filter(endswith(".patch"), readdir(joinpath(DEPS, "patches"); join = true)))
        run(`patch -p1 -d $BUILD -i $p`)
    end
    cp(joinpath(DEPS, "shim", "hybridsolve_capi.cpp"), joinpath(BUILD, "hybridsolve_capi.cpp"))
    objs = String[]
    inc = ["-I$BUILD", "-I$(joinpath(BUILD, "header"))"]
    for (dir, files) in ((BUILD, readdir(BUILD)), map(d -> (joinpath(BUILD, d), readdir(joinpath(BUILD, d))), ["GMRES", "JacobiGaussQuad", "FMM3dlib", "sht"])...)
        for f in files
            src = joinpath(dir, f); obj = joinpath(BUILD, splitext(f)[1] * ".o")
            if endswith(f, ".f")
                run(`gfortran -O2 -fPIC -std=legacy -fallow-argument-mismatch -w -c $src -o $obj`)
            elseif endswith(f, ".c")
                run(`gcc -O2 -fPIC $inc -c $src -o $obj`)
            elseif endswith(f, ".cpp")
                run(`g++ -O2 -fPIC -ansi -Wno-long-long -w $inc -c $src -o $obj`)
            else
                continue
            end
            push!(objs, obj)
        end
    end
    mkpath(LIBDIR)
    run(`g++ -shared -o $(joinpath(LIBDIR, "libhybridsolve.so")) $objs -lgfortran -lm -Wl,--no-undefined`)
end

build()
```

`-Wl,--no-undefined` is required: an unresolved symbol must fail the build, not the first `ccall`. If it reports symbols from files that are not vendored (e.g. an MD routine referenced from `MDpara.cpp` or `output_force.cpp`), provide a stub in the shim with a comment naming the upstream file it replaces — do not vendor MD code. If `-ansi` rejects code, use `-std=gnu++98`. If `jacobi_rule.cpp` defines `main`, that is harmless in a shared library.

- [ ] **Step 4: Julia bindings + failing test**

`src/libhybrid.jl`:

```julia
const libhybrid = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib", "libhybridsolve.so"))
const LIB_LOCK = ReentrantLock()

function _check_library()
    isfile(libhybrid) || error("libhybridsolve.so not found at $libhybrid; run `using Pkg; Pkg.build(\"HybridSolve\")`")
    return nothing
end

abi_version() = (_check_library(); Int(ccall((:hs_abi_version, libhybrid), Cint, ())))
```

Include it first in `src/HybridSolve.jl` (`using Libdl, LinearAlgebra` at the top of the module).

`test/library.jl`:

```julia
@testset "shared library" begin
    @test isfile(HybridSolve.libhybrid)
    @test HybridSolve.abi_version() == 1
    # raw smoke solve: one unit sphere, one charge at z = 1.5
    centers = [0.0, 0.0, 0.0]; radii = [1.0]; eps = [10.0]
    qpos = [0.0, 0.0, 1.5]; qv = [1.0]
    rc = lock(HybridSolve.LIB_LOCK) do
        ccall((:hs_solve, HybridSolve.libhybrid), Cint,
              (Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
               Cint, Cint, Cdouble, Cint, Cdouble, Cdouble, Cint),
              1, centers, radii, eps, 1, qpos, qv, 10, 6, 1e-12, 5, 4.0, 4.0, 0)
    end
    @test rc == 0
    nim = Ref{Cint}(0); pp = Ref{Cint}(0); ns = Ref{Cint}(0)
    ccall((:hs_result_sizes, HybridSolve.libhybrid), Cint, (Ref{Cint}, Ref{Cint}, Ref{Cint}), nim, pp, ns)
    @test nim[] == 6 && pp[] == 10 && ns[] == 1
end
```

Add `include("library.jl")` to `test/runtests.jl`.

- [ ] **Step 5: Build and run** — `julia --project=. -e 'using Pkg; Pkg.build(); Pkg.test()'`. Expected: PASS. Then add the GMRES check: read how upstream's `gmres(matsize,Bknm1D,&matvec,NULL,NULL,b,opt,info)` at `Coulomb_accelerations_Hybrid.cpp:543` reports failure (`info` is `struct ITLIN_INFO*`, see `header/itlin.h`; it may be a local). If the convergence status is reachable from the shim without further patching, return 2 when it signals failure; if not, leave `return 0` and record in `deps/patches/README.md` that non-convergence is not detected. Do not add a fourth patch for this.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "build HybridMD solver as a shared library with a C shim"`.

---

### Task 4: `hybrid_solve`, `eval_exterior_pot`, `electrostatic_energy`

**Files:**
- Create: `src/solve.jl`, `src/evaluation.jl`, `test/single_sphere.jl`, `test/validation.jl`
- Modify: `src/HybridSolve.jl`, `test/runtests.jl`

**Interfaces:**
- Consumes: Task 3 C ABI; Task 2 analytic functions (tests only).
- Produces:

```julia
struct HybridSolution
    centers::Matrix{Float64}      # ns × 3
    radii::Vector{Float64}
    eps_r::Vector{Float64}
    charges::Vector{Float64}
    charge_pos::Matrix{Float64}   # 3 × nq
    p::Int
    im::Int
    image_pos::Matrix{Float64}    # 3 × nimages
    image_q::Vector{Float64}      # HybridMD units (bare q/r)
    image_sphere::Vector{Int}     # 1-based owning sphere
    multipoles::Array{ComplexF64,3}   # (p+1) × (2p+1) × ns, n fastest, m = j - p - 1
    energy_raw::Float64           # HybridMD printed total electrostatic energy
end

hybrid_solve(centers::AbstractMatrix{<:Real}, radii::AbstractVector{<:Real}, eps_r::AbstractVector{<:Real},
             charges::AbstractVector{<:Real}, charge_pos::AbstractMatrix{<:Real};
             p::Integer = 20, im::Integer = 6, gmres_tol::Real = 1e-12, fmm_iprec::Integer = 5,
             source_tol::Real = 4.0, sph_tol::Real = 4.0, verbose::Bool = false)::HybridSolution
eval_exterior_pot(sol::HybridSolution, targets::AbstractMatrix{<:Real})::Vector{Float64}
electrostatic_energy(sol::HybridSolution)::Float64   # sol.energy_raw / (4π)
```

- [ ] **Step 1: Write failing tests** — `test/single_sphere.jl`

```julia
using LinearAlgebra

function _targets_around(c, a)
    dirs = [normalize(v) for v in ([0.0, 0, 1], [0.0, 0, -1], [1.0, 0, 0], [0.3, -0.5, 0.8], [-0.6, 0.2, 0.1])]
    reduce(hcat, [c .+ s * a .* u for u in dirs for s in (1.05, 1.5, 3.0)])
end

@testset "single sphere vs Legendre series" begin
    for (a, c) in ((1.0, zeros(3)), (0.7, [0.3, -0.2, 0.5]))
        for da in (3.0, 2.0, 1.5, 1.2)
            X = c .+ da * a .* normalize([0.2, 0.1, 1.0])
            T = _targets_around(c, a)
            sol = hybrid_solve(reshape(c, 1, 3), [a], [2.5], [1.0], reshape(X, 3, 1); p = 20, im = 12)
            u = eval_exterior_pot(sol, T)
            ref = single_sphere_pointcharge_exterior(T, c, a, 2.5, 1.0, X)
            @test norm(u - ref) / norm(ref) < 1e-9
        end
    end
end

@testset "near-surface target is finite and accurate" begin
    c = zeros(3); a = 1.0; X = [0.0, 0.0, 2.0]
    T = reshape([0.0, 0.0, 1.01], 3, 1)
    sol = hybrid_solve(reshape(c, 1, 3), [a], [2.5], [1.0], reshape(X, 3, 1); p = 20, im = 12)
    u = eval_exterior_pot(sol, T)
    @test all(isfinite, u)
    @test abs(u[1] - single_sphere_pointcharge_exterior(T, c, a, 2.5, 1.0, X)[1]) / abs(u[1]) < 1e-6
end

@testset "energy = pair energy + ½ Σ q u_scat" begin
    C = [0.0 0.0 0.0; 0.0 0.0 3.0]
    Q = [1.0, -0.7]
    X = [0.0 1.4; 0.0 0.3; -2.0 1.5]
    sol = hybrid_solve(C, [1.0, 1.0], [2.5, 4.0], Q, X; p = 20, im = 8)
    pair = Q[1] * Q[2] / (4π * norm(X[:, 1] - X[:, 2]))
    self = 0.5 * sum(Q .* eval_exterior_pot(sol, X))
    @test electrostatic_energy(sol) ≈ pair + self rtol = 1e-9
end

@testset "verbose = false prints nothing" begin
    out = mktemp() do path, io
        redirect_stdout(io) do
            hybrid_solve(zeros(1, 3), [1.0], [2.5], [1.0], reshape([0.0, 0, 2.0], 3, 1); p = 6)
            Libc.flush_cstdio()
        end
        close(io); read(path, String)
    end
    @test isempty(out)
end
```

`eval_exterior_pot` at a charge position `X_k` is valid (charges are outside every sphere); it returns the scattered potential only, so the self term above is exactly the polarisation energy.

`test/validation.jl`:

```julia
@testset "input validation" begin
    C = zeros(1, 3); X = reshape([0.0, 0, 2.0], 3, 1)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [1.0], reshape([0.0, 0, 0.5], 3, 1))  # charge inside
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [1.0], reshape([0.0, 0, 1.0], 3, 1))  # on surface
    @test_throws ArgumentError hybrid_solve([0.0 0 0; 0 0 1.5], [1.0, 1.0], [2.5, 2.5], [1.0], X)  # overlap
    @test_throws ArgumentError hybrid_solve(C, [-1.0], [2.5], [1.0], X)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [0.0], [1.0], X)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [1.0], X; p = 0)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [1.0], X; im = 1)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [NaN], X)
    @test_throws DimensionMismatch hybrid_solve(zeros(3, 1), [1.0], [2.5], [1.0], X)   # centers transposed
    @test_throws DimensionMismatch hybrid_solve(C, [1.0, 2.0], [2.5], [1.0], X)
    @test_throws DimensionMismatch hybrid_solve(C, [1.0], [2.5], [1.0, 2.0], X)
    sol = hybrid_solve(C, [1.0], [2.5], [1.0], X; p = 6)
    @test_throws ArgumentError eval_exterior_pot(sol, reshape([0.0, 0, 0.5], 3, 1))
    @test_throws DimensionMismatch eval_exterior_pot(sol, zeros(2, 4))
end
```

Include both files from `test/runtests.jl`. Run; expected FAIL (`UndefVarError: hybrid_solve`).

- [ ] **Step 2: Implement `src/solve.jl`**

```julia
function _validate(centers, radii, eps_r, charges, charge_pos, p, im, gmres_tol)
    size(centers, 2) == 3 || throw(DimensionMismatch("centers must be ns × 3"))
    ns = size(centers, 1)
    length(radii) == ns || throw(DimensionMismatch("length(radii) must equal ns = $ns"))
    length(eps_r) == ns || throw(DimensionMismatch("length(eps_r) must equal ns = $ns"))
    size(charge_pos, 1) == 3 || throw(DimensionMismatch("charge_pos must be 3 × nq"))
    length(charges) == size(charge_pos, 2) || throw(DimensionMismatch("length(charges) must equal nq"))
    ns ≥ 1 || throw(ArgumentError("need at least one sphere"))
    !isempty(charges) || throw(ArgumentError("need at least one charge"))
    for A in (centers, radii, eps_r, charges, charge_pos)
        all(isfinite, A) || throw(ArgumentError("inputs must be finite"))
    end
    all(>(0), radii) || throw(ArgumentError("radii must be positive"))
    all(>(0), eps_r) || throw(ArgumentError("eps_r must be positive"))
    p ≥ 1 || throw(ArgumentError("p must be ≥ 1"))
    im ≥ 2 || throw(ArgumentError("im must be ≥ 2"))
    gmres_tol > 0 || throw(ArgumentError("gmres_tol must be positive"))
    for i in 1:ns, j in i+1:ns
        norm(centers[i, :] - centers[j, :]) > radii[i] + radii[j] ||
            throw(ArgumentError("spheres $i and $j overlap or touch"))
    end
    _check_outside(centers, radii, charge_pos, "charge")
    return nothing
end

function _check_outside(centers, radii, pts, what)
    for k in axes(pts, 2), s in axes(centers, 1)
        norm(pts[:, k] - centers[s, :]) > radii[s] ||
            throw(ArgumentError("$what $k is not strictly outside sphere $s"))
    end
end

function hybrid_solve(centers::AbstractMatrix{<:Real}, radii::AbstractVector{<:Real}, eps_r::AbstractVector{<:Real},
                      charges::AbstractVector{<:Real}, charge_pos::AbstractMatrix{<:Real};
                      p::Integer = 20, im::Integer = 6, gmres_tol::Real = 1e-12, fmm_iprec::Integer = 5,
                      source_tol::Real = 4.0, sph_tol::Real = 4.0, verbose::Bool = false)
    _validate(centers, radii, eps_r, charges, charge_pos, p, im, gmres_tol)
    _check_library()
    C = Matrix{Float64}(centers); R = Vector{Float64}(radii); E = Vector{Float64}(eps_r)
    Q = Vector{Float64}(charges); X = Matrix{Float64}(charge_pos)
    ns = size(C, 1); nq = length(Q)
    Ct = Matrix(permutedims(C))   # 3 × ns, column-major = xyz per sphere
    lock(LIB_LOCK) do
        rc = ccall((:hs_solve, libhybrid), Cint,
                   (Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
                    Cint, Cint, Cdouble, Cint, Cdouble, Cdouble, Cint),
                   ns, Ct, R, E, nq, X, Q, p, im, gmres_tol, fmm_iprec, source_tol, sph_tol, verbose)
        rc == 0 || error(rc == 2 ? "HybridMD GMRES did not converge" : "hs_solve failed with code $rc")
        nim = Ref{Cint}(0); pp = Ref{Cint}(0); nsr = Ref{Cint}(0)
        ccall((:hs_result_sizes, libhybrid), Cint, (Ref{Cint}, Ref{Cint}, Ref{Cint}), nim, pp, nsr)
        n = Int(nim[])
        ix = zeros(n); iy = zeros(n); iz = zeros(n); iq = zeros(n); isph = zeros(Cint, n)
        B = zeros(ComplexF64, p + 1, 2p + 1, ns)
        en = Ref{Float64}(0.0)
        ccall((:hs_copy_results, libhybrid), Cint,
              (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{ComplexF64}, Ref{Float64}),
              ix, iy, iz, iq, isph, B, en)
        HybridSolution(C, R, E, Q, X, Int(p), Int(im), permutedims(hcat(ix, iy, iz)), iq,
                       Int.(isph) .+ 1, B, en[])
    end
end
```

- [ ] **Step 3: Implement `src/evaluation.jl`**

```julia
function _multipole_pot(sol::HybridSolution, s::Int, t::AbstractVector{Float64}, scratch::Matrix{ComplexF64})
    d = t .- view(sol.centers, s, :)
    ρ = norm(d); a = sol.radii[s]; p = sol.p
    B = view(sol.multipoles, :, :, s)
    for j in axes(B, 2), n in 0:p
        scratch[n + 1, j] = B[n + 1, j] * _radial(n, ρ, a)
    end
    out = zeros(2)
    ccall((:hs_ssheval, libhybrid), Cvoid, (Ptr{ComplexF64}, Cint, Ptr{Float64}, Ptr{Float64}), scratch, p, d, out)
    return out[1]
end

# Radial factor of the per-sphere expansion. Determine the correct form (1/ρ^(n+1),
# (a/ρ)^(n+1), or with a further power of a) from the non-unit-radius single-sphere test
# and a reading of how Bknm is formed in Coulomb_accelerations_Hybrid.cpp (≈ lines
# 480–560, ycoef scaled by pow(orad, k+1)). Record the reasoning here in one comment.
_radial(n, ρ, a) = (a / ρ)^(n + 1)

function eval_exterior_pot(sol::HybridSolution, targets::AbstractMatrix{<:Real})
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × ntrg"))
    T = Matrix{Float64}(targets)
    _check_outside(sol.centers, sol.radii, T, "target")
    ns = size(sol.centers, 1)
    scratch = zeros(ComplexF64, sol.p + 1, 2sol.p + 1)
    out = zeros(size(T, 2))
    lock(LIB_LOCK) do
        for k in axes(T, 2)
            t = view(T, :, k)
            φ = 0.0
            for i in eachindex(sol.image_q)
                φ += sol.image_q[i] / norm(t .- view(sol.image_pos, :, i))
            end
            for s in 1:ns
                φ += _multipole_pot(sol, s, t, scratch)
            end
            out[k] = φ / (4π)
        end
    end
    return out
end

electrostatic_energy(sol::HybridSolution) = sol.energy_raw / (4π)
```

Export `HybridSolution, hybrid_solve, eval_exterior_pot, electrostatic_energy`; include `solve.jl` then `evaluation.jl` after `libhybrid.jl`. The `HybridSolution` struct lives at the top of `solve.jl`.

- [ ] **Step 4: Run and resolve conventions.** `Pkg.test()`. The unit-radius cases pin the overall sign and the `1/(4π)` factor; the `a = 0.7` cases pin `_radial`. If the single-sphere test fails, diagnose — do not loosen: print `u ./ ref` per target. A constant ratio means units; a ratio that varies with `ρ` or with `a` means the radial factor; a ratio that varies with angle means the multipole layout (`m = j − p − 1`, n fastest) or which of `Bknm`/`BknmCopy` holds the final coefficients at the point `hs_copy_results` reads them (the energy block rescales `Bknm` by `sqrtk` and back — read `Coulomb_accelerations_Hybrid.cpp` around lines 600–740 and 2380–2400). If the analytic test cannot reach 1e-9 at `p = 20, im = 12`, sweep `im ∈ {6, 12, 24}` and `p ∈ {20, 30}` and choose parameters that converge; report the sweep in the commit message.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "add hybrid_solve, scattered-potential evaluation and energy"`.

---

### Task 5: Multi-sphere reference tests

**Files:**
- Create: `test/multisphere.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `hybrid_solve`, `eval_exterior_pot` from Task 4.

- [ ] **Step 1: Write the tests** — `test/multisphere.jl`

```julia
using LinearAlgebra

# Geometry of LaplaceMFS.jl/docs/figures/precond.jl: a = 1, eps_r = 2.5, q = 1 at (0,0,-2),
# spheres at the origin and at (0,0,R). Reference values: findings doc §3.7.
const PRECOND_TARGETS = [0.0 0.0 1.3 -1.4  0.9;
                         0.0 0.9 0.0  0.0 -1.1;
                        -1.6 2.6 0.7  1.9  0.3]

function _precond_solve(R; p, im)
    hybrid_solve([0.0 0.0 0.0; 0.0 0.0 R], [1.0, 1.0], [2.5, 2.5], [1.0], reshape([0.0, 0.0, -2.0], 3, 1);
                 p = p, im = im, gmres_tol = 1e-12)
end

@testset "matches converged MFS at a 1a gap" begin
    ref = [-0.004005725937, 0.000323354212, 0.001332353924, 0.000605745006, 0.000915527134]
    u = eval_exterior_pot(_precond_solve(3.0; p = 20, im = 8), PRECOND_TARGETS)
    @test maximum(abs.(u - ref)) < 5e-12
end

@testset "matches converged MFS at a 0.05a gap" begin
    ref = [-0.004156595607, 0.001662586859, 0.001069724756, 0.000895478853, 0.000691702359]
    u = eval_exterior_pot(_precond_solve(2.05; p = 40, im = 8), PRECOND_TARGETS)
    @test maximum(abs.(u - ref)) < 5e-12
end

@testset "per-sphere eps_r is honoured" begin
    # mirror symmetry through z = 0: spheres swap, charge flips sign of z
    C = [0.0 0.0 -1.6; 0.0 0.0 1.6]
    X = reshape([0.4, 0.0, 0.0], 3, 1)
    T = [0.0 0.0; 0.5 0.5; 3.0 -3.0]
    s1 = hybrid_solve(C, [1.0, 1.0], [2.0, 8.0], [1.0], X; p = 20, im = 8)
    s2 = hybrid_solve(C, [1.0, 1.0], [8.0, 2.0], [1.0], X; p = 20, im = 8)
    u1 = eval_exterior_pot(s1, T); u2 = eval_exterior_pot(s2, T)
    @test u1[1] ≈ u2[2] rtol = 1e-8
    @test u1[2] ≈ u2[1] rtol = 1e-8
    @test abs(u1[1] - u1[2]) > 1e-4 * abs(u1[1])   # the two spheres really differ
end

@testset "repeated calls are reproducible across different sizes" begin
    a = eval_exterior_pot(_precond_solve(3.0; p = 12, im = 6), PRECOND_TARGETS)
    # interleave a solve with a different ns and p
    C8 = reduce(vcat, [[x y z] for x in (-1.5, 1.5) for y in (-1.5, 1.5) for z in (-1.5, 1.5)])
    hybrid_solve(C8, fill(1.0, 8), fill(3.0, 8), [1.0], reshape([0.0, 0.0, 0.0], 3, 1); p = 16, im = 6)
    b = eval_exterior_pot(_precond_solve(3.0; p = 12, im = 6), PRECOND_TARGETS)
    @test a == b
end

@testset "8-sphere cube is symmetric" begin
    C8 = reduce(vcat, [[x y z] for x in (-1.5, 1.5) for y in (-1.5, 1.5) for z in (-1.5, 1.5)])
    sol = hybrid_solve(C8, fill(1.0, 8), fill(3.0, 8), [1.0], reshape([0.0, 0.0, 0.0], 3, 1); p = 16, im = 6)
    T = [3.5 -3.5 0.0 0.0 0.0 0.0;
         0.0 0.0 3.5 -3.5 0.0 0.0;
         0.0 0.0 0.0 0.0 3.5 -3.5]
    u = eval_exterior_pot(sol, T)
    # the sht grid has a polar axis, so symmetry holds only to discretisation error
    @test maximum(u) - minimum(u) < 1e-6 * maximum(abs.(u))
    @test all(isfinite, u) && maximum(abs.(u)) > 0
end
```

Add `include("multisphere.jl")` to `test/runtests.jl`.

- [ ] **Step 2: Run** — `Pkg.test()`. Expected: PASS. If the report-value tests fail, first check the geometry against `precond.jl` and the findings doc (§3.7, `0.05a` needs `p ≥ 40`, `im ≥ 6`), then sweep `p` upward; the reference is correct to ~1e-12 per the findings. The charge in the cube test sits at the centre, outside all spheres (distance √3·1.5 ≈ 2.6 > 1). If symmetry fails, print `u` and the spread at `p = 16, 24` — a spread that does not shrink with `p` is a real defect (e.g. a per-sphere index bug), report it.

- [ ] **Step 3: Commit** — `git add test && git commit -m "add multi-sphere reference tests against converged MFS"`.

---

### Task 6: Documentation and CI

**Files:**
- Create: `README.md`, `AGENTS.md`, `.github/workflows/CI.yml`

- [ ] **Step 1: `README.md`** — sections: purpose (reference solver for LaplaceMFS.jl, GPL-3.0 because it vendors HybridMD); install (`Pkg.develop(path=...)`, `Pkg.build("HybridSolve")`, needs `gcc`, `g++`, `gfortran`, `patch`); quick start (the one-sphere example from `test/single_sphere.jl`); conventions (the Global Constraints bullets on shapes and units, scattered-only potential, energy includes the free-pair Coulomb term); accuracy knobs (`im` is the limiting parameter; shipped HybridMD precision table is not used; `0.05a` gaps need `p ≥ 40`); limitations (no forces — upstream force path is broken, see `deps/patches/README.md`; no uniform field; not thread-safe, calls serialised by a lock; each call leaks the upstream per-solve allocations of size `O(ns·p² + N²)`, so very long loops of solves will grow memory). Mention that `eval_exterior_pot` shares its name with LaplaceMFS's — qualify when both are loaded.

- [ ] **Step 2: `AGENTS.md`** — adapt `LaplaceMFS.jl/AGENTS.md`: structure (`src/`, `deps/upstream` read-only, `deps/patches`, `deps/shim`, `test/`), commands (`julia --project=. -e 'using Pkg; Pkg.build(); Pkg.test()'`), rule that upstream files are never edited and every change to upstream behaviour is a numbered patch documented in `deps/patches/README.md`.

- [ ] **Step 3: CI** — copy the `test` job shape from `LaplaceMFS.jl/.github/workflows/CI.yml` (Julia `1.12`, ubuntu-latest, x64, `julia-actions/setup-julia@v2`, `cache@v2`, `julia-buildpkg@v1`, `julia-runtest@v1`), adding before build: `- run: sudo apt-get update && sudo apt-get install -y gfortran patch`. No docs job, no codecov.

- [ ] **Step 4: Full clean run** — `rm -rf deps/build deps/usr && julia --project=. -e 'using Pkg; Pkg.build(); Pkg.test()'`. Expected: build succeeds, all tests PASS. Paste the test summary into the commit message body.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "add README, AGENTS guide and CI"`.
