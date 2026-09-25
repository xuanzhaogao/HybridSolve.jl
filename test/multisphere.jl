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
    # Target 2, (0, 0.9, 2.6), is 0.985 from sphere 2's centre (0, 0, 3.0) at this gap: it sits
    # inside sphere 2's nominal a = 1 surface (though outside the r_p = 0.5a MFS proxy shell the
    # findings-doc reference was sourced from), so eval_exterior_pot's _check_outside correctly
    # rejects it -- precond.jl's target set was reused unchanged across R in {8, 4, 3, 2.5, 2.2,
    # 2.05} and is only geometrically valid for every target at R = 8, 4, 2.05. Compare the four
    # targets that are genuinely exterior to both spheres here.
    keep = [1, 3, 4, 5]
    u = eval_exterior_pot(_precond_solve(3.0; p = 20, im = 8), PRECOND_TARGETS[:, keep])
    @test maximum(abs.(u - ref[keep])) < 5e-12
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
    keep = [1, 3, 4, 5]   # target 2 is inside sphere 2 at this R = 3.0 gap; see the 1a-gap testset
    a = eval_exterior_pot(_precond_solve(3.0; p = 12, im = 6), PRECOND_TARGETS[:, keep])
    # interleave a solve with a different ns and p
    C8 = reduce(vcat, [[x y z] for x in (-1.5, 1.5) for y in (-1.5, 1.5) for z in (-1.5, 1.5)])
    hybrid_solve(C8, fill(1.0, 8), fill(3.0, 8), [1.0], reshape([0.0, 0.0, 0.0], 3, 1); p = 16, im = 6)
    b = eval_exterior_pot(_precond_solve(3.0; p = 12, im = 6), PRECOND_TARGETS[:, keep])
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

# 200 charges within source_tol of both spheres give 2412 images, so imcount + N and
# M·ntheta·nphi = 2·(2p)² both exceed upstream's FMM_thresh = 2000 and the FMM path runs.
# The response is linear, so it must equal the superposition of single-charge (direct-sum) solves.
@testset "FMM path matches superposition of single-charge solves" begin
    C = [0.0 0.0 -1.25; 0.0 0.0 1.25]
    nq = 200; gold = π * (3 - sqrt(5))
    X = reduce(hcat, [begin z = 1 - 2(k - 0.5) / nq; r = sqrt(1 - z^2)
                          2.5 .* [r * cos(gold * k), r * sin(gold * k), z] end for k in 1:nq])
    Q = [cos(1.7k) for k in 1:nq]
    T = [3.0 0.0 -2.0 0.5; 0.0 3.2 1.0 0.3; 0.0 1.0 -3.5 0.0]
    sol = hybrid_solve(C, [1.0, 1.0], [2.5, 4.0], Q, X; p = 20, im = 6)
    @test length(sol.image_q) + nq + 2 ≥ 2000
    u = eval_exterior_pot(sol, T)
    ref = sum(Q[k] .* eval_exterior_pot(hybrid_solve(C, [1.0, 1.0], [2.5, 4.0], [1.0], X[:, k:k]; p = 20, im = 6), T)
              for k in 1:nq)
    @test norm(u - ref) / norm(ref) < 1e-11   # measured 1.3e-12
end
