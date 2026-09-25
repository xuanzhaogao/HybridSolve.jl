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

# With the default source_tol = 4 the charge is imaged and the multipoles are ~1e-18, so the
# test above cannot see the coefficient convention. source_tol = 1 gives no images: the whole
# response lives in the multipoles, and a wrong radial factor or √(2n+1) scaling gives O(0.5) error.
@testset "multipole-only path pins the coefficient convention" begin
    for (a, c) in ((1.0, zeros(3)), (0.7, [0.3, -0.2, 0.5])), da in (3.0, 2.0)
        X = c .+ da * a .* normalize([0.2, 0.1, 1.0])
        T = _targets_around(c, a)
        sol = hybrid_solve(reshape(c, 1, 3), [a], [2.5], [1.0], reshape(X, 3, 1); p = 30, im = 12, source_tol = 1.0)
        @test isempty(sol.image_q)
        @test all(iszero(sol.multipoles[n + 1, j, 1]) for n in 0:sol.p, j in 1:(2sol.p + 1) if abs(j - sol.p - 1) > n)
        u = eval_exterior_pot(sol, T)
        ref = single_sphere_pointcharge_exterior(T, c, a, 2.5, 1.0, X)
        @test norm(u - ref) / norm(ref) < 1e-9
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

@testset "verbose = true prints HybridMD output" begin
    out = mktemp() do path, io
        redirect_stdout(io) do
            hybrid_solve(zeros(1, 3), [1.0], [2.5], [1.0], reshape([0.0, 0, 2.0], 3, 1); p = 6, verbose = true)
            Libc.flush_cstdio()
        end
        close(io); read(path, String)
    end
    @test !isempty(out)
end
