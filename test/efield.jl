using LinearAlgebra

# Uniform external field E, LaplaceMFS convention u_inc = -E·x. For one sphere of radius a and
# relative permittivity eps_r the scattered field is α a³ E·(t - c) / |t - c|³, α = (eps_r-1)/(eps_r+2).
_dipole_ref(T, c, a, eps_r, E) = [((eps_r - 1) / (eps_r + 2)) * a^3 * dot(E, T[:, j] - c) / norm(T[:, j] - c)^3
                                  for j in axes(T, 2)]

@testset "uniform field: single sphere vs analytic dipole" begin
    for (a, c, E, eps_r) in ((1.0, zeros(3), [0.0, 0.0, 1.0], 2.5),
                             (0.7, [0.3, -0.2, 0.5], [0.4, -1.1, 0.3], 10.0))
        T = reduce(hcat, [c .+ s * a .* normalize(u) for u in ([1.0, 0, 0], [0.2, 0.5, -0.8], [-0.3, 0.9, 0.1])
                          for s in (1.05, 2.0, 4.0)])
        sol = hybrid_solve(reshape(c, 1, 3), [a], [eps_r], Float64[], zeros(3, 0); p = 8, efield = E)
        ref = _dipole_ref(T, c, a, eps_r, E)
        @test norm(eval_exterior_pot(sol, T) - ref) / norm(ref) < 1e-11
    end
end

@testset "uniform field: superposition with point charges" begin
    C = [0.0 0.0 0.0; 0.0 0.0 3.0]; X = [0.0 1.8; 0.0 0.0; -2.0 1.5]; q = [1.0, -0.5]
    E = [0.7, 0.0, -0.4]
    T = [0.0 1.3 -1.4; 0.0 0.0 0.5; -1.6 0.7 4.5]
    both = eval_exterior_pot(hybrid_solve(C, [1.0, 1.0], [2.5, 2.5], q, X; p = 20, im = 8, efield = E), T)
    qonly = eval_exterior_pot(hybrid_solve(C, [1.0, 1.0], [2.5, 2.5], q, X; p = 20, im = 8), T)
    fonly = eval_exterior_pot(hybrid_solve(C, [1.0, 1.0], [2.5, 2.5], Float64[], zeros(3, 0); p = 20, efield = E), T)
    @test norm(both - (qonly + fonly)) / norm(both) < 1e-11
end

@testset "uniform field: length scaling" begin
    C = [0.0 0.0 0.0; 0.0 0.0 2.6]; E = [0.0, 1.0, 1.0]; T = [0.5 -1.2; 1.4 0.3; 1.3 3.9]
    u1 = eval_exterior_pot(hybrid_solve(C, [1.0, 1.0], [2.5, 4.0], Float64[], zeros(3, 0); p = 20, efield = E), T)
    for L in (1e-3, 100.0)
        uL = eval_exterior_pot(hybrid_solve(L .* C, [L, L], [2.5, 4.0], Float64[], zeros(3, 0); p = 20, efield = E), L .* T)
        @test norm(uL ./ L - u1) / norm(u1) < 1e-11          # u = α a³ E·r/ρ³ scales like E L
    end
end

@testset "uniform field: validation and energy" begin
    C = zeros(1, 3)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], Float64[], zeros(3, 0))          # nothing drives it
    @test_throws DimensionMismatch hybrid_solve(C, [1.0], [2.5], Float64[], zeros(3, 0); efield = [1.0, 0.0])
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], Float64[], zeros(3, 0); efield = [NaN, 0.0, 0.0])
    sol = hybrid_solve(C, [1.0], [2.5], Float64[], zeros(3, 0); p = 6, efield = [0.0, 0.0, 1.0])
    @test_throws ArgumentError electrostatic_energy(sol)
    # a zero field is the same as no field
    X = reshape([0.0, 0.0, 2.0], 3, 1); T = reshape([0.0, 1.5, 0.0], 3, 1)
    @test eval_exterior_pot(hybrid_solve(C, [1.0], [2.5], [1.0], X; p = 10, efield = zeros(3)), T) ==
          eval_exterior_pot(hybrid_solve(C, [1.0], [2.5], [1.0], X; p = 10), T)
end
