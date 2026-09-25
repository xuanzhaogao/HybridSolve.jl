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

@testset "analytic series argument validation" begin
    a, eps_r, q = 1.0, 2.5, 1.0
    c = zeros(3)
    X_out = [0.0, 0.0, 2.0]   # |X - c| = 2.0 > a, valid charge position
    X_in = [0.0, 0.0, 0.3]    # |X - c| = 0.3 < a, invalid charge position

    # exterior: target must be outside the sphere
    t_in = reshape([0.0, 0.0, 0.5], 3, 1)   # ρ = 0.5 < a
    @test_throws ArgumentError single_sphere_pointcharge_exterior(t_in, c, a, eps_r, q, X_out)
    # exterior: charge must be outside the sphere
    t_out = reshape([0.0, 0.0, 1.5], 3, 1)  # ρ = 1.5 > a
    @test_throws ArgumentError single_sphere_pointcharge_exterior(t_out, c, a, eps_r, q, X_in)

    # interior: target must be inside the sphere
    @test_throws ArgumentError single_sphere_pointcharge_interior(t_out, c, a, eps_r, q, X_out)
    # interior: charge must be outside the sphere (regression: d=0.3 < a=1, ρ=0.6 < a=1
    # previously returned q/(4π·0.3) silently instead of throwing)
    t_reg = reshape([0.0, 0.0, 0.6], 3, 1)
    @test_throws ArgumentError single_sphere_pointcharge_interior(t_reg, c, a, eps_r, q, X_in)
end
