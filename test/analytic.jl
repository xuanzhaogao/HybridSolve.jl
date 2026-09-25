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
