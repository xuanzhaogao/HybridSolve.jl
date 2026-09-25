using LinearAlgebra

# Upstream GMRES stops on the absolute residual, so hybrid_solve nondimensionalises (lengths by
# max(radii), charges by max|q|) before the ccall. The physics is linear in q and scales as q/L
# (potential) and q²/L (energy); these tests pin that down at the 0.05a reference geometry and on
# the multipole-only single-sphere path.
_rel(a, b) = norm(a - b) / norm(b)

const _GAP_C = [0.0 0.0 0.0; 0.0 0.0 2.05]
const _GAP_X = reshape([0.0, 0.0, -2.0], 3, 1)
_gap_solve(L, q) = hybrid_solve(L .* _GAP_C, [L, L], [2.5, 2.5], [q], L .* _GAP_X; p = 40, im = 8)

const _SS_C = reshape([0.3, -0.2, 0.5], 1, 3)
const _SS_X = reshape([0.3, -0.2, 0.5] .+ 2.0 * 0.7 .* normalize([0.2, 0.1, 1.0]), 3, 1)
const _SS_T = [0.3 1.4 -0.6; -0.2 0.1 -0.2; 2.0 0.4 -1.0]
_ss_solve(L, q) = hybrid_solve(L .* _SS_C, [0.7L], [2.5], [q], L .* _SS_X; p = 30, im = 12, source_tol = 1.0)

@testset "linearity in the charge" begin
    ref = _gap_solve(1.0, 1.0)
    uref = eval_exterior_pot(ref, PRECOND_TARGETS)
    for q in (1e-8, 1e6)
        s = _gap_solve(1.0, q)
        @test _rel(eval_exterior_pot(s, PRECOND_TARGETS) ./ q, uref) < 1e-11
        @test electrostatic_energy(s) / q^2 ≈ electrostatic_energy(ref) rtol = 1e-11
    end
    ref = _ss_solve(1.0, 1.0)
    s = _ss_solve(1.0, 1e-8)
    @test isempty(s.image_q)
    @test _rel(eval_exterior_pot(s, _SS_T) ./ 1e-8, eval_exterior_pot(ref, _SS_T)) < 1e-11
    @test electrostatic_energy(s) / 1e-16 ≈ electrostatic_energy(ref) rtol = 1e-11
end

@testset "invariance under length scaling" begin
    ref = _gap_solve(1.0, 1.0)
    uref = eval_exterior_pot(ref, PRECOND_TARGETS)
    for L in (1e-4, 100.0)
        s = _gap_solve(L, 1.0)
        @test _rel(eval_exterior_pot(s, L .* PRECOND_TARGETS) .* L, uref) < 1e-11
        @test electrostatic_energy(s) * L ≈ electrostatic_energy(ref) rtol = 1e-11
        @test s.image_pos ≈ L .* ref.image_pos rtol = 1e-12
    end
    ref = _ss_solve(1.0, 1.0)
    uref = eval_exterior_pot(ref, _SS_T)
    for L in (1e-4, 100.0)
        s = _ss_solve(L, 1.0)
        @test _rel(eval_exterior_pot(s, L .* _SS_T) .* L, uref) < 1e-11
        @test electrostatic_energy(s) * L ≈ electrostatic_energy(ref) rtol = 1e-11
    end
end
