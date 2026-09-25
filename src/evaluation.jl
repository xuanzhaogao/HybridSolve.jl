function _multipole_pot(sol::HybridSolution, s::Int, t::AbstractVector{Float64}, scratch::Matrix{ComplexF64})
    d = (t .- view(sol.centers, s, :)) ./ sol.L   # multipoles live in scaled units
    ρ = norm(d); p = sol.p
    B = view(sol.multipoles, :, :, s)
    for j in axes(B, 2), n in 0:p
        scratch[n + 1, j] = B[n + 1, j] * _radial(n, ρ)
    end
    out = zeros(2)
    ccall((:hs_ssheval, libhybrid), Cvoid, (Ptr{ComplexF64}, Cint, Ptr{Float64}, Ptr{Float64}), scratch, p, d, out)
    return out[1]
end

# Radial factor: the RHS already carries a^(n+1) (ycoef·pow(orad,k+1), Coulomb_accelerations_Hybrid.cpp:500)
# and upstream's energy block evaluates Bknm/r^(n+1) with the raw distance r (:2135-2151), so ρ is the
# absolute distance in the (scaled) units the solve ran in, not ρ/a.
_radial(n, ρ) = inv(ρ)^(n + 1)

"""
    eval_exterior_pot(sol, targets) -> Vector{Float64}

Scattered (polarisation) potential at the `3 × ntrg` `targets`, all strictly outside every sphere,
in `1/(4π r)` units. The free-charge potential is excluded.
"""
function eval_exterior_pot(sol::HybridSolution, targets::AbstractMatrix{<:Real})
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × ntrg"))
    T = Matrix{Float64}(targets)
    all(isfinite, T) || throw(ArgumentError("targets must be finite"))
    _check_outside(sol.centers, sol.radii, T, "target")
    _check_library()
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
            φm = 0.0
            for s in 1:ns
                φm += _multipole_pot(sol, s, t, scratch)
            end
            φ += φm * (sol.Qs / sol.L)
            out[k] = φ / (4π)
        end
    end
    return out
end

"""
    electrostatic_energy(sol) -> Float64

Total electrostatic energy (free-charge pair energy plus polarisation energy) in `1/(4π r)` units.
"""
electrostatic_energy(sol::HybridSolution) = sol.energy_raw / (4π)
