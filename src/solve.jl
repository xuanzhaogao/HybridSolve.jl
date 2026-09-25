"""
    HybridSolution

Result of [`hybrid_solve`](@ref). `image_q` is in HybridMD units (potential `q/r`, no `1/(4π)`).
`multipoles[n+1, j, s]` is the coefficient of `Y_n^m(θ, φ) / ρ^(n+1)` about sphere `s`, with
`m = j - p - 1` and `ρ` the absolute distance to the centre, in the `sht`/`ssheval` convention
(no extra `√(2n+1)`); entries with `|m| > n` are zero. `energy_raw` is HybridMD's printed total
electrostatic energy (HybridMD units).
"""
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

"""
    hybrid_solve(centers, radii, eps_r, charges, charge_pos; p = 20, im = 6, gmres_tol = 1e-12,
                 fmm_iprec = 5, source_tol = 4.0, sph_tol = 4.0, verbose = false) -> HybridSolution

Solve the multi-sphere dielectric problem (exterior permittivity 1) with HybridMD's
image-charge + spherical-harmonic hybrid method. `centers` is `ns × 3`, `charge_pos` is `3 × nq`.
"""
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
        rc = ccall((:hs_result_sizes, libhybrid), Cint, (Ref{Cint}, Ref{Cint}, Ref{Cint}), nim, pp, nsr)
        (rc == 0 && pp[] == p && nsr[] == ns) || error("hs_result_sizes failed (rc = $rc)")
        n = Int(nim[])
        ix = zeros(n); iy = zeros(n); iz = zeros(n); iq = zeros(n); isph = zeros(Cint, n)
        B = zeros(ComplexF64, p + 1, 2p + 1, ns)
        en = Ref{Float64}(0.0)
        rc = ccall((:hs_copy_results, libhybrid), Cint,
                   (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{ComplexF64}, Ref{Float64}),
                   ix, iy, iz, iq, isph, B, en)
        rc == 0 || error("hs_copy_results failed (rc = $rc)")
        # Upstream multiplies Bknm by √(2n+1) for its FMM step and divides it back before the
        # energy block (Coulomb_accelerations_Hybrid.cpp:600-605 and :730-736), so the copy is
        # already in the ssheval convention. Entries with |m| > n still hold GMRES's initial
        # guess (the RHS, :519-525) and are never written back (:563-570): zero them.
        for s in 1:ns, j in 1:(2p + 1), n1 in 1:(p + 1)
            abs(j - p - 1) > n1 - 1 && (B[n1, j, s] = zero(ComplexF64))
        end
        HybridSolution(C, R, E, Q, X, Int(p), Int(im), Matrix(permutedims(hcat(ix, iy, iz))), iq,
                       Int.(isph) .+ 1, B, en[])
    end
end
