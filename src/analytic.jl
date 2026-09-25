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
        if ρ == 0
            out[j] = q / (4π * d)
            continue
        end
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
