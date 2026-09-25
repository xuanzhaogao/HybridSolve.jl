module HybridSolve

using Libdl, LinearAlgebra

export single_sphere_pointcharge_exterior, single_sphere_pointcharge_interior

include("libhybrid.jl")
include("analytic.jl")

end
