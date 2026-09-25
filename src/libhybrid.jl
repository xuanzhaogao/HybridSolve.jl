const libhybrid = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib", "libhybridsolve.so"))
const LIB_LOCK = ReentrantLock()

function _check_library()
    isfile(libhybrid) || error("libhybridsolve.so not found at $libhybrid; run `using Pkg; Pkg.build(\"HybridSolve\")`")
    return nothing
end

abi_version() = (_check_library(); Int(ccall((:hs_abi_version, libhybrid), Cint, ())))
