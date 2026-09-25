const libhybrid = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib", "libhybridsolve.so"))
const LIB_LOCK = ReentrantLock()
const ABI_VERSION = 1
const _ABI_CHECKED = Ref(false)

function _check_library()
    _ABI_CHECKED[] && return nothing
    isfile(libhybrid) || error("libhybridsolve.so not found at $libhybrid; run `using Pkg; Pkg.build(\"HybridSolve\")`")
    v = Int(ccall((:hs_abi_version, libhybrid), Cint, ()))
    v == ABI_VERSION || error("libhybridsolve.so has ABI version $v but HybridSolve expects $ABI_VERSION; " *
                              "rebuild with `using Pkg; Pkg.build(\"HybridSolve\")`")
    _ABI_CHECKED[] = true
    return nothing
end

abi_version() = (_check_library(); Int(ccall((:hs_abi_version, libhybrid), Cint, ())))
