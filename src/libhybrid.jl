const libhybrid = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib", "libhybridsolve.so"))
const LIB_LOCK = ReentrantLock()
const ABI_VERSION = 1
const _ABI_CHECKED = Ref(false)

const _DEPS_DIR = normpath(joinpath(@__DIR__, "..", "deps"))

# Pkg does not build test-only dependencies, so compile on first use when the library is missing.
function _build_library()
    log = joinpath(_DEPS_DIR, "build.log")
    @info "HybridSolve: compiling libhybridsolve.so (first use; needs gcc, g++, gfortran, patch). Log: $log"
    cmd = `$(Base.julia_cmd()) --startup-file=no $(joinpath(_DEPS_DIR, "build.jl"))`
    success(pipeline(cmd; stdout = log, stderr = log)) ||
        error("building libhybridsolve.so failed; see $log")
    return nothing
end

function _check_library()
    _ABI_CHECKED[] && return nothing
    lock(LIB_LOCK) do
        isfile(libhybrid) || _build_library()
    end
    isfile(libhybrid) || error("libhybridsolve.so not found at $libhybrid; run `using Pkg; Pkg.build(\"HybridSolve\")`")
    v = Int(ccall((:hs_abi_version, libhybrid), Cint, ()))
    v == ABI_VERSION || error("libhybridsolve.so has ABI version $v but HybridSolve expects $ABI_VERSION; " *
                              "rebuild with `using Pkg; Pkg.build(\"HybridSolve\")`")
    _ABI_CHECKED[] = true
    return nothing
end

abi_version() = (_check_library(); Int(ccall((:hs_abi_version, libhybrid), Cint, ())))
