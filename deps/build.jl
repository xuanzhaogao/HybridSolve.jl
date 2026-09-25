# Builds deps/usr/lib/libhybridsolve.so from the vendored HybridMD sources
# (deps/upstream, never edited), the patches in deps/patches and the C shim.
const DEPS = @__DIR__
const BUILD = joinpath(DEPS, "build")
const LIBDIR = joinpath(DEPS, "usr", "lib")
const SUBDIRS = ["GMRES", "JacobiGaussQuad", "FMM3dlib", "sht"]

function build()
    rm(BUILD; force = true, recursive = true)
    cp(joinpath(DEPS, "upstream"), BUILD)
    for p in sort(filter(endswith(".patch"), readdir(joinpath(DEPS, "patches"); join = true)))
        run(`patch -p1 -d $BUILD -i $p`)
    end
    cp(joinpath(DEPS, "shim", "hybridsolve_capi.cpp"), joinpath(BUILD, "hybridsolve_capi.cpp"))
    objs = String[]
    inc = ["-I$BUILD", "-I$(joinpath(BUILD, "header"))"]
    dirs = [BUILD; [joinpath(BUILD, d) for d in SUBDIRS]]
    for (dir, files) in [(d, readdir(d)) for d in dirs]
        for f in files
            src = joinpath(dir, f)
            obj = joinpath(BUILD, splitext(f)[1] * ".o")
            if endswith(f, ".f")
                run(`gfortran -O2 -fPIC -std=legacy -fallow-argument-mismatch -w -c $src -o $obj`)
            elseif endswith(f, ".c")
                run(`gcc -O2 -fPIC $inc -c $src -o $obj`)
            elseif endswith(f, ".cpp")
                run(`g++ -O2 -fPIC -ansi -Wno-long-long -w $inc -c $src -o $obj`)
            else
                continue
            end
            obj in objs && error("object name collision: $obj")
            push!(objs, obj)
        end
    end
    mkpath(LIBDIR)
    lib = joinpath(LIBDIR, "libhybridsolve.so")
    run(`g++ -shared -o $lib $objs -lgfortran -lm -Wl,--no-undefined`)
    return lib
end

build()
