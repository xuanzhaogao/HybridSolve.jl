@testset "shared library" begin
    @test isfile(HybridSolve.libhybrid)
    @test HybridSolve.abi_version() == 1
    # raw smoke solve: one unit sphere, one charge at z = 1.5
    centers = [0.0, 0.0, 0.0]; radii = [1.0]; eps = [10.0]
    qpos = [0.0, 0.0, 1.5]; qv = [1.0]
    rc = lock(HybridSolve.LIB_LOCK) do
        ccall((:hs_solve, HybridSolve.libhybrid), Cint,
              (Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
               Cint, Cint, Cdouble, Cint, Cdouble, Cdouble, Cint),
              1, centers, radii, eps, 1, qpos, qv, 10, 6, 1e-12, 5, 4.0, 4.0, 0)
    end
    @test rc == 0
    nim = Ref{Cint}(0); pp = Ref{Cint}(0); ns = Ref{Cint}(0)
    ccall((:hs_result_sizes, HybridSolve.libhybrid), Cint, (Ref{Cint}, Ref{Cint}, Ref{Cint}), nim, pp, ns)
    @test nim[] == 6 && pp[] == 10 && ns[] == 1
    # energy (HybridMD units, 1/r kernel) equals 0.5 q * 4π φ_scattered(charge)
    imx = zeros(6); imy = zeros(6); imz = zeros(6); imq = zeros(6); imsph = zeros(Cint, 6)
    bknm = zeros(2 * 11 * 21); energy = Ref(0.0)
    ccall((:hs_copy_results, HybridSolve.libhybrid), Cint,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Float64}, Ref{Float64}),
          imx, imy, imz, imq, imsph, bknm, energy)
    @test all(==(0), imsph)
    @test imq[1] ≈ -(9 / 11) / 1.5 rtol = 1e-14   # Kelvin image, gamma = (eps-1)/(eps+1)
    φ = single_sphere_pointcharge_exterior(reshape(qpos, 3, 1), centers, 1.0, 10.0, 1.0, qpos)[1]
    @test energy[] ≈ 0.5 * 4π * φ rtol = 1e-7

    raw_solve(tol, im) = lock(HybridSolve.LIB_LOCK) do
        ccall((:hs_solve, HybridSolve.libhybrid), Cint,
              (Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64},
               Cint, Cint, Cdouble, Cint, Cdouble, Cdouble, Cint),
              1, centers, radii, eps, 1, qpos, qv, 10, im, tol, 5, 4.0, 4.0, 0)
    end
    @test raw_solve(1e-12, 1) == 3     # im < 2 rejected
    @test raw_solve(1e-40, 6) == 2     # unreachable tolerance: GMRES reports maxiter exceeded
    # after a failed solve (rc = 2) the result accessors refuse
    sizes() = ccall((:hs_result_sizes, HybridSolve.libhybrid), Cint, (Ref{Cint}, Ref{Cint}, Ref{Cint}),
                    Ref{Cint}(0), Ref{Cint}(0), Ref{Cint}(0))
    @test sizes() == 3
    copy_rc = ccall((:hs_copy_results, HybridSolve.libhybrid), Cint,
                    (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Float64}, Ref{Float64}),
                    imx, imy, imz, imq, imsph, bknm, energy)
    @test copy_rc == 3

    # in a fresh process, before any solve, the accessors return 3 instead of segfaulting
    code = """
        using HybridSolve
        f = HybridSolve.libhybrid
        a = Ref{Cint}(0); b = Ref{Cint}(0); c = Ref{Cint}(0)
        rc1 = ccall((:hs_result_sizes, f), Cint, (Ref{Cint}, Ref{Cint}, Ref{Cint}), a, b, c)
        v = zeros(1); iv = zeros(Cint, 1); e = Ref(0.0)
        rc2 = ccall((:hs_copy_results, f), Cint,
                    (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Float64}, Ref{Float64}),
                    v, v, v, v, iv, v, e)
        exit(rc1 == 3 && rc2 == 3 ? 0 : 1)
        """
    @test success(run(ignorestatus(`$(Base.julia_cmd()) --startup-file=no --project=$(pkgdir(HybridSolve)) -e $code`)))
end
