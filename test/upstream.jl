using SHA

@testset "vendored upstream is byte-identical" begin
    root = joinpath(pkgdir(HybridSolve), "deps", "upstream")
    lines = filter(!isempty, readlines(joinpath(root, "SHA256SUMS")))
    @test length(lines) == 36
    for line in lines
        hex, rel = split(line; limit = 2)
        rel = strip(rel)
        @test bytes2hex(open(sha256, joinpath(root, rel))) == hex
    end
end
