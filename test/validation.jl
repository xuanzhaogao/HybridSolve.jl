@testset "input validation" begin
    C = zeros(1, 3); X = reshape([0.0, 0, 2.0], 3, 1)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [1.0], reshape([0.0, 0, 0.5], 3, 1))  # charge inside
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [1.0], reshape([0.0, 0, 1.0], 3, 1))  # on surface
    @test_throws ArgumentError hybrid_solve([0.0 0 0; 0 0 1.5], [1.0, 1.0], [2.5, 2.5], [1.0], X)  # overlap
    @test_throws ArgumentError hybrid_solve(C, [-1.0], [2.5], [1.0], X)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [0.0], [1.0], X)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [1.0], X; p = 0)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [1.0], X; im = 1)
    @test_throws ArgumentError hybrid_solve(C, [1.0], [2.5], [NaN], X)
    @test_throws DimensionMismatch hybrid_solve(zeros(3, 1), [1.0], [2.5], [1.0], X)   # centers transposed
    @test_throws DimensionMismatch hybrid_solve(C, [1.0, 2.0], [2.5], [1.0], X)
    @test_throws DimensionMismatch hybrid_solve(C, [1.0], [2.5], [1.0, 2.0], X)
    sol = hybrid_solve(C, [1.0], [2.5], [1.0], X; p = 6)
    @test_throws ArgumentError eval_exterior_pot(sol, reshape([0.0, 0, 0.5], 3, 1))
    @test_throws DimensionMismatch eval_exterior_pot(sol, zeros(2, 4))
end
