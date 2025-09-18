using Test

include("../src/rr_gee.jl")

# Rank test
@testset "Reduced rank test" begin
	@test rank(genrr(10, 8, 5; rng=StableRNG(1))) == 5;
	@test rank(genrr(5, 4, 2; rng=StableRNG(23))) == 2;
	@test rank(genrr(30, 30, 1; rng=StableRNG(3))) == 1;
end;

# Positive definite test
function isposdef(A)
    try 
        cholesky(A)
    catch e
        return false
    end
    return true
end

@testset "Positive definite test" begin
	@test isposdef(gencov(10; rng=StableRNG(3)));
	@test isposdef(gencov(5; rng=StableRNG(7)));
	@test isposdef(gencov(13; rng=StableRNG(9)));
end;
