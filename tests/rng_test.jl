# Test for consistency of random number generation for the same seed

using StableRNGs
using Test

include("../src/rr_gee.jl")

function rng_consistency(fn, args)
	output_1 = fn(args...; rng=StableRNG(32))
	output_2 = fn(args...; rng=StableRNG(32))
	output_1 == output_2
end

# Test randorth(n, p)
arglist = (10, 3)
@test rng_consistency(randorth, arglist)

# Test gencov(n)
arglist = (10)
@test rng_consistency(gencov, arglist)

# Test genrr(n,p,r)
arglist = (5, 3, 2)
@test rng_consistency(genrr, arglist)

# Test mgeedata
Bm = zeros(5, 3)
Bm[:,1] = [1, 0, -1, 2, 3]
Bm[:,2] = [4, 0.1, 1, 1, -2]
Bm[:,3] = [2, 1, -1, 0, 0]
arglist = (100, 5, 3, 5, 4, Bm)

@test rng_consistency(mgeedata, arglist)

# Test mgee_rr
Bm = genrr(5, 3, 2)
testlog = open("logs/testlog.txt", "w")
arglist = (100, 5, 3, 5, 4, 2, Bm, testlog)
@test rng_consistency(mgee_rr, arglist)

# Test sim_gee

