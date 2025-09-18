# Test for consistency of random number generation for the same seed

using StableRNGs
using HypothesisTests
using Distributions
using Test

include("../src/rr_gee.jl")

function rng_consistency(fn, args)
	output_1 = fn(args...; rng=StableRNG(32))
	output_2 = fn(args...; rng=StableRNG(32))
	output_1 == output_2
end

@testset "Random Number Generation Consistency" begin
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
end

# For this test, determine if the generated random unitary matrices 
# are distributed according to Haar measure
# According to this paper: https://www.jstor.org/stable/2951843?seq=1
# The trace of a sample of matrices generated from Haar measure should be 
# N(0, 1) as the size of the matrix goes to infinity

function haar_test(nsamples, n; rng=StableRNG(1))
 A = [randorth(n,n;rng=rng)]
 for i in 2:nsamples
 	push!(A, randorth(n,n;rng=rng))
 end
 sample_traces = tr.(A)
 normality_test = ApproximateOneSampleKSTest(sample_traces, Normal())
 pvalue(normality_test) > 0.05
end

@testset "Haar distribution test" begin
    @test haar_test(1000, 30; rng=StableRNG(852)) 
    @test haar_test(1000, 20; rng=StableRNG(3))
end
