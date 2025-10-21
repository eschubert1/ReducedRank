using LinearAlgebra
using StableRNGs
using EstimatingEquationsRegression
using Statistics
using Distributions

include("rr_sims.jl")
include("covariance_generators.jl")

"""
    generate_gee2_covariates(nobs, ngroup, m, num_mean_covariates, num_scale_covariates)

Generate design matrices for mean, scale, and correlation models.
"""
function generate_gee2_covariates(nobs, ngroup, num_mean_covariates, num_scale_covariates; rng=StableRNG(1))
 N = nobs*ngroup
 pm = num_mean_covariates
 pv = num_scale_covariates

 # Generate group position covariate
 group_position = repeat(collect(1:ngroup), nobs)

# Generate mean model design matrix with correlated columns
 rm = 0.5625 # correlation parameter
 X1 = randn(rng, N)
 X2 = reshape(repeat(X1, pm-2), N, pm-2) + randn(rng, N, pm-2)*sqrt(rm)
 Xm = hcat(ones(N), X1, X2)

 # Generate scale model design matrix
 # Xv = randn(rng, N, pv)
 Xv = hcat(ones(N), group_position) # Predict scale by group position 

 # Generate correlation model design matrix
 Xr = reshape(group_position, N, 1) # Predict correlations by group position

 Xm, Xv, Xr
end

"""
    generate_gee2_response(nobs, ngroup, m, Xm, Bm, C; rng=StableRNG(1))

Generate multivariate response data for GEE2 simulations.
"""
function generate_gee2_response(nobs, ngroup, Xm, Bm, C; rng=StableRNG(1))
 N = nobs*ngroup
 m = size(Bm, 2)
 # Mean structure
 Ey = Xm*Bm
 
 # Generate errors
 E = generate_errors(nobs, ngroup, m, C; rng=rng)

 Y = Ey + E
 Y
end
