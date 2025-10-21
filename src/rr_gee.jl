using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra
using Distributions
using JLD2

using EstimatingEquationsRegression: IdentityLink, LogLink, SigmoidLink

include("rr_sims.jl")
include("estimators.jl")
include("metrics.jl")
include("covariance_generators.jl")
include("data_generators.jl")


"""
    mgeedata(nobs, ngroup, m, pm, pv, Bm; rng=StableRNG(1), err_method=0)

Generate data for a multivariate GEE with provided mean coefficient matrix Bm.

Data is generated at three levels:
 - the conditional mean of y: E(y) = Xm*Bm
 - the scale of y: Var(y) = exp.(Xv*Bv)
 - the correlation of y, which is generated differently  
   depending on the choice of err_method:
  1) For each each cluster of size ni x m, generate a \
   general (ni x m)x(ni x m) positive definite correlation matrix
  2) (Default) generate Gaussian noise such that different \
    responses in a cluster have correlation f=0.9 \
    and different rows in a cluster for the j^th response \
    have correlation rr[j], where rr = range(0.1, 0.9, length=m)

Additionally, the columns of Xm are generated to be correlated with each other.
The outcome data consists of 'm' response variables from 'nobs' clusters,
with each cluster of size 'ngroup' so that the total sample size is
N = nobs*ngroup. Xm and Xv have dimensions N x pm and N x pv,
respectively, while Bm and Bv have dimension pm x m and pv x m, respectively.

# Arguments
- `nobs::Integer`: the number of independent clusters
- `ngroup::Integer`: the number of observations in a cluster
- `m::Integer`: the number of response variables
- `pm::Integer`: the number of mean model covariates
- `pv::Integer`: the number of scale model covariates
- `Bm::AbstractMatrix`: the mean model coefficient matrix
- `rng::StableRNG`: stable random number generation stream. \
		    Default is StableRNG(1)
- `err_method::Integer`: option of generating correlation structure \
			in random errors.

# See also
[`sim_gee`](@ref), [`mgee_rr`](@ref), [`clustered_errors`](@ref)
"""
function mgeedata(nobs::Integer, ngroup::Integer, m::Integer, pm::Integer,
				 pv::Integer, Bm::AbstractMatrix;
				 rng=StableRNG(1), err_method=0)
 N = nobs*ngroup

 # Residual correlation for each outcome, spaced equally between 0.1 and 0.9
 rr = range(0.1, 0.9, length=m)

 # Coefficients for scale model
 Bv = rand(rng, pv, m)

 groups = repeat(1:nobs, inner=ngroup)
 # Vector of positions within each group
 group_position = repeat(1:ngroup, outer=nobs)

 # Generate mean model design matrix with correlated columns
 rm = 0.5625 # correlation parameter
 X1 = randn(rng, N)
 X2 = reshape(repeat(X1, pm-2), N, pm-2) + randn(rng, N, pm-2)*sqrt(rm)
 Xm = hcat(ones(N), X1, X2)
 	
 h = range(1, 5, m) # response-specific scale parameter

 # Generate scale model design matrix
 # Xv = randn(rng, N, pv)
 Xv = hcat(ones(N), group_position) # Predict scale by group position 
 Xr = hcat(ones(N), group_position) # Predict correlations by group position

 Ey = Xm*Bm

 #Vy = exp.(Xv * Bv)
 #clamp!(Vy, 0.1, 10)

 if err_method == 1
	E, CovB = generate_errors(groups, m, Xm; rng=rng)
 elseif err_method == 2
 	parms = (2, 3, 1)	
	E, CovB = generate_errors(groups, m, Xm; rng=rng, 
					  cov_method="spacetime", parms=parms)
 else
	# Heteroscedasticity through group position
 	Vy = reshape(repeat(group_position, m), N, m)
	f = 0.9
	E, CovB = additive_cor(nobs, ngroup, m, Xm, Vy, f, rr, h; rng=rng)
 end

 y = Ey + E

 y, Bm, Bv, Xm, Xv, Xr, groups, CovB
end

"""
    mgee_rr(nobs, ngroup, m, pm, pv, rank, Bm, xlog; rng=StableRNG(1))

Evaluate the performance of several reduced rank estimators for the
mean coefficient matrix.

Fits a multivariate GEE2 model, constructs several reduced rank estimators, 
and computes the Frobenius distance of estimators to
Bm, the true mean model coefficient matrix, 
and Ey = Xm*Bm, the true conditional mean of Y.

The covariance of vec(Bhat) is estimated using vcov(mm), 
where mm is a list of the m GEE2 models.
Then, the dense estimate (Bhat) is computed along
with 8 reduced rank estimators:

1) Brr, a reduced rank version of Bhat weighted by its covariance matrix using \
the steepest descent algorithm in Manton et al. (2003)
2) Bsvd, the truncated singular value decomposition of Bhat
3) Ysvd, the truncated singular value decomposition of the fitted values Yhat
4) Bkron, a weighted truncated singular value decomposition using Crow, \
	and Ccol, where kron(Ccol, Crow) is the best Kronecker approximation \
	to the covariance matrix of Bhat
5) Bblock, a reduced rank version of Bhat weighted by the block \
diagonal components of its covariance matrix, using steepest descent
6) Yrr, classical reduced rank (OLS) regression on the \
fitted values with estimator weighted by residual matrix inner product.
7) Yresid, truncated svd of Yhat weighted by rrhat = (Y-Yhat)'*(Y-Yhat)
8) Ytr, a truncated singular value decomposition of Yhat weighted by \
C_trace, which is formed by taking the trace of blocks of \
the covariance matrix of Bhat

# See also 
[`sim_gee`](@ref), [`mgeedata`](@ref)
"""
function mgee_rr(nobs::Integer, ngroup::Integer,
				y::AbstractMatrix, Xm::AbstractMatrix, 
				Xv::AbstractMatrix, Xr::AbstractMatrix,
				rank::Integer, Bm::AbstractMatrix, 
				C::AbstractMatrix, CovB::AbstractMatrix,
				xlog::IOStream;
				rng=StableRNG(1))

 N = nobs*ngroup
 m = size(y, 2)
 pm = size(Xm, 2)
 pv = size(Xv, 2)
 write(xlog, "Generating data... \n")
 #y, Bm, Bv, Xm, Xv, Xr, g, C = mgeedata(nobs, ngroup, m, pm, pv, Bm;
 #					      rng=rng, err_method=err_method)

 #y = generate_gee2_response(nobs, ngroup, Xm, Bm, C; rng=rng)

 # Estimate mean coefficents and covariance matrix
 Bhat, mc = dense_gee_estimates(Xm, Xv, Xr, y, g, xlog)

 # Compare estimated covariance matrix to population covariance matrix
 covdist = sqrt(sum((C-mc).^2))

 write(xlog, "Computing Full WLRA... \n")
 # Compute WLRA to Bhat using the estimated covariance of Bhat
 Brr, Nrr = rr_sd(Bhat, mc, pm, m, rank; tol=1e-8)
 
 # Compute truncated singular value decomposition of Bhat
 Bsvd = tsvd_estimate(Bhat, rank)

 # Compute truncated singular value decomposition of fitted values
 Yhat = Xm*Bhat
 Ysvd = tsvd_estimate(Yhat, rank)

 # Compute classical RR estimator
 Yrr = rr_ols(y, Xm, rank)

 # Compute residual weighted GEE based estimator
 Yresid = rr_resid(y, Yhat, rank)

 write(xlog, "Computing Kronecker approximation... \n")
 Bkron = kron_estimate(Bhat, mc, rank)

 write(xlog, "Computing Block WLRA... \n")
 # Compute best WLRA based on Cblock
 Bblock, Nblock = block_estimate(Bhat, mc, rank)

 # Compute truncated SVD of Yhat weighted by 
 # block-wise trace of mc
 Ytr = trace_estimate(Yhat, mc, pm, rank)

 write(xlog, "All estimators computed. \n\n")
 # Compare distance of estimators to true coefficient matrix via Frobenius norm
 R1 = frobenius_distance(Bhat,Bm)
 R2 = frobenius_distance(Brr,Bm)
 R3 = frobenius_distance(Bsvd,Bm)
 R4 = frobenius_distance((Xm \ Ysvd),Bm)
 R5 = frobenius_distance(Bkron,Bm)
 R6 = frobenius_distance(Bblock,Bm)
 R7 = frobenius_distance((Xm \ Yrr),Bm)
 R8 = frobenius_distance((Xm \ Yresid),Bm)
 R9 = frobenius_distance((Xm \ Ytr),Bm)

 # Compare distance of estimators to mean of response via Frobenius norm
 F1 = frobenius_distance(Xm*Bhat, Xm*Bm)
 F2 = frobenius_distance(Xm*Brr, Xm*Bm)
 F3 = frobenius_distance(Xm*Bsvd, Xm*Bm)
 F4 = frobenius_distance(Ysvd, Xm*Bm)
 F5 = frobenius_distance(Xm*Bkron, Xm*Bm)
 F6 = frobenius_distance(Xm*Bblock, Xm*Bm)
 F7 = frobenius_distance(Yrr, Xm*Bm)
 F8 = frobenius_distance(Yresid, Xm*Bm)
 F9 = frobenius_distance(Ytr, Xm*Bm)

 [R1, R2, R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, covdist]
end

"""
    compute_estimates(nobs, ngroup, y, g, Xm, Xv, Xr, rank, xlog)

Compute several reduced rank estimators for the mean coefficient matrix
"""
function compute_estimates(nobs::Integer, ngroup::Integer, 
				y::AbstractMatrix, g::Vector, Xm::AbstractMatrix, 
				Xv::AbstractMatrix, Xr::AbstractMatrix,
				rank::Integer, make_rcov, xlog::IOStream;
				link_mean=IdentityLink(), link_scale=LogLink(),
				link_cor=SigmoidLink(-1,1),
				rng=StableRNG(1))

 N = nobs*ngroup
 m = size(y, 2)
 pm = size(Xm, 2)
 pv = size(Xv, 2)
 write(xlog, "Generating data... \n")

 # Estimate mean coefficents and covariance matrix
 Bhat, mc = dense_gee_estimates(Xm, Xv, Xr, y, g, make_rcov, xlog;
						link_mean=link_mean,
						link_scale=link_scale,
						link_cor=link_cor)

 write(xlog, "Computing Full WLRA... \n")
 # Compute WLRA to Bhat using the estimated covariance of Bhat
 Brr, Nrr = rr_sd(Bhat, mc, pm, m, rank; tol=1e-8)
 
 # Compute truncated singular value decomposition of Bhat
 Bsvd = tsvd_estimate(Bhat, rank)

 # Compute truncated singular value decomposition of fitted values
 Yhat = Xm*Bhat
 Ysvd = tsvd_estimate(Yhat, rank)

 # Compute classical RR estimator
 Yrr = rr_ols(y, Xm, rank)

 # Compute residual weighted GEE based estimator
 Yresid = rr_resid(y, Yhat, rank)

 write(xlog, "Computing Kronecker approximation... \n")
 Bkron = kron_estimate(Bhat, mc, rank)

 write(xlog, "Computing Block WLRA... \n")
 # Compute best WLRA based on Cblock
 Bblock, Nblock = block_estimate(Bhat, mc, rank)

 # Compute truncated SVD of Yhat weighted by 
 # block-wise trace of mc
 Ytr = trace_estimate(Yhat, mc, pm, rank)

 write(xlog, "All estimators computed. \n\n")
 Bhat, Brr, Bsvd, Ysvd, Bkron, Bblock, Yrr, Yresid, Ytr, mc
end

"""
    compute_metrics(estimators, Xm, Bm, CovB)

Compute Frobenius distance
for several reduced rank estimators of the mean coefficient matrix
"""
function compute_metrics(estimators, Xm, Bm, CovB)
# Unpack estimators
 Bhat = estimators[1]
 Brr = estimators[2]
 Bsvd = estimators[3]
 Ysvd = estimators[4]
 Bkron = estimators[5]
 Bblock = estimators[6]
 Yrr = estimators[7]
 Yresid = estimators[8]
 Ytr = estimators[9]
 mc = estimators[10]

 # Compare estimated covariance matrix to population covariance matrix
 covdist = sqrt(sum((CovB-mc).^2))

 # Compare distance of estimators to true coefficient matrix via Frobenius norm
 R1 = frobenius_distance(Bhat,Bm)
 R2 = frobenius_distance(Brr,Bm)
 R3 = frobenius_distance(Bsvd,Bm)
 R4 = frobenius_distance((Xm \ Ysvd),Bm)
 R5 = frobenius_distance(Bkron,Bm)
 R6 = frobenius_distance(Bblock,Bm)
 R7 = frobenius_distance((Xm \ Yrr),Bm)
 R8 = frobenius_distance((Xm \ Yresid),Bm)
 R9 = frobenius_distance((Xm \ Ytr),Bm)

 # Compare distance of estimators to mean of response via Frobenius norm
 F1 = frobenius_distance(Xm*Bhat, Xm*Bm)
 F2 = frobenius_distance(Xm*Brr, Xm*Bm)
 F3 = frobenius_distance(Xm*Bsvd, Xm*Bm)
 F4 = frobenius_distance(Ysvd, Xm*Bm)
 F5 = frobenius_distance(Xm*Bkron, Xm*Bm)
 F6 = frobenius_distance(Xm*Bblock, Xm*Bm)
 F7 = frobenius_distance(Yrr, Xm*Bm)
 F8 = frobenius_distance(Yresid, Xm*Bm)
 F9 = frobenius_distance(Ytr, Xm*Bm)

 [R1, R2, R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, covdist]
end

"""
    sim_gee(nobs, ngroup, m, pm, pv, rank, xlog; nsim=100, rng=StableRNG(1))

Simulates the performance of several GEE2-based
reduced rank estimators of the mean coefficients.

The mean coefficient matrix is generated with the given rank, and the
estimators are evaluated in terms of the Frobenius distance to the
true mean coefficient matrix and the mean response matrix.

# Arguments
- `nobs::Integer`: the number of independent clusters
- `ngroup::Integer`: the number of observations in a cluster
- `m::Integer`: the number of response variables
- `pm::Integer`: the number of mean model covariates
- `pv::Integer`: the number of scale model covariates
- `rank::Integer`: the rank of the mean model coefficient matrix
- `xlog::IOStream`: the IOStream for progress logging
- `nsim::Integer`: the number of simulation iterations to run. Default is 100.
- `rng::StableRNG`: stable random number generation stream. \
                    Default is StableRNG(1)

# See also 
[`mgee_rr`](@ref), [`mgeedata`](@ref)
"""
function sim_gee(nobs::Integer, ngroup::Integer, m::Integer, 
				pm::Integer, pv::Integer, 
				rank::Integer, xlog::IOStream;
				nsim=100, rng=StableRNG(1),
				err_method="general")

 scenario_parameters = (nobs, ngroup, m, pm, pv, rank, nsim, err_method)

 R = zeros(Float64, nsim, 18)
 Rcov = zeros(Float64, nsim)
 # Coefficients for mean model
 Bm = genrr(pm, m, rank; rng=rng)

# Define link functions for GEE2 model
mean_link=IdentityLink()
scale_link=LogLink()
cor_link = SigmoidLink(-1,1)

make_rcov = function(err_method)
	if err_method == "additive"
		return function(x1, x2)
			return [1, abs(x1[1]-x2[1])]
		end
	elseif err_method == "space_time"
		return function(x1, x2)
			return [1, (x1[1]-x2[1])^2]
		end
	else
		return function(x1, x2)
			return [1, abs(x1[1]-x2[1])]
		end
	end
end

 # Covariance matrix for a cluster of size ngroup x m
 if err_method == "additive"
	parms = (0.9, range(0.1, 0.9, length=m), range(1,m*ngroup,m*ngroup))
 elseif err_method == "space_time"
	parms = (2, 3, 1, range(1,m*ngroup,m*ngroup))
	cor_link = StableInverseLink()
 else
	parms = range(1,m*ngroup,m*ngroup)
 end
 C = generate_cluster_cov(ngroup, m; err_method=err_method, 
							parms=parms, rng=rng)

 # Generate design matrices
 Xm, Xv, Xr = generate_gee2_covariates(nobs, ngroup, pm, pv; rng=rng)

 # Construct theoretical covariance of vec(Bhat)
 CovB = generate_cov_bhat(nobs, ngroup, m, Xm, C)

 true_coefficients = (Bm, C, CovB)

 covariates = (Xm, Xv, Xr)

 low_rank_results = []

 grouping = repeat(1:nobs, inner=ngroup)

 write(xlog, "Beginning simulation: \n")
 for i in 1:nsim
 	write(xlog, string("Iteration #", i, ":\n"))
	y = generate_gee2_response(nobs, ngroup, Xm, Bm, C; rng=rng)
 	estimators = compute_estimates(nobs, ngroup, y, grouping, Xm, Xv, Xr, rank,
									 make_rcov(err_method), xlog; link_mean = mean_link,
									 link_scale = scale_link,
									 link_cor = cor_link,
									 rng=rng)
	push!(low_rank_results, (i,estimators...))
	a = compute_metrics(estimators, Xm, Bm, CovB)
	R[i,:] = a[1:18]
	Rcov[i] = a[19]
	flush(xlog)
 end
 @save "artifacts/estimators/sim_gee2_$(nobs)_$(ngroup)_$(m)_$(err_method)_$(rank).jld2" scenario_parameters true_coefficients covariates low_rank_results
 R, Rcov, Bm, C, CovB
end