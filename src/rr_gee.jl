using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra
using Distributions

include("rr_sims.jl")

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
    additive_cor(nobs, ngroup, m, Xm, Vy, f, rr)

Generate errors with row-wise and column-wise errors with its associated 
covariance matrix
"""
function additive_cor(nobs, ngroup, m, Xm, Vy, f, rr, h; rng=StableRNG(1))
 N = nobs*ngroup

 # Generate additive errors with both row and column correlations.
 # The strength of the
 # correlations is controlled by the parameter 'f'.
 
 E = zeros(N, m)
 v = repeat(randn(rng, nobs), inner=ngroup)
 ee = randn(rng, N)
 for j in 1:m
 	u = repeat(randn(rng, nobs), inner=ngroup)
  	u = sqrt(f)*v + sqrt(1-f)*u
        e = sqrt(f)*ee + sqrt(1-f)*randn(rng, N)
        E[:, j] = (sqrt.(Vy[:, j]) .* (sqrt(rr[j])*u +
                        sqrt(1-rr[j])*e)*sqrt(h[j]))
 end
 V = Vy*Diagonal(h)
 C = _build_cov(V, rr, f, Xm, nobs, ngroup, m)
 E, C
end

function _build_cov(Vy, rr, f, Xm, nobs, ngroup, m)
 # Construct the theoretical covariance matrix of vec(Bhat)
 R = zeros(ngroup*m, ngroup*m)
 for i in 1:m
 	ii = (ngroup*(i-1)+1):(ngroup*i)
	M = rr[i].*ones(ngroup, ngroup)
	M[diagind(M)] .= 1
	R[ii,ii] = M
	for j in (i+1):m
		jj = (ngroup*(j-1)+1):(ngroup*j)
		M = sqrt(rr[i]*rr[j]).*ones(ngroup,ngroup)
		M[diagind(M)] .= 1
		M = f.*M
		R[ii,jj] = M
		R[jj,ii] = M
	end
 end

 pm = size(Xm, 2)
 B = zeros(pm*m, pm*m) # Bread matrix
 M = zeros(pm*m, pm*m) # Meat matrix

 for i in 1:nobs
 	ii = (ngroup*(i-1)+1):(ngroup*i)
	Vd = vec(Vy[ii,:])
	Vd = Diagonal(Vd)
	VRV = sqrt(Vd)*R*sqrt(Vd)
	for j in 1:m
		jj = (ngroup*(j-1)+1):(ngroup*j)
		xx = (pm*(j-1)+1):(pm*j)
		A = Xm[ii,:]'*inv(VRV[jj,jj])*Xm[ii,:]
		B[xx,xx] = B[xx,xx] + A
		M[xx,xx] = M[xx,xx] + A
		for k in (j+1):m
			kk = (ngroup*(k-1)+1):(ngroup*k)
			yy = (pm*(k-1)+1):(pm*k)
			M[xx,yy] = (M[xx,yy] + 
				Xm[ii,:]'*inv(VRV[jj,jj])*VRV[jj,kk]*
				inv(VRV[kk,kk])*Xm[ii,:])
			M[yy,xx] = M[xx,yy]
		end
	end
 end

 C = B \ M / B

 C
end

"""
    _spt_cov(space_lag, time_lag)

Compute a non-separable spatio-temporal covariance function from
Cressie and Huang 1999, Example 3.
"""
function _spt_cov(space_lag, time_lag; a=1, b=1, s2=1)
 d = length(space_lag)
 num = s2*(a^2*time_lag^2 + 1)
 denom = ((a^2*time_lag^2+1)^2 + b^2*(space_lag*space_lag))^((d+1)/2)
 num/denom
end

"""
    space_time_cov(s_ix, t_ix)

Compute space time covariance at specified points.
"""
function space_time_cov(s_ix, t_ix; parms)
 
 ns = length(s_ix)
 nt = length(t_ix)
 n = ns*nt
 points = hcat(repeat(s_ix, nt), repeat(t_ix, inner=ns))

 a = parms[1]
 b = parms[2]
 s2 = parms[3]

 # Compute lags and covariance
 space_lags = zeros(n,n)
 time_lags = zeros(n,n)
 covs = zeros(n,n)
 for i in 1:n
 	for j in 1:i
		space_lags[i,j] = abs(points[i,1] - points[j,1])
		time_lags[i,j] = abs(points[i,2] - points[j,2])
		covs[i,j] = _spt_cov(space_lags[i,j], time_lags[i,j];
						      a=a, b=b, s2=s2)
		
		space_lags[j,i] = space_lags[i,j]
		time_lags[j,i] = time_lags[i,j]
		covs[j,i] = covs[i,j]
	end
 end

 covs
end

"""
    generate_errors(groups, m, Xm)

Generate clustered errors with cluster membership defined by groups.
m is the number of response variables.

If a cluster Y is of size n x m, the generated covariance matrix corresponds to
Cov(vec(Y)) of size nm x nm
"""
function generate_errors(groups, m, Xm; cov_method="general", parms=(1,2),
				 	rng=StableRNG(1))
 N = length(groups)
 group_ids = unique(groups)
 E = zeros(N, m)
 pm = size(Xm, 2)
 B = zeros(pm*m, pm*m)
 M = zeros(pm*m, pm*m)
 for i in group_ids
 	ii = findall(j->(j==i), groups)
 	ng = length(ii)
	if cov_method == "spacetime"
		C = Symmetric(space_time_cov(1:ng, 1:m; parms=parms))
	else 
		C = Symmetric(gencov(ng*m))
	end 
	for j in 1:m
		jj = (ng*(j-1)+1):(ng*j)
		xx = (pm*(j-1)+1):(pm*j)
		Z = Xm[ii,:]'*inv(C[jj,jj])*Xm[ii,:]
		B[xx,xx] = B[xx,xx] + Z
		M[xx,xx] = M[xx,xx] + Z
		for k in (j+1):m
			kk = (ng*(k-1)+1):(ng*k)
			yy = (pm*(k-1)+1):(pm*k)
                        M[xx,yy] = (M[xx,yy] + Xm[ii,:]'*inv(C[jj,jj])*
					     C[jj,kk]*inv(C[kk,kk])*Xm[ii,:])
                        M[yy,xx] = M[xx,yy]
		end
	end
	A = cholesky(C)
 	Eg = A.L*randn(rng, ng*m)
	Eg = reshape(Eg, (ng, m))
	E[ii, :] = Eg
 end
 Cov_B = B \ M / B
 E, Cov_B
end

"""
    sas(A, B)

Computes the Singular Angle Similarity between two matrices, A and B, 
of the same size.

Singular Angle Similarity is defined in Albers et al. (2024), available 
at https://arxiv.org/abs/2403.17687.
"""
function sas(A, B)
 Asvd = svd(A)
 Bsvd = svd(B)
 m = min(size(A)[1], size(A)[2])
 left1 = zeros(m)
 right1 = zeros(m)
 left2 = zeros(m)
 right2 = zeros(m)
 angles = zeros(m)
 weights = zeros(m)
 for i in 1:m
 	left1[i] = acos(clamp(Asvd.U[:, i]⋅Bsvd.U[:,i], -1, 1))
	right1[i] = acos(clamp(Asvd.V[:, i]⋅Bsvd.V[:,i], -1, 1))
	left2[i] = acos(clamp(-Asvd.U[:, i]⋅Bsvd.U[:,i], -1, 1))
	right2[i] = acos(clamp(-Asvd.V[:, i]⋅Bsvd.V[:,i], -1, 1))
 	angles[i] = min((left1[i]+right1[i]), (left2[i]+right2[i]))
	weights[i] = sqrt(Asvd.S[i]*Bsvd.S[i])
 end
 angles = angles ./ 2
 delta = 1 .- angles ./ (pi/2)
 sas = sum(weights .* delta)/sum(weights)
 sas
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
function mgee_rr(nobs::Integer, ngroup::Integer, m::Integer, 
				pm::Integer, pv::Integer, rank::Integer,
				Bm::AbstractMatrix, xlog::IOStream;
				rng=StableRNG(1), err_method=0)
 N = nobs*ngroup
 write(xlog, "Generating data... \n")
 y, Bm, Bv, Xm, Xv, Xr, g, C = mgeedata(nobs, ngroup, m, pm, pv, Bm;
					      rng=rng, err_method=err_method)

 N = ngroup*nobs

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
 R1 = sqrt(sum((Bhat-Bm).^2))
 R2 = sqrt(sum((Brr-Bm).^2))
 R3 = sqrt(sum((Bsvd-Bm).^2))
 R4 = sqrt(sum(((Xm \ Ysvd)-Bm).^2))
 R5 = sqrt(sum((Bkron-Bm).^2))
 R6 = sqrt(sum((Bblock-Bm).^2))
 R7 = sqrt(sum(((Xm \ Yrr)-Bm).^2))
 R8 = sqrt(sum(((Xm \ Yresid)-Bm).^2))
 R9 = sqrt(sum(((Xm \ Ytr)-Bm).^2))

 # Compare distance of estimators to mean of response via Frobenius norm
 F1 = sqrt(sum((Xm*(Bhat-Bm)).^2))
 F2 = sqrt(sum((Xm*(Brr-Bm)).^2))
 F3 = sqrt(sum((Xm*(Bsvd-Bm)).^2))
 F4 = sqrt(sum((Ysvd - Xm*Bm).^2))
 F5 = sqrt(sum((Xm*(Bkron-Bm)).^2))
 F6 = sqrt(sum((Xm*(Bblock-Bm)).^2))
 F7 = sqrt(sum((Yrr - Xm*Bm).^2))
 F8 = sqrt(sum((Yresid-Xm*Bm).^2))
 F9 = sqrt(sum((Ytr - Xm*Bm).^2))

 [R1, R2, R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, covdist]
end

"""
    dense_gee_estimates(Xm, Xv, Xr, y, g, xlog)

Estimates mean model coefficient matrix Bhat and Cov(vec(Bhat))
"""
function dense_gee_estimates(Xm, Xv, Xr, y, g, xlog)
 
 function make_rcov(x1, x2)
    return [1, abs(x1[2]-x2[2])]
 end

 m = size(y, 2)
 pm = size(Xm, 2)

 mm = Array{Union{Nothing, GeneralizedEstimatingEquations2Model}}(nothing, m)

 write(xlog, "Fitting GEE2 models... \n")
 for j in 1:m
        mm[j] = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, j],
                        g, make_rcov; corstruct_mean=ExchangeableCor(),
                        verbosity=0, cor_pair_factor=0.0)
 end

 write(xlog, "Constructing covariance matrix... \n")
 vc = vcov(mm...)

 b = Array{Union{Nothing, Vector}}(nothing, m)
 for j in 1:m
        b[j] = mm[j].mean_model.pp.beta0
 end

 Bhat = hcat(b...)

 # Subset full covariance matrix to covariances between mean model parameters
 np = Int(div(size(vc[1], 1),m))
 ii = Vector{Union{Nothing,UnitRange{Int64}}}(nothing, m)
 for j in 1:m
  ii[j] = (1:pm) .+ np*(j-1)
 end
 inds = vcat(ii...)
 mc = vc[1][inds,inds]

 Bhat, mc
end

"""
    tsvd_estimate(X, rank)

Construct a reduced rank estimate of X via truncated SVD
"""
function tsvd_estimate(X, rank)
 Xsvd = tsvd(X, rank)
 Xsvd = Xsvd.U*Diagonal(Xsvd.S)*Xsvd.Vt
 Xsvd
end

"""
    rr_ols(y, Xm, rank)

Compute classical RR estimate based on OLS
"""
function rr_ols(y, Xm, rank)
 N = size(y, 1)
 Yols = Xm*inv(Xm'*Xm)*Xm'*y
 rrhat = (y-Yols)'*(y-Yols) / N
 Yrr = tsvd_estimate(Yols*sqrt(inv(rrhat)), rank)
 Yrr*sqrt(rrhat)
end

"""
    rr_resid(y, Yhat, rank)

Compute classical RR estimate based on GEE2 fit and residuals
"""
function rr_resid(y, Yhat, rank)
 N = size(y, 1)
 rrhat = (y-Yhat)'*(y-Yhat) / N
 Yresid = tsvd_estimate(Yhat*sqrt(inv(rrhat)), rank)
 Yresid*sqrt(rrhat)
end

"""
    kron_estimate(Bhat, mc, rank)

Compute Kronecker product approximation to mc = Cov(vec(Bhat))
and then construct reduced rank estimator weighted by the product matrices
"""
function kron_estimate(Bhat, mc, rank)
 m = size(Bhat, 2)
 pm = size(Bhat, 1)
 
 # Compute Kronecker approximation to estimated covariance matrix of Bhat
 Ccol, Crow = kronapprox(mc, float.(Matrix(I, pm, pm)), m, m, pm, pm)

 # Determine low-rank approximation weighted by estimated
 # row and column covariances
 Bkron = rrkron(Bhat, Crow, Ccol, rank)
 Bkron
end

"""
    block_estimate(Bhat, mc, rank)

Compute reduced rank estimator of Bhat weighted by
diagonal blocks of mc = Cov(vec(Bhat))
"""
function block_estimate(Bhat, mc, rank)
 m = size(Bhat, 2)
 pm = size(Bhat, 1)

 # Compute block covariance matrix of Bhat
 Cblock = zeros(m*pm, m*pm)
 for i in 1:m
        inds = (1+pm*(i-1)):(pm*i)
        Cblock[inds, inds] = mc[inds, inds]
 end

 # Compute best WLRA based on Cblock
 Bblock, Nblock = rr_sd(Bhat, Cblock, pm, m, rank; tol=1e-8)
 Bblock, Nblock
end

"""
    trace_estimate(Yhat, mc, pm, rank)

Compute reduced rank estimator of Yhat with columns weighted
by the trace of blocks of mc = Cov(vec(Bhat))
"""
function trace_estimate(Yhat, mc, pm, rank)
 m = size(Yhat, 2)

 Ctrace = zeros(m, m)
 for i in 1:m
        inds1 = (1+pm*(i-1)):(pm*i)
        for j in i:m
                inds2 = (1+pm*(j-1)):(pm*j)
                Ctrace[i, j] = tr(mc[inds1, inds2])
                Ctrace[j, i] = Ctrace[i, j]
        end
 end

 # Compute truncated SVD of Yhat weighted by Ctrace
 Ctr_1 = sqrt(inv(Ctrace))
 Ytr = tsvd_estimate(Yhat*Ctr_1, rank)
 Ytr*sqrt(Ctrace)
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
				err_method=0)
 R = zeros(Float64, nsim, 18)
 Rcov = zeros(Float64, nsim)
 # Coefficients for mean model
 Bm = genrr(pm, m, rank; rng=rng)
 write(xlog, "Beginning simulation: \n")
 for i in 1:nsim
 	write(xlog, string("Iteration #", i, ":\n"))
 	a = mgee_rr(nobs, ngroup, m, pm, pv, rank, Bm, xlog; rng=rng,
			  err_method=err_method)
	R[i,:] = a[1:18]
	Rcov[i] = a[19]
	flush(xlog)
 end
 R, Rcov, Bm
end
