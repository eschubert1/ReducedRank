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

 # Generate mean model design matrix with correlated columns
 rm = 0.5625 # correlation parameter
 X1 = randn(rng, N)
 X2 = reshape(repeat(X1, pm-2), N, pm-2) + randn(rng, N, pm-2)*sqrt(rm)
 Xm = hcat(ones(N), X1, X2)
 	
 h = range(1, 5, m) # heteroscedasticity parameter

 # Generate scale model design matrix
 Xv = randn(rng, N, pv)
 Xv[:, 1] .= 1
 Xr = hcat(ones(N), randn(rng, N))

 Ey = Xm*Bm

 Vy = exp.(Xv * Bv)
 clamp!(Vy, 0.1, 10)
 Vy = ones(N, m) # Remove scale model, heteroscedasticity through h only

 if err_method == 1
	E, C = clustered_errors(groups, m, Xm)
 else
 	# Generate additive errors with both row and column correlations.  
	# The strength of the
 	# correlations is controlled by the parameter 'f'.
 	f = 0.9
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
 end

 y = Ey + E

 y, Bm, Bv, Xm, Xv, Xr, groups, C
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
    clustered_errors(groups, m)

Generate clustered errors with cluster membership defined by groups.
m is the number of response variables.

If a cluster Y is of size n x m, the generated covariance matrix corresponds to
Cov(vec(Y)) of size nm x nm
"""
function clustered_errors(groups, m, Xm)
 N = length(groups)
 group_ids = unique(groups)
 E = zeros(N, m)
 pm = size(Xm, 2)
 B = zeros(pm*m, pm*m)
 M = zeros(pm*m, pm*m)
 for i in group_ids
 	ii = findall(j->(j==i), groups)
 	ng = length(ii)
	C = Symmetric(gencov(ng*m))
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
 	Eg = A.L*randn(ng*m)
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
				rng=StableRNG(1))
 N = nobs*ngroup
 write(xlog, "Generating data... \n")
 y, Bm, Bv, Xm, Xv, Xr, g, C = mgeedata(nobs, ngroup, m, pm, pv, Bm; rng=rng)

 N = ngroup*nobs

 function make_rcov(x1, x2)
    return [1., x1[2]+x2[2], abs(x1[2]-x2[2])]
 end
 
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

 # Compare estimated covariance matrix to population covariance matrix
 # println(sqrt(sum((C-mc).^2)))

 write(xlog, "Computing Full WLRA... \n")
 # Compute WLRA to Bhat using the estimated covariance of Bhat
 Brr, Nrr = rr_sd(Bhat, mc, pm, m, rank; tol=1e-8)
 
 # Compute truncated singular value decomposition of Bhat
 Bsvd = tsvd(Bhat, rank)
 Bsvd = Bsvd.U*Diagonal(Bsvd.S)*Bsvd.Vt

 # Compute truncated singular value decomposition of fitted values
 Yhat = Xm*Bhat
 Ysvd = tsvd(Yhat, rank)
 Ysvd = Ysvd.U*Diagonal(Ysvd.S)*Ysvd.Vt

 # Compute classical RR estimator
 Yols = Xm*inv(Xm'*Xm)*Xm'*y
 rrhat = (y-Yols)'*(y-Yols) / N
 Yrr = tsvd(Yols*sqrt(inv(rrhat)), rank)
 Yrr = Yrr.U*Diagonal(Yrr.S)*Yrr.Vt*sqrt(rrhat)

 # Compute residual weighted GEE based estimator
 rrhat = (y-Yhat)'*(y-Yhat) / N
 Yresid = tsvd(Yhat*sqrt(inv(rrhat)), rank)
 Yresid = Yresid.U*Diagonal(Yresid.S)*Yresid.Vt*sqrt(rrhat)

 write(xlog, "Computing Kronecker approximation... \n")
 # Compute Kronecker approximation to estimated covariance matrix of Bhat
 Ccol, Crow = kronapprox(mc, float.(Matrix(I, pm, pm)), m, m, pm, pm)
 # Determine low-rank approximation weighted by estimated 
 # row and column covariances
 Bkron = rrkron(Bhat, Crow, Ccol, rank)

 # Compute block covariance matrix of Bhat
 Cblock = zeros(m*pm, m*pm)
 for i in 1:m
 	inds = (1+pm*(i-1)):(pm*i)
	Cblock[inds, inds] = mc[inds, inds]
 end

 write(xlog, "Computing Block WLRA... \n")
 # Compute best WLRA based on Cblock
 Bblock, Nblock = rr_sd(Bhat, Cblock, pm, m, rank; tol=1e-8)

 # Estimate column covariance by taking trace of blocks of covariance of Bhat
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
 Ytr = tsvd(Yhat*Ctr_1, rank)
 Ytr = Ytr.U*Diagonal(Ytr.S)*Ytr.Vt*sqrt(Ctrace)

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

 [R1, R2, R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9]
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
				nsim=100, rng=StableRNG(1))
 R = zeros(Float64, nsim, 18)
 # Coefficients for mean model
 Bm = genrr(pm, m, rank; rng=rng)
 write(xlog, "Beginning simulation: \n")
 for i in 1:nsim
 	write(xlog, string("Iteration #", i, ":\n"))
 	a = mgee_rr(nobs, ngroup, m, pm, pv, rank, Bm, xlog; rng=rng)
	R[i,:] = a
	flush(xlog)
 end
 R, Bm
end
