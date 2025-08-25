using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra
using Distributions

include("rr_sims.jl")

# Generate data for a multivariate GEE with given mean coefficients
function mgeedata(nobs, ngroup, m, pm, pv, rank;rng=StableRNG(1))
 N = nobs*ngroup

 # Residual correlation for each outcome, each element sampled from U(-1,1)
 rr = 2 .* rand(rng, m) .- 1

 # Coefficients for mean model
 Bm = genrr(m, p, r)
 # Coefficients for scale model
 Bv = randn(rng, pv, m)

 g = repeat(1:nobs, inner=ngroup)
 Xm = randn(rng, N, pm)
 Xv = randn(rng, N, pv)
 Xv[:, 1] .= 1
 Xr = randn(rng, N, 2)

 Ey = Xm * Bm
 Vy = exp.(Xv * Bv)
 clamp!(Vy, 0.1, 20)

 # Generate additive errors with both row and column correlations.  The strength of the
 # correlations is controlled by the parameter 'f'.
 f = 0.5
 E = zeros(N, m)
 v = repeat(randn(nobs), inner=ngroup)
 ee = randn(N)
 for j in 1:m
    u = repeat(randn(nobs), inner=ngroup)
    u = sqrt(f)*v + sqrt(1-f)*u
    e = sqrt(f)*ee + sqrt(1-f)*randn(N)
    E[:, j] = sqrt.(Vy[:, j]) .* (sqrt(rr[j])*u + sqrt(1-rr[j])*e)
 end

 y = Ey + E

 y, Bm, Bv, Xm, Xv, Xr, g
end

# Generate data for a multivariate GEE with reduced rank mean coefficients
function mgeedata(nobs, ngroup, m, pm, pv, rank, Bm;rng=StableRNG(1), err_method=0)
 N = nobs*ngroup

 # Residual correlation for each outcome, spaced equally between 0.1 and 0.9
 rr = range(0.1, 0.9, length=m)

 # Coefficients for scale model
 Bv = rand(rng, pv, m)

 g = repeat(1:nobs, inner=ngroup)
 
 # Generate mean model design matrix with correlated columns
 rm = 0.5625 # correlation parameter
 X1 = randn(rng, N)
 Xm = reshape(repeat(X1, pm-2), N, pm-2) + randn(rng, N, pm-2)*sqrt(rm)
 Xm = hcat(ones(N), X1, Xm)

 # Generate scale model design matrix
 Xv = randn(rng, N, pv)
 Xv[:, 1] .= 1
 Xr = randn(rng, N, 2)

 Ey = Xm * Bm
 Vy = exp.(Xv * Bv)
 clamp!(Vy, 0.1, 10)

 if err_method == 1
 	groups = repeat(1:nobs, inner=ngroup)
	E = clustered_errors(groups, m)
 else
 	# Generate additive errors with both row and column correlations.  The strength of the
 	# correlations is controlled by the parameter 'f'.
 	f = 0.5
 	E = zeros(N, m)
 	v = repeat(randn(nobs), inner=ngroup)
 	ee = randn(N)
 	for j in 1:m
    		u = repeat(randn(nobs), inner=ngroup)
    		u = sqrt(f)*v + sqrt(1-f)*u
    		e = sqrt(f)*ee + sqrt(1-f)*randn(N)
    		E[:, j] = sqrt.(Vy[:, j]) .* (sqrt(rr[j])*u + sqrt(1-rr[j])*e)
 	end
 end

 y = Ey + E

 y, Bm, Bv, Xm, Xv, Xr, g
end

# Generate clustered errors with cluster membership defined by groups, and m is the number of response variables
function clustered_errors(groups, m)
 N = length(groups)
 group_ids = unique(groups)
 E = zeros(N, m)
 for i in group_ids
 	ii = findall(j->(j==i), groups)
 	ng = length(ii)
 	C = Symmetric(gencov(ng*m))
	A = cholesky(C)
 	Eg = A.L*randn(ng*m)
	Eg = reshape(Eg, (ng, m))
	E[ii, :] = Eg
 end
 E
end

# Determine Singular Angle Similarity between two matrices, A and B, of the same size
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

# Fit multivariate response GEE2 (or GEE1) model and construct reduced rank estimator via WLRA.
function mgee_rr(nobs, ngroup, m, pm, pv, rank, Bm, xlog; rng=StableRNG(1))
 N = nobs*ngroup
 write(xlog, "Generating data... \n")
 y, Bm, Bv, Xm, Xv, Xr, g = mgeedata(nobs, ngroup, m, pm, pv, rank, Bm; rng=rng)

 function make_rcov(x1, x2)
    return [1., x1[2]+x2[2], abs(x1[2]-x2[2])]
 end
 
 mm = Array{Union{Nothing, GeneralizedEstimatingEquations2Model}}(nothing, m)
 
 write(xlog, "Fitting GEE2 models... \n")
 for j in 1:m
 	mm[j] = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, j], g, make_rcov; verbosity=0, cor_pair_factor=0.0)
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

 write(xlog, "Computing Full WLRA... \n")
 # Compute WLRA to Bhat using the estimated covariance of Bhat
 Brr, Nrr = rr_sd(Bhat, mc, pm, m, rank; tol=1e-8)
 
 # Compute truncated singular value decomposition of Bhat
 Bsvd = tsvd(Bhat, rank)
 Bsvd = Bsvd.U*Diagonal(Bsvd.S)*Bsvd.Vt

 # Compute truncated singular value decomposition of fitted values
 Ysvd = tsvd(Xm*Bhat, rank)
 Ysvd = Ysvd.U*Diagonal(Ysvd.S)*Ysvd.Vt

 write(xlog, "Computing Kronecker approximation... \n")
 # Compute Kronecker approximation to estimated covariance matrix of Bhat
 Ccol, Crow = kronapprox(mc, float.(Matrix(I, pm, pm)), m, m, pm, pm)
 # Determine low-rank approximation weighted by estimated  row and column covariances
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

 # Compute truncated SVD of Bhat weighted by Ctrace
 Ctr_1 = sqrt(inv(Ctrace))
 Btr = tsvd(Bhat*Ctr_1, rank)
 Btr = Btr.U*Diagonal(Btr.S)*Btr.Vt*sqrt(Ctrace)

 # Compute truncated SVD of Yhat weighted by Ctrace
 Ytr = tsvd(Xm*Bhat*Ctr_1, rank)
 Ytr = Ytr.U*Diagonal(Ytr.S)*Ytr.Vt*sqrt(Ctrace)

 write(xlog, "All estimators computed. \n\n")
 # Compare distance of estimators to true coefficient matrix via Frobenius norm
 R1 = sqrt(sum((Bhat-Bm).^2))
 R2 = sqrt(sum((Brr-Bm).^2))
 R3 = sqrt(sum((Bsvd-Bm).^2))
 R4 = sqrt(sum(((Xm \ Ysvd)-Bm).^2))
 R5 = sqrt(sum((Bkron-Bm).^2))
 R6 = sqrt(sum((Bblock-Bm).^2))
 R7 = sqrt(sum((Btr-Bm).^2))
 R8 = sqrt(sum(((Xm \ Ytr)-Bm).^2))

 # Compare distance of estimators to mean of response via Frobenius norm
 F1 = sqrt(sum((Xm*(Bhat-Bm)).^2))
 F2 = sqrt(sum((Xm*(Brr-Bm)).^2))
 F3 = sqrt(sum((Xm*(Bsvd-Bm)).^2))
 F4 = sqrt(sum((Ysvd - Xm*Bm).^2))
 F5 = sqrt(sum((Xm*(Bkron-Bm)).^2))
 F6 = sqrt(sum((Xm*(Bblock-Bm)).^2))
 F7 = sqrt(sum((Xm*(Btr-Bm)).^2))
 F8 = sqrt(sum((Ytr - Xm*Bm).^2))

 # Compare differences of estimators using singular angle similarity
 #R1 = sas(Bhat, Bm)
 #R2 = sas(Brr, Bm)
 #R3 = sas(Bsvd, Bm)

 [R1, R2, R3, R4, R5, R6, R7, R8, F1, F2, F3, F4, F5, F6, F7, F8]
 #Bm, Bhat, Bkron, mc, Crow, Ccol
end

function sim_gee(nobs, ngroup, m, pm, pv, rank, xlog;nsim=100, rng=StableRNG(1))
 R = zeros(Float64, nsim, 16)
 # Coefficients for mean model
 Bm = genrr(pm, m, rank)
 write(xlog, "Beginning simulation: \n")
 for i in 1:nsim
 	write(xlog, string("Iteration #", i, ":\n"))
 	a = mgee_rr(nobs, ngroup, m, pm, pv, rank, Bm, xlog; rng=rng)
	R[i,:] = a
 end
 write(xlog, string("Finished ", nsim, " iterations. \n"))
 #out = round.(mean(R, dims=1); digits=8)
 #println("Frobenius distance to true coefficient matrix:")
 #println(["Dense GEE2", "Full WLRA", "Bhat SVD", "Yhat SVD", "Bhat KA", "Block WLRA", "Bhat CW", "Yhat CW"])
 #println(out[1:8])
 #println("Frobenius distance to mean of response:")
 #println(["Dense GEE2", "Full WLRA", "Bhat SVD", "Yhat SVD", "Bhat KA", "Block WLRA", "Bhat CW", "Yhat CW"])
 #println(out[9:16])
 R, Bm
end
