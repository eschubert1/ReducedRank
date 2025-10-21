# Generate covariance matrices for simulations
using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra
using Distributions

"""
    gencov(n)

Generate a random n x n positive definite covariance matrix.
Scale uniformly multiplies the eigenvalues which are generated from U(0,1)
"""
function gencov(n::Integer; scale=1, rng=StableRNG(1))
 Q = randorth(n,n; rng=rng)
 normalize!.(eachcol(Q))
 S = diagm(scale.*rand(rng, Float64, n))
 Symmetric(Q*S*Q')
end

"""
    generate_cluster_cov(ngroup, m)

Generate a covariance matrix for a cluster of size ngroup x m.
"""
function generate_cluster_cov(ngroup, m; err_method="additive", parms=nothing, rng=StableRNG(1))
	if err_method == "additive"
		# Additive correlation structure
		f, rr, h = parms
		C = additive_cov(ngroup, m, f, rr, h)
	elseif err_method == "space_time"
		# Spatio-temporal correlation structure
		s_ix = 1:ngroup
		t_ix = 1:m
		a, b, s2, h = parms
		p = (a, b, s2)
		C = space_time_cov(s_ix, t_ix; parms=p)
		D = Diagonal(sqrt.(h))
		C = D*C*D
	else
		h = parms
		H = Diagonal(sqrt.(h))
		C = gencov(ngroup*m; rng=rng)
		# Standardize to correlation matrix
		D = Diagonal(1 ./ sqrt.(diag(C)))
		C = D*C*D
		# Scale by H
		C = H*gencov(ngroup*m; rng=rng)*H
	end
	C
end

"""
    generate_errors(nobs, ngroup, m, C)

Generate multivariate normal errors with covariance matrix C for nobs clusters.
"""
function generate_errors(nobs, ngroup, m, C; rng=StableRNG(1))
 N = nobs*ngroup
 E = zeros(N, m)
 L = cholesky(Symmetric(C)).L
 for i in 1:nobs
	errors = L*randn(rng, ngroup*m)
	E[((i-1)*ngroup+1):(i*ngroup), :] = reshape(errors, ngroup, m)
 end
 E
end

"""
	generate_cov_bhat(nobs, ngroup, m, Xm, C)

Construct the theoretical covariance matrix of vec(Bhat)
"""
function generate_cov_bhat(nobs, ngroup, m, Xm, C)
 N = nobs*ngroup

 pm = size(Xm, 2)
 B = zeros(pm*m, pm*m) # Bread matrix
 M = zeros(pm*m, pm*m) # Meat matrix

 for i in 1:nobs
 	ii = (ngroup*(i-1)+1):(ngroup*i)
	for j in 1:m
		jj = (ngroup*(j-1)+1):(ngroup*j)
		xx = (pm*(j-1)+1):(pm*j)
		A = Xm[ii,:]'*inv(C[jj,jj])*Xm[ii,:]
		B[xx,xx] = B[xx,xx] + A
		M[xx,xx] = M[xx,xx] + A
		for k in (j+1):m
			kk = (ngroup*(k-1)+1):(ngroup*k)
			yy = (pm*(k-1)+1):(pm*k)
			M[xx,yy] = (M[xx,yy] + 
				Xm[ii,:]'*inv(C[jj,jj])*C[jj,kk]*inv(C[kk,kk])*Xm[ii,:])
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

 a, b, s2 = parms

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
    additive_cov(ngroup, m, f, rr, h)

Generate additive covariance matrix with both row and column correlations.
"""
function additive_cov(ngroup, m, f, rr, h)
 # Generate additive errors with both row and column correlations.
 # The strength of the
 # correlations is controlled by the parameter 'f'.
 
 V = Diagonal(sqrt.(h))
 R = zeros(ngroup*m, ngroup*m)
 for i in 1:m
 	ii = (ngroup*(i-1)+1):(ngroup*i)
	M = rr[i].*ones(ngroup, ngroup)
	M[diagind(M)] .= 1
	R[ii,ii] = M
	for j in (i+1):m
		jj = (ngroup*(j-1)+1):(ngroup*j)
		M = sqrt(rr[i]*rr[j]).*ones(ngroup,ngroup)
		M[diagind(M)] .= sqrt(rr[i]*rr[j]) + sqrt((1-rr[i])*(1-rr[j]))
		M = f.*M
		R[ii,jj] = M
		R[jj,ii] = M
	end
 end

 C = V*R*V
 C
end

"""
    additive_cor_errors(nobs, ngroup, m, Xm, Vy, f, rr)

Generate errors with row-wise and column-wise errors with its associated 
covariance matrix
"""
function additive_cor_errors(nobs, ngroup, m, Xm, Vy, f, rr, h; rng=StableRNG(1))
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
