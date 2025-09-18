# Simulations for understanding the performance of weighted reduced 
# rank approximations
using StableRNGs
using LinearAlgebra
using Statistics

"""
    randorth(n, p)

Generate a random n x p orthogonal matrix
When n = p, the generated matrix is distributed according to Haar measure
"""
function randorth(n::Integer,p::Integer; rng=StableRNG(1))
 Q,R = qr(randn(rng, n,p))
 O = Q*Diagonal(sign.(diag(R)))
 O
end

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
    genrr(n, p, r)

Generate a random n x p matrix of rank r < min(n,p)
"""
function genrr(n::Integer,p::Integer,r::Integer; rng=StableRNG(1))
 if r > min(n, p)
 	throw(DomainError(r, "rank must be less than matrix dimensions"))
 elseif r == min(n, p)
 	@warn "Resulting matrix has full rank"
 end
 U = randorth(rng=rng,n,r)
 V = randorth(rng=rng,p,r)
 D = diagm(randn(rng,r))
 U*D*V'
end

"""
    cs(n, scale, rho)

Generate compound symmetry structured covariance matrix of size n
Scale represents the variance (diagonal) terms, 
and rho is the correlation parameter
"""
function cs(n::Integer, scale, rho)
  S = zeros(n,n) .+ rho + diagm(repeat([1-rho], n))
  S = S.*scale
end

"""
    gendat(n, m, p, r, B=nothing, C=nothing)

Generate data and compute OLS estimate and fitted values.
If B and C are not provided, these are generated randomly.

# Arguments
 - `n::Integer`: the sample size
 - `m::Integer`: the number of response variables
 - `p::Integer`: the number of predictors
 - `r::Integer`: the rank of the mean coefficient matrix
 - `B`: the p x m mean parameter coefficient matrix
 - 'C': the pm x pm conditional covariance matrix of the response
"""
function gendat(n::Integer, m::Integer, p::Integer, r::Integer,
			    B=nothing, C=nothing; rng=StableRNG(1))
 X = hcat(ones(n), randn(rng,n, p-1))
 isnothing(B) ? B = genrr(rng,p, m, r) : B=B
 Ey = X*B
 isnothing(C) ? C = Symmetric(gencov(rng,m*n)) : C = Symmetric(C)
 A = cholesky(C)
 E = A.L*randn(rng,n*m)
 Y = Ey + reshape(E, (n,m))
 XY = X'*Y
 Bhat = (X'*X) \ XY
 Yhat = X * Bhat
 Y, X, B, C, Yhat
end

"""
    tsvd(A, r)

Compute truncated singular value decomposition of matrix A at rank r
"""
function tsvd(A::AbstractMatrix, r::Integer)
 if r > minimum(size(A))
 	throw(DomainError(r, "rank must be smaller than matrix dimensions"))
 elseif r == minimum(size(A))
 	@warn "Truncation value is the maximum possible rank"
 end
 C = svd(A)
 Ur = C.U[:,1:r]
 Sr = C.S[1:r]
 Vtr = C.Vt[1:r,:]
 SVD(Ur, Sr, Vtr)
end

"""
    rrkron(Y, Crow, Ccol, r)

Compute best reduced rank r approximation (w.r.t. Frobenius norm) assuming 
Kronecker product structure between (estimated) row and column covariances
Note that the assumed Kronecker product structure is kron(Ccol, Crow)
"""
function rrkron(Y::AbstractMatrix, Crow::AbstractMatrix,
				  Ccol::AbstractMatrix, r::Integer)
 
 Srow = Crow # cov(Y')
 Scol = Ccol # cov(Y)
 # At least one of Srow and Scol will not be invertible.
 # Use Moore-Penrose inverse for both
 Srow_inv = pinv(Srow)
 Scol_inv = pinv(Scol)
 # Compute sqrt matrix of each inverse
 S2row = sqrt(Symmetric(Srow))
 S2col = sqrt(Symmetric(Scol))
 S2row_inv = sqrt(Symmetric(Srow_inv))
 S2col_inv = sqrt(Symmetric(Scol_inv))
 Q = S2row_inv*Y*S2col_inv
 P = tsvd(Q, r)
 Yrr = S2row*P.U*Diagonal(P.S)*P.Vt*S2col
 Yrr
end

"""
    rrdist(F, R, W1, W2)

Compute trace(W2^-1 (F-R)' W1^-1 (F-R))
F is full rank R is reduced rank
"""
function rrdist(F::AbstractMatrix, R::AbstractMatrix,
				   W1::AbstractMatrix, W2::AbstractMatrix)
 W1i = inv(W1)
 W2i = inv(W2)
 tr(W2i*(F-R)'*W1i*(F-R))
end

"""
    kronapprox(A, C0, m1, n1, m2, n2)

Find the best (Frobenius) Kronecker product approximation B, C for matrix A.
B is of size m1 x n1, C of size m2 x n2, A of size m1 m1*m2 x n1*n2
"""
function kronapprox(A::AbstractMatrix, C0::AbstractMatrix, 
				       m1::Integer, n1::Integer,
				       m2::Integer, n2::Integer)
 C = C0
 m = m1*m2
 n = n1*n2
 B = zeros(m1,n1)
 for r in 1:10
	g = tr(C'*C)
  	for i in 1:m1
    		i1 = 1+(i-1)*m2
		i2 = m2*i
    		for j in 1:n1
			j1 = 1+(j-1)*n2
			j2 = n2*j
			Asub = A[i1:i2, j1:j2]
			B[i,j] = tr(C'*Asub)/g
		end
	end
	b = tr(B'*B)
	for i in 1:m2
		i1 = i:m2:m
		for j in 1:n2
			i2 = j:n2:n
			Asub = A[i1,i2]
			C[i,j] = tr(B'*Asub)/b
		end
	end
 end
 B,C
end

"""
    frobenius_bounds(F, W, r)

WARNING: MAY BE INCORRECT
Compute bounds for the minimal value of tr(vec(F-R)' W^-1 vec(F-R)) 
minimized over R, where R is a reduced rank r approximation of F
This minimal value is bounded by the lower and upper singular values 
of W1 multiplied by the sum of the smallest r singular values
"""
function frobenius_bounds(F::AbstractMatrix, W::AbstractMatrix, r::Integer)
	W1 = inv(W)
	Wsvd = svd(W1)
	Fsvd = svd(F)
	a = sum(r->r^2, Fsvd.S[r+1:end])
	s = extrema(Wsvd.S)
	s.*a
end

"""
    kronI(W, N, n; transpose = false)

Efficiently computes the multiplication W*kron(N, I) 
where I is an nxn identity matrix
"""
function kronI(W::AbstractMatrix, N::AbstractMatrix, n::Integer; 
				  transpose = false)
	m = size(W)[1]
	p = size(N)[2]
	if transpose # then compute kron(N', I)*W 
		     # by looping through columns of W
		m2 = size(W)[2]
		A = zeros(n*p, m2)
		for j in 1:p
			for k in 1:n
				inds = k:n:m
				ii = (j-1)*n+k
				for i in 1:m2
					A[ii, i] = dot(W[inds, i],N[:,j])
				end
			end
		end
	else # Loop through rows of W, multiply subset of row by column of N 
		A = zeros(m, n*p)
		for i in 1:m
			for j in 1:p
				for k in 1:n
					inds = k:n:m
					ii = (j-1)*n+k
					A[i,ii] = dot(W[i,inds],N[:,j])
				end
			end
		end
	end
	A
end
			
"""
    rr_sd(X, W, n, m, r; tol=1e-4, verbose=false)

Compute a weighted reduced rank approximation to estimator X via steepest
descent.

This implements Algorithm 11 from Manton, Mahoney, and Hua (2003) 
https://ieeexplore.ieee.org/abstract/document/1166684
minimizing vec(X-R)'*inv(W)*vec(X-R),
where R is the reduced rank approximation

This algorithm applies the rank constraint through the 
null space of R, i.e. by finding an 
orthogonal m x m-r matrix N such that RN = 0.

# Arguments
- `X::Matrix{float}`: the n x m dense estimator
- `W::Matrix{float}`: the nm x nm weight matrix
- `r::Integer`: the target rank of the approximation
"""
function rr_sd(X::AbstractMatrix, W::AbstractMatrix, 
				  n::Integer, m::Integer,
				  r::Integer; tol=1e-4, verbose = false)

 # Determine an initial constraint matrix N 
 # and a m x r matrix Nperp orthogonal to N via SVD of X
 XS = svd(X)
 M = XS.V
 N = M[:,(r+1):end]
 Nperp = M[:,1:r]
 repeat = true
 
 # Count number of iterations and terminate after 1000
 # if algorithm does not converge
 iter = 1
 while repeat
 	lambda = 1 # Set step size for new iteration  
	# Step 2, Compute minimum of loss f(N), 
	# as given by Theorem 1 in IEEE paper
 	
 	P = kronI(W, N, n)
	P = kronI(P, N, n; transpose=true)
	P1 = P \ vec(X*N)
	fN = vec(X*N)' * P1
	if verbose
		println(fN)
	end
 
	# 3, Compute descent direction K and its squared Frobenius norm K2
 	A = reshape(P1, (n, m-r))
 	Q = W*vec(A*N')
 	B = reshape(Q, (n, m))
	K = -2*Nperp'*(X-B)'*A
 	K2 = sum(K.^2)

 	# 4, Evaluate g(N) = f(N + 2*lambda*Nperp*K). 
	# If f(N)-g(N) >= lambda * K2, update lambda = 2*lambda and repeat
	# Terminate loop after 100 iterations if the 
	# stopping condition is not satisfied

 	check = true
 	ntry = 0
 	while check && ntry < 100
 		NK = N + 2*lambda*Nperp*K
		PK = kronI(W, NK, n)
		PK = kronI(PK, NK, n; transpose=true)
 		fNK = vec(X*NK)' * (PK \ vec(X*NK))
 		fN - fNK >= lambda*K2 ? lambda = 2*lambda : check = false
		ntry = ntry + 1
 	end	

	# 5, Evaluate h(N) = f(N + lambda*Nperp*K). 
	# If f(N) - h(N) < 0.5*lambda*K2, set lambda = 0.5*lambda and repeat
	# Skip this step if Step 4 was repeated at least once
	# Terminate loop after 100 iterations if 
	# the stopping condition is not satisfied
	
	ntry > 1 ? test = false : test = true
	nrep = 0
	while test && nrep < 100
		NK = N + lambda*Nperp*K
        	PK = kronI(W, NK, n)
		PK = kronI(PK, NK, n; transpose=true)
		fNK = vec(X*NK)' * (PK \ vec(X*NK))
        	fN - fNK < 0.5*lambda*K2 ? lambda = 0.5*lambda : test = false
		nrep = nrep + 1
 	end

	# 6, Update N = N + lambda*Nperp*K. 
	# Renormalize by setting [N, Nperp] = Q, 
	# where Q is the Q-factor of the QR decomposition of (updated) N.
	# Stop if K2/lambda < tol

 	N = N + lambda*Nperp*K
 	M = qr(N).Q
	N = M[:,1:(m-r)]
	Nperp = M[:,(m-r):m]
	iter = iter + 1
	if K2/lambda < tol
		repeat = false
	elseif iter > 1000
		repeat = false
	end
 end

 # Compute minimizer R using obtained constraint matrix N. 
 # See equation 4 in IEEE paper. Reshape result to appropriate dimensions
 R = (vec(X) - (W*kron(N, Matrix(I, n, n)) / 
	(kron(N', Matrix(I, n, n))*W*kron(N, Matrix(I, n, n))))
	     *kron(N', Matrix(I, n, n))*vec(X) )
 R = reshape(R, (n, m))
 R, N
end

"""
    mask(A; pct=0.1)

Return matrix A with 100*pct of its entries randomly set to 0.
"""
function mask(A::AbstractMatrix; pct=0.1)
 dims = size(A)
 N = round(pct*dims[1]*dims[2])
 r = rand(CartesianIndices(A), Int(N))
 B = copy(A)
 B[r] .= 0
 B
end

"""
    simulate(n, m, p, r, B, C, Cinv, Crow, Ccol)

Simulate one data set and compute reduced rank r approximation using 5 methods
The data are generated such that Y is n x m, X is n x p, 
B is p x m, C is mp x mp (the covariance of vec(Y)),
and Cinv is the inverse of C, where kron(Ccol, Crow) 
is a Kronecker product approximation to C,
where Crow should be n x n and Ccol should be m x m.
"""
function simulate(n, m, p, r, B, C, Cinv, Crow, Ccol; rng=StableRNG(1))
 Y, X, Bout, Cout, Yhat = gendat(n,m,p,r,B,C;rng=rng)
 Bhat = X \ Yhat
 IX = kron(Matrix(I, m, m), X)
 Y0 = IX*inv(IX'*Cinv*IX)*IX'*Cinv*vec(Y) # GLS fitted values
 Y0 = reshape(Y0, (n, m))
 
 # method 1: take truncated svd of Yhat directly
 Chat = (Y-Yhat)'*(Y-Yhat)
 S = Bhat*sqrt(inv(Chat))
 B1 = tsvd(S, r)
 Y1 = X*B1.U*Diagonal(B1.S)*B1.Vt*sqrt(Chat)

 # method 2: estimate best weighted low-rank approximation of Yhat
 Y2, N2  = rr_sd(Yhat, C, n, m, r; tol=1e-4) 

 # method 3: use Kronecker product decomposition to weight reduced rank approx
 Y3 = rrkron(Yhat, Crow, Ccol, r) 

 # method 4: estimate best weighted low-rank approximation 
 # of Y0 - GLS fitted values
 Y4, N4 = rr_sd(Y0, C, n, m, r; tol=1e-4) 
 CovB = inv(IX'*Cinv*IX)
 Bgls = inv(IX'*Cinv*IX)*IX'*Cinv*vec(Y)
 Bgls = reshape(Bgls, (p, m))

 # method 5: estimate best WLRA of Bhat
 B5, N5 = rr_sd(Bgls, CovB, p, m, r; tol=1e-4) 
 Y5 = X*B5
 
 R0 = sqrt(vec(Y0-X*B)'*Cinv*vec(Y0-X*B))
 R1 = sqrt(vec(Y1-X*B)'*Cinv*vec(Y1-X*B))
 R2 = sqrt(vec(Y2-X*B)'*Cinv*vec(Y2-X*B))
 R3 = sqrt(vec(Y3-X*B)'*Cinv*vec(Y3-X*B))
 R4 = sqrt(vec(Y4-X*B)'*Cinv*vec(Y4-X*B))
 R5 = sqrt(vec(Y5-X*B)'*Cinv*vec(Y5-X*B))
 
 # Add metric in terms of Frobenius distance, not Mahalanobis distance
 F0 = sqrt(tr((Y0-X*B)'*(Y0-X*B)))
 F1 = sqrt(tr((Y1-X*B)'*(Y1-X*B)))
 F2 = sqrt(tr((Y2-X*B)'*(Y2-X*B)))
 F3 = sqrt(tr((Y3-X*B)'*(Y3-X*B)))
 F4 = sqrt(tr((Y4-X*B)'*(Y4-X*B)))
 F5 = sqrt(tr((Y5-X*B)'*(Y5-X*B)))

 [R0, R1, R2, R3, R4, R5, F0, F1, F2, F3, F4, F5]
end

"""
	rr_sim(n, m, p, r; nsim=100, scale=1)

A simulation which generates correlated data and examines the performance of 
several GLS based reduced rank estimators, using Mahalanobis distance.
"""
function rr_sim(n, m, p, r; nsim=100, scale=1, rng=StableRNG(1))
 B = genrr(p, m, r; rng=rng)
 C = gencov(n*m, scale; rng=rng)
 Cinv = inv(C)
 Ccol, Crow = kronapprox(C, float.(Matrix(I, n, n)), m, m, n, n)
 results = zeros(Float64, nsim, 12)
 for i in 1:nsim
	println(i)
	a = simulate(n, m, p, r, B, C, Cinv, Crow, Ccol; rng=rng)
 	results[i,:] = a
 end
 out = round.(mean(results, dims=1); digits=8)
 println(string(nsim,
		 " data sets generated with ",n," observations, ",
		 m," response variables, and ",p," covariates"))
 println(string("Reduced rank ",r," approximation results:"))
 println("Mahalanobis distance:")
 println(["GLS fit", "Yhat svd", "Yhat WLRA", 
	  "Yhat KA", "Ygls WLRA", "Bgls WLRA"])
 println(out[1:6])
 println("Frobenius metric:")
 println(["GLS fit", "Yhat svd", "Yhat WLRA",
	  "Yhat KA", "Ygls WLRA", "Bgls WLRA"])
 println(out[7:12])
 results, B, C
end
 
