# File contains code to compute various estimators
using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra
using Distributions
import GLM

using GLM: Link

include("rr_sims.jl")

struct StableInverseLink <: GLM.Link
end

function GLM.linkfun(::StableInverseLink, μ::Real)
    return inv(max(μ, 1/10000))
end

function GLM.linkinv(::StableInverseLink, η::Real)
    return inv(max(η, 1/10000))
end

function GLM.mueta(::StableInverseLink, η::T) where T <: Real
    return -1/(max(η, 1/10000)^2)
end

"""
    dense_gee_estimates(Xm, Xv, Xr, y, g, make_rcov, xlog; mean_link, scale_link, cor_link)

Estimates mean model coefficient matrix Bhat and Cov(vec(Bhat))
"""
function dense_gee_estimates(Xm::AbstractMatrix, Xv::AbstractMatrix, 
    Xr::AbstractMatrix, y, g, make_rcov, xlog;
    link_mean=IdentityLink(), link_scale=LogLink(), link_cor=SigmoidLink(-1,1))

 m = size(y, 2)
 pm = size(Xm, 2)

 mm = Array{Union{Nothing, GeneralizedEstimatingEquations2Model}}(nothing, m)

 write(xlog, "Fitting GEE2 models... \n")
 for j in 1:m
        mm[j] = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, j],
                        g, make_rcov; corstruct_mean=ExchangeableCor(),
                        link_mean=link_mean, link_scale=link_scale,
                        link_cor=link_cor,
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