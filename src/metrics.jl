# Functions for various performance metrics

"""
    frobenius_distance(estimate, target)

Compute the Frobenius distance between two matrices
"""
function frobenius_distance(estimate::AbstractMatrix, target::AbstractMatrix)
    return norm(estimate - target, 2)
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