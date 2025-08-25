using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra
using Distributions
using PrettyTables

include("rr_gee.jl")

outstr = "gee2_simresults_"
#outfile = open("gee2_simresults.txt", "w")

xlog = open("gee2_simlog.txt", "w")

# Set number of clusters
n = 300 #[100, 500, 1000]

# Set cluster size
ng = [2, 5, 10, 20, 30]

# Set number of responses
m = [3] #[3, 5, 8]

# Set rank of mean coefficient matrix r < m
r = [2] #[2, 3, 4]

# Set number of mean covariates
pm = 10

# Set number of scale covariates
ps = 4

i = 0
for ni in ng
	for mi in m
		for ri in r
			if ri < mi
				i = i+1
				outfile = open(string(outstr, i, ".txt"), "w")
				R, B = sim_gee(n, ni, mi, pm, ps, ri, xlog; nsim=100, method=2)
				flush(xlog)
				out = round.(mean(R, dims=1); digits = 8)
				method = ["Dense GEE2", "Full WLRA", "Bhat SVD", "Yhat SVD", "Bhat KA", "Block WLRA", "Bhat CW", "Yhat CW"]
				write(outfile, string("Number of clusters: ", n, ", Cluster size: ", ni, "\n"))
				write(outfile, string("Number of responses: ", mi, ", Number of mean covariates: ", pm, ", Number of scale covariates: ", ps, "\n"))
				write(outfile, string("Rank of mean coefficient matrix: ", ri, "\n"))
				bestout = findmin(out[1:8])
				write(outfile, string("Best result was ", method[bestout[2]], ": ", bestout[1], "\n"))
				write(outfile, "Frobenius distance to true coefficient matrix:\n")
				show(outfile, method)
				write(outfile, "\n")
				show(outfile, out[1:8])
				write(outfile, "\n")
				write(outfile, "Frobenius distance to mean response: \n")
				show(outfile, method)
				write(outfile, "\n")
				show(outfile, out[9:16])
				write(outfile, "\n Mean model coefficient matrix: \n")
				pretty_table(outfile, B; tf=tf_borderless, show_header=false, alignment=:l)
				write(outfile, "\n\n")
				close(outfile)
			end
		end
	end
end

#close(outfile)
close(xlog)
