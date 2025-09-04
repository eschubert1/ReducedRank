using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra
using Distributions
using PrettyTables
using Dates

# Source code for simulation study
include("src/rr_gee.jl")

outstr = "results/gee2_simresults_"
#outfile = open("gee2_simresults.txt", "w")

sim_progress_log = open("logs/gee2_simlog.txt", "w")

param_log = open("logs/parameters.txt", "w")
current_date = today()

write(param_log, string("Simulation run on: ", current_date, "\nParameters:\n"))

# Set number of clusters
num_clusters = 300 #[100, 500, 1000]

# Set cluster size
cluster_sizes = [20] #[2, 5, 10, 20, 30]

# Set number of responses
num_responses = [8] # [3, 5, 8]

# Set rank of mean coefficient matrix r < m
r = [4] # [2, 3, 4]

# Set number of mean covariates
num_mean_covariates = 10

# Set number of scale covariates
num_scale_covariates = 4

write(param_log, string("Number of clusters: ", n, "\n"))
write(param_log, string("Cluster sizes: ", ng, "\n"))
write(param_log, string("Number of responses: ", m, "\n"))
write(param_log, string("Ranks of mean coefficient matrices: ", r, "\n"))
write(param_log, string("Number of mean model covariates: ", pm, "\n"))
write(param_log, string("Number of scale model covariates: ", ps, "\n"))
close(param_log)

for ni in cluster_sizes
	for mi in num_responses
		for ri in r
			if ri < mi
				println("Beginning new iteration:")
				outfile = open(string(outstr, ni, "_", mi, "_", ri,".txt"), "w")	
				R, B = sim_gee(num_clusters, ni, mi, num_mean_covariates, num_scale_covariates, ri, sim_progress_log; nsim=100)
				flush(xlog)
				out = round.(mean(R, dims=1); digits = 8)
				method = ["Dense GEE2", "Full WLRA", "Bhat SVD", "Yhat SVD", "Bhat KA", "Block WLRA", "Bhat CW", "Yhat CW"]
				write(outfile, string("Number of clusters: ", num_clusters, ", Cluster size: ", ni, "\n"))
				write(outfile, string("Number of responses: ", mi, ", Number of mean covariates: ", num_mean_covariates, ", Number of scale covariates: ", num_scale_covariates, "\n"))
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

close(sim_progress_log)
