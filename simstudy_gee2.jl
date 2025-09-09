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
cluster_sizes = [2, 5, 10, 20, 30]

# Set number of responses
num_responses = [3, 5, 8]

# Set rank of mean coefficient matrix r < m
r = [2, 3, 4]

# Set number of mean covariates
num_mean_covariates = 10

# Set number of scale covariates
num_scale_covariates = 4

write(param_log, string("Number of clusters: ", num_clusters, "\n"))
write(param_log, string("Cluster sizes: ", cluster_sizes, "\n"))
write(param_log, string("Number of responses: ", num_responses, "\n"))
write(param_log, string("Ranks of mean coefficient matrices: ", r, "\n"))
write(param_log, string("Number of mean model covariates: ", num_mean_covariates, "\n"))
write(param_log, string("Number of scale model covariates: ", num_scale_covariates, "\n"))
close(param_log)

for ni in cluster_sizes
	for mi in num_responses
		for ri in r
			if ri < mi
				println("Beginning new iteration:")
				outfile = open(string(outstr, ni, "_", mi, "_", ri,".txt"), "w")	
				R, B = sim_gee(num_clusters, ni, mi, num_mean_covariates, num_scale_covariates, ri, sim_progress_log; nsim=10)
				flush(sim_progress_log)
				means = mean(R, dims=1)
				sds = sqrt.(var(R, dims=1) ./ size(R, 1))
				lower = means - 1.96.*sds
				upper = means + 1.96*sds
				results = hcat(means, sds, lower, upper)
				results = reshape(results, 18, 4)
				columns = ["Mean distance", "Std dev", "Lower CI", "Upper CI"]
				method = ["Dense GEE2", "Full WLRA", "Bhat SVD", "Yhat SVD", "Bhat KA", "Block WLRA", "OlS RR", "Yhat RR", "Yhat CW"]
				write(outfile, string("Number of clusters: ", num_clusters, ", Cluster size: ", ni, "\n"))
				write(outfile, string("Number of responses: ", mi, 
						      ", Number of mean covariates: ", num_mean_covariates,
						      ", Number of scale covariates: ", num_scale_covariates, "\n"))
				write(outfile, string("Rank of mean coefficient matrix: ", ri, "\n"))
				bestout = findmin(means[1:9])
				write(outfile, string("Best result was ", method[bestout[2]], ": ", bestout[1], "\n"))
				write(outfile, "\nFrobenius distance to true coefficient matrix:\n")
				pretty_table(outfile, results[1:9,:]; header=columns, row_labels=method)
				write(outfile, "\nFrobenius distance to mean response: \n")
				pretty_table(outfile, results[10:18,:]; header=columns, row_labels=method)
				write(outfile, "\n Mean model coefficient matrix: \n")
				pretty_table(outfile, B; tf=tf_borderless, show_header=false, alignment=:l)
				write(outfile, "\n\n")
				close(outfile)
			end
		end
	end
end

close(sim_progress_log)
