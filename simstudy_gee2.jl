using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra
using Distributions
using PrettyTables
using Dates

# Source code for simulation study
include("src/rr_gee.jl")
include("src/utils.jl")

# Beginning of file path for results from each parameter configuration
outstr = "artifacts/result_files/gee2_simresults_"

# CSV file tracking the best estimator for each simulation
best_estimators = open("results/best_estimators.csv", "w")
best_estimator_header = ["Cluster size", "Number of responses", "Rank",
			 "Estimator", "Mean distance", "Lower CI", "Upper CI",
			 "Overlap Dense GEE2", "Overlap Full WLRA", 
                         "Overlap Bhat SVD", "Overlap Yhat SVD", 
			 "Overlap Bhat KA", "Overlap Block WLRA", 
			 "Overlap OlS RR", "Overlap Yhat RR", 
                         "Overlap Yhat CW\n"]

best_estimator_header = join(best_estimator_header, ",")
best_estimators = open("results/best_estimators.csv", "w")
write(best_estimators, best_estimator_header)

sim_progress_log = open("logs/gee2_simlog.txt", "w")

# Set number of clusters
n_clusters = 300 #[100, 500, 1000]

# Set cluster size
cluster_sizes = [2, 5, 10, 20, 30]

# Set number of responses
n_responses = [5] # [3, 5, 8]

# Set rank of mean coefficient matrix r < m
r = [2, 3, 4]

# Set number of mean covariates
n_mean_covariates = 10

# Set number of scale covariates
n_scale_covariates = 4

# Set number of simulation iterations
n_sim_iterations = 100

# Set methods for generating error covariances
cov_methods = ["additive","general", "space_time"]

params_ix = Iterators.product(n_clusters, cluster_sizes, n_responses, 
			       r, n_mean_covariates, n_scale_covariates,
				   n_sim_iterations, cov_methods)

write_parameters("results/parameters.txt", params_ix)

# Set random number generation stream
rng = StableRNG(482)

for param in params_ix
	# Unpack parameters
	num_clusters, ni, mi, ri, num_mean_covariates, num_scale_covariates, 
	num_sim_iterations, e = param
	if ri < mi
		println("Beginning new iteration:")
	
		# Open new file
		outfile = open(string(outstr, 
						ni, "_", mi, "_", 
						e,"_",ri,".txt"), "w")
	
		R, Rcov, B = sim_gee(num_clusters, ni, mi, num_mean_covariates, 
							num_scale_covariates, ri, sim_progress_log; 
						nsim=num_sim_iterations, rng=rng, err_method=e)
		flush(sim_progress_log)

		# Write results to file
		write_results(outfile, best_estimators, R, Rcov, B, param)

	end
end

close(sim_progress_log)
close(best_estimators)

# Concatenate files from each set of parameters
gee_results("results/combined_gee2_results.txt")

# Create summary file of best performing estimators
best_result_summary("results/best_estimators.csv", 
		    "results/simulation_summary.txt")
