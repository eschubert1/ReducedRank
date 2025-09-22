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
outstr = "artifacts/gee2_simresults_"

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

param_log = open("results/parameters.txt", "w")
current_date = today()

write(param_log, string("Simulation run on: ", 
			current_date, "\n\nParameters:\n"))

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

# Set number of simulation iterations
num_sim_iterations = 100

write(param_log, string("Number of clusters: ", num_clusters, "\n"))
write(param_log, string("Cluster sizes: ", cluster_sizes, "\n"))
write(param_log, string("Number of responses: ", num_responses, "\n"))
write(param_log, string("Ranks of mean coefficient matrices: ", r, "\n"))
write(param_log, 
       string("Number of mean model covariates: ", num_mean_covariates, "\n"))
write(param_log, 
       string("Number of scale model covariates: ", num_scale_covariates, "\n"))
write(param_log,
       string("Number of simulation iterations: ", num_sim_iterations, "\n"))
close(param_log)

# Set random number generation stream
rng = StableRNG(482)

for ni in cluster_sizes
	for mi in num_responses
		for ri in r
			if ri < mi
				println("Beginning new iteration:")
				
				# Open new file
				outfile = open(string(outstr, 
						       ni, "_", mi, "_", 
						       ri,".txt"), "w")	
				
				R, B = sim_gee(num_clusters, 
						ni, mi, num_mean_covariates, 
						num_scale_covariates, ri, 
						sim_progress_log; 
						nsim=num_sim_iterations, 
						rng=rng)
				flush(sim_progress_log)
			
				# Compute summary statistics
				means = mean(R, dims=1)
				sds = sqrt.(var(R, dims=1) ./ size(R, 1))
				lower = means - 1.96.*sds
				upper = means + 1.96*sds
				results = hcat(means, sds, lower, upper)
				results = reshape(results, 18, 4)
				columns = ["Mean distance", "Std dev", 
					   "Lower CI", "Upper CI"]
				method = ["Dense GEE2", "Full WLRA", 
					  "Bhat SVD", "Yhat SVD", "Bhat KA", 
					  "Block WLRA", "OLS RR", "Yhat RR", 
					  "Yhat CW"]

				# Write results
				write(outfile, string("Number of clusters: ", 
						      num_clusters, 
						      ", Cluster size: ", ni, 
						      "\n"))
				write(outfile, string("Number of responses: ", 
						      mi, 
					", Number of mean covariates: ", 
					num_mean_covariates,
					", Number of scale covariates: ", 
					num_scale_covariates, "\n"))
				write(outfile, string(
				"Rank of mean coefficient matrix: ", ri, "\n"))
				bestout = findmin(means[10:18])
				write(outfile, string("Best result was ",
						      method[bestout[2]], ": ",
						      bestout[1], "\n"))
				write(outfile, 
			"\nFrobenius distance to true coefficient matrix:\n")
				pretty_table(outfile, results[1:9,:];
						      header=columns, 
						      row_labels=method)
				write(outfile, 
			"\nFrobenius distance to mean response: \n")
				pretty_table(outfile, results[10:18,:];
						      header=columns, 
						      row_labels=method)
				write(outfile, 
				     "\n Mean model coefficient matrix: \n")
				pretty_table(outfile, B; 
						      tf=tf_borderless, 
						      show_header=false, 
						      alignment=:l)
				write(outfile, "\n\n")
				close(outfile)

				# Find estimators which have overlapping 
				# confidence intervals with the best estimator
				overlap = lower[10:18] .<= upper[bestout[2]+9]

				# Save result summary to csv file
				csv_results = [ni, mi, ri, method[bestout[2]],
					       bestout[1], lower[bestout[2]+9],
					       upper[bestout[2]+9]]
				csv_results = vcat(csv_results, overlap)
				csv_results = join(csv_results, ",")
				write(best_estimators, csv_results)
				write(best_estimators, "\n")
				flush(best_estimators)
			end
		end
	end
end

close(sim_progress_log)
close(best_estimators)

# Concatenate files from each set of parameters
gee_results("results/combined_gee2_results.txt")

# Create summary file of best performing estimators
best_result_summary("results/best_estimators.csv", 
		    "results/simulation_summary.txt")
