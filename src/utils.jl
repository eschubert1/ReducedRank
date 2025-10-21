using CSV
using PrettyTables

function _concatenate_files(input_files::Vector{String}, output_file::String)
    open(output_file, "w") do outfile
        for file_path in input_files
            open(file_path, "r") do infile
                write(outfile, read(infile, String))
                write(outfile, "\n")
            end
        end
    end
end

function findbest(input_files::Vector{String})
	for file in input_files
		open(file) do f
           for i in eachline(f)
               if contains(i, "Best result was") println(i) end
           end
       end
	end
end

"""
	gee_results(output_file)

Concatenate all files in the results directory 
which begin with 'gee2_simresults_'
"""
function gee_results(output_file::String)
	files = filter(isfile, readdir("artifacts/result_files",join=true))
	geefiles = first.(files, 39) .== "artifacts/result_files/gee2_simresults_"
	geefiles = files[geefiles]
	# Combine files
	_concatenate_files(geefiles, output_file)
end


function best_result_summary(input_file, output_file)
	results = CSV.File(input_file)
	methods = ["Dense GEE2", "Full WLRA", "Bhat SVD", "Yhat SVD", 
		   "Bhat KA", "Block WLRA", "OLS RR", "Yhat RR", "Yhat CW"]

	best_frequency = zeros(9)
	overlap_frequency = zeros(9, 9)
	for row in results
		best_method = row.Estimator
		best_index = findall(contains(best_method), methods)
		best_frequency[best_index] .+= 1
		overlap_columns = propertynames(row)[8:16]
		overlap_update = map(p -> getproperty(row, p),
                                          overlap_columns)
		overlap_frequency[best_index,:] += reshape(overlap_update, (1,9))
	end
	
	frequency_table = hcat(best_frequency, overlap_frequency)
	col_header = vcat("Frequency", methods)
	output_stream = open(output_file, "w")
	write(output_stream, "Simulation results summary:\n\n")
	pretty_table(output_stream, frequency_table;
				    header = col_header,
				    row_labels=methods)
	close(output_stream)
end

"""
	write_parameters(outfile, params_ix)

Write simulation parameters to a log file.
"""
function write_parameters(outfile, params_ix)
	param_log = open(outfile, "w")
	current_date = today()

	write(param_log, string("Simulation run on: ", current_date))

	write(param_log, "\n\nSimulation parameters:\n")
	
	p, ix = Iterators.peel(params_ix)
	A = zeros(length(params_ix), length(values(p))-1)
	cov_parms = String[]
	for (i, param) in enumerate(params_ix)
		vals = collect(values(param))
		A[i,:] = vals[1:end-1]
		push!(cov_parms, string(vals[end]))
	end
	
	num_clusters = unique(A[:,1])
	cluster_sizes = unique(A[:,2])
	num_responses = unique(A[:,3])
	rank_mean = unique(A[:,4])
	num_mean_covariates = unique(A[:,5])
	num_scale_covariates = unique(A[:,6])
	num_sim_iterations = unique(A[:,7])
	cov_methods = unique(cov_parms)
	write(param_log, string("Cluster sizes: ", cluster_sizes, "\n"))
	write(param_log, string("Number of clusters: ", num_clusters, "\n"))
	write(param_log, string("Number of responses: ", num_responses, "\n"))
	write(param_log, string("Ranks of mean coefficient matrices: ", rank_mean, "\n"))
	write(param_log, 
	      string("Number of mean model covariates: ", num_mean_covariates, "\n"))
	write(param_log, 
	      string("Number of scale model covariates: ", num_scale_covariates, "\n"))
	write(param_log,
	      string("Number of simulation iterations: ", num_sim_iterations, "\n"))
	write(param_log, string("Covariance generation methods: ", cov_methods, "\n"))
	close(param_log)
end

function write_results(outfile, best_estimators, R, Rcov, B, param)
	# Unpack parameters
	num_clusters, ni, mi, ri, num_mean_covariates, num_scale_covariates, 
	num_sim_iterations, e = param

	# Compute summary statistics
	means = mean(R, dims=1)
	sds = sqrt.(var(R, dims=1) ./ size(R, 1))
	lower = means - 1.96.*sds
	upper = means + 1.96*sds
	results = hcat(means, sds, lower, upper)
	results = reshape(results, 18, 4)

	# Names of estimators and metrics
	columns = ["Mean distance", "Std dev", "Lower CI", "Upper CI"]
	method = ["Dense GEE2", "Full WLRA", "Bhat SVD", "Yhat SVD", 
			"Bhat KA", "Block WLRA", "OLS RR", "Yhat RR", "Yhat CW"]

	write(outfile, string("Number of clusters: ", num_clusters, 
					", Cluster size: ", ni, "\n"))

	write(outfile, string("Number of responses: ", mi, 
		", Number of mean covariates: ", num_mean_covariates,
		", Number of scale covariates: ", num_scale_covariates, "\n"))
	
	write(outfile, string("Rank of mean coefficient matrix: ", ri, "\n"))
	write(outfile, string("Covariance generation method: ", e, "\n"))

	bestout = findmin(means[10:18])
	write(outfile, string("Best result was ", method[bestout[2]], ": ",
					bestout[1], "\n"))

	write(outfile, string("Mean distance to true covariance matrix:",
	mean(Rcov)))

	write(outfile, "\nFrobenius distance to true coefficient matrix:\n")
	pretty_table(outfile, results[1:9,:];
					header=columns, 
					row_labels=method)
	write(outfile, "\nFrobenius distance to mean response: \n")
	pretty_table(outfile, results[10:18,:];
					header=columns, 
					row_labels=method)
	write(outfile, "\n Mean model coefficient matrix: \n")
	pretty_table(outfile, B; 
					tf=tf_borderless, 
					show_header=false, 
					alignment=:l)
	write(outfile, "\n\n")
			
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
	close(outfile)
end