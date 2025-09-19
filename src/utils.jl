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

"""
	gee_results(output_file)

Concatenate all files in the results directory 
which begin with 'gee2_simresults_'
"""
function gee_results(output_file::String)
	files = filter(isfile, readdir("artifacts/",join=true))
	geefiles = first.(files, 26) .== "artifacts/gee2_simresults_"
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
