
function concatenate_files(input_files::Vector{String}, output_file::String)
    open(output_file, "w") do outfile
        for file_path in input_files
            open(file_path, "r") do infile
                write(outfile, read(infile, String))
                write(outfile, "\n") # Add a newline between file contents for readability
            end
        end
    end
  #  close(output_file)
end

"""
	gee_results(output_file::String)

Concatenates all files in current working directory which begin with 'gee2_simresults_'
"""
function gee_results(output_file::String)
	files = filter(f->isfile(f), readdir())
	geefiles = first.(files, 16) .== "gee2_simresults_"
	geefiles = files[geefiles]
	# Combine files
	concatenate_files(geefiles, output_file)
end
