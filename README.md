# Project description and objectives:
The goal of this project is to understand the performance of weighted
reduced rank estimators through simulation.

Specifically, the project considers multivariate regression where 
the data consistent of independent clusters, and within each cluster 
the observations are correlated. Additionally, the different response 
variables are also correlated. We can express the model 
for one cluster as Y~i~ = X~i~ B + E_i, where Y is n_i x m, X is n_i x p,
B is p x m, and E_i is also n_i x m. B has rank r < m, and in this study 
the rank of B is assumed to be known.

Furthermore, in this framework, E(Y_i | X) = X_i B, 
and Cov(vec(Y_i)|X) = Cov(vec(E_i)|X) = R,
where R is an (n_i x m) x (n_i x m) covariance matrix. 
The matrix R is not necessarily factorizable into row-wise and 
colum-wise covariance matrices.

For each run of the simulation, a dataset is generated with the above 
mean model, and the columns of B are
estimated using the GEE2 method of Yan and Fine (2004) by 
considering each outcome variable separately.
Once each model has been fit, the estimates of the mean coefficients 
are concatenated to form an estimate of B. In general, this estimate 
will be a full rank matrix. Several low-rank approximations are constructed,
and their distance to the true mean coefficients are measured. 

# Installation instructions:
To run this simulation successfully, follow the steps below:

1) Clone this repository onto your machine in a directory
2) Install Julia if needed by running 
```
curl -fsSL https://install.julialang.org | sh
```
3) Clone the EstimatingEquationsRegression.jl package from the repository at
   https://github.com/kshedden/EstimatingEquationsRegression.jl/tree/main
   This should be saved two parent directories above the directory in step 1.
   The reason for doing this is that the version of EstimatingEquationsRegression.jl used
   for this project is not yet available as a Julia package, and thus needs to be built
   locally. Placing EstimatingEquationsRegression.jl two parent directories above matches the path
   saved in this project's Manifest.toml and should minimize complications with building the package.
4) Navigate to the ReducedRank directory cloned in step 1.
5) Run Julia by typing `julia` into the command line
6) Press `]` to open Julia's package manager. First type `activate .` to activate a project in the current 
   directory, and then type `instantiate` to activate the virtual environment
   used in this project. Julia will install the exact versions of the required packages for this project,
   if these have not yet been installed.
8) Hit the backspace key to return to the Julia REPL. Now you should be able to run the simulation
   by typing `include("simstudy_gee2.jl")` and hitting ENTER. To run tests, type `include("tests/runtests.jl")`
   Note the simulation takes approximately 24 hours to run. The parameters for the simulation can be modified
   by editing simstudy_gee2.jl



