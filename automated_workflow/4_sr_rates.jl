
import Pkg
# Activate environment in sibling directory if it exists, otherwise assume current
script_dir = @__DIR__
project_path = joinpath(dirname(script_dir), "physics_informed_SR")
Pkg.activate(project_path)
Pkg.instantiate()

using SymbolicRegression
using DelimitedFiles

# Args: num_exp (optional, default 5)
num_datasets = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 5

# Paths
const_dir = joinpath(script_dir, "const_data")
hof_dir = joinpath(script_dir, "hof_files")


num_states = 2

# Custom loss function (from original code)
function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    return sum((prediction .- dataset.y) .^ 2)
end

println("Starting Symbolic Regression for Rate Models...")

conc_data_path = joinpath(const_dir, "conc_data_for_rate_models.csv")
if !isfile(conc_data_path)
    println("Error: $conc_data_path not found. Cannot proceed.")
    exit(1)
end

conc_data = readdlm(conc_data_path, ',', Float64, '\n')
# conc_data shape from Python: (num_species, total_samples) = (2, 75)
# readdlm reads it as (2, 75).
# We need X to be (n_features, n_samples). It should already be correct.
X = conc_data

for j in 1:num_states
    if j == 1
        species = "A"
        name = joinpath(hof_dir, "hall_of_fame_rate_A$num_datasets.csv")
        rate_path = joinpath(const_dir, "rate_data_A.csv")

        # Original options for A with constraints
        options = Options(
            binary_operators=[+, *, /, -],
            loss_function=my_loss,
            maxsize=18,
            parsimony=0.00001,
            timeout_in_seconds=600,
            constraint_initial_condition=false,
            constraint_concentration_equilibrium=false,
            constraint_always_positive=false,
            constraint_always_negative=true,
            constraint_always_increasing=true,
            constraint_always_decreasing=false,
            hofFile=name,
            verbosity=0
        )
        iterations = 400

    elseif j == 2
        species = "B"
        name = joinpath(hof_dir, "hall_of_fame_rate_B$num_datasets.csv")
        rate_path = joinpath(const_dir, "rate_data_B.csv")

        # Original options for B with constraints
        options = Options(
            binary_operators=[+, *, /, -],
            loss_function=my_loss,
            maxsize=18,
            parsimony=0.00001,
            timeout_in_seconds=600,
            constraint_initial_condition=false,
            constraint_concentration_equilibrium=false,
            constraint_always_positive=true,
            constraint_always_negative=false,
            constraint_always_increasing=false,
            constraint_always_decreasing=true,
            hofFile=name,
            verbosity=0
        )
        iterations = 200
    end

    if !isfile(rate_path)
        println("Error: $rate_path not found.")
        continue
    end

    rate_data = readdlm(rate_path, ',', Float64, '\n')
    y = reshape(rate_data, 1, :)

    println("  Running SR for Rate Model $species...")
    equation_search(
        X, y, niterations=iterations, options=options, parallelism=:serial, variable_names=["A", "B"]
    )
end

println("Finished Rate SR.")
