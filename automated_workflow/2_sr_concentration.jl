
import Pkg
# Activate environment in sibling directory if it exists, otherwise assume current
csv_path = "../physics_informed_SR"
if isdir(csv_path)
    Pkg.activate(csv_path)
end
# Pkg.instantiate()

using SymbolicRegression
using DelimitedFiles

# Args: num_exp (optional, default 5)
num_datasets = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 5

# Paths
exp_dir = "exp_data"
hof_dir = "hof_files"

# Ensure directories exist (redundant if python script ran first, but good practice)
if !isdir(hof_dir)
    mkdir(hof_dir)
end

tspan = (0.0, 10.0)
num_timepoints = 15
times_per_dataset = range(tspan[1], tspan[2], length=num_timepoints)
num_states = 2

# Custom loss function (from original code)
function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    return sum((prediction .- dataset.y) .^ 2)
end

println("Starting Symbolic Regression for Concentration Profiles...")

for i in 1:num_datasets
    println("Processing Experiment $i...")
    dataset_path = joinpath(exp_dir, "exp_$i.csv")
    if !isfile(dataset_path)
        println("File not found: $dataset_path")
        continue
    end
    
    datasets = readdlm(dataset_path, ',', Float64, '\n')
    
    for j in 1:num_states
        # Original code: X = reshape(times_per_dataset, 1, :)
        # times_per_dataset is a StepRangeLen. Collect it to array.
        X = reshape(collect(times_per_dataset), 1, :)
        y = reshape(datasets[j, :], 1, :)
        
        # Determine species name and options
        if j == 1
            species = "A"
            name = joinpath(hof_dir, "hall_of_fame_A$i.csv")
            options = Options(
                binary_operators=[+, *, /, -],
                unary_operators=[exp],
                loss_function=my_loss,
                maxsize=9,
                parsimony=0.00001,
                timeout_in_seconds=300,
                constraint_initial_condition=true,
                constraint_concentration_equilibrium=true,
                constraint_always_positive=true,
                constraint_always_negative=false,
                constraint_always_increasing=false,
                constraint_always_decreasing=true,
                hofFile=name
            )
        elseif j == 2
            species = "B"
            name = joinpath(hof_dir, "hall_of_fame_B$i.csv")
            options = Options(
                binary_operators=[+, *, /, -],
                unary_operators=[exp],
                loss_function=my_loss,
                maxsize=9,
                parsimony=0.00001,
                timeout_in_seconds=300,
                constraint_initial_condition=true,
                constraint_concentration_equilibrium=true,
                constraint_always_positive=true,
                constraint_always_negative=false,
                constraint_always_increasing=true,
                constraint_always_decreasing=false,
                hofFile=name
            )
        end
        
        println("  Running SR for Species $species...")
        equation_search(
            X, y, niterations=200, options=options, parallelism=:serial, variable_names=["t"]
        )
    end
end

println("Finished Concentration SR.")
