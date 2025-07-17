import Pkg
project_dir = "/Users/md1621/Desktop/PhD-Code/Physics-Informed_ADoK/physics_informed_SR"
# exp_dir = "/Users/md1621/Desktop/PhD-Code/Physics-Informed_ADoK/physics_informed_SR/exp_data"
# rate_dir = "/Users/md1621/Desktop/PhD-Code/Physics-Informed_ADoK/physics_informed_SR/const_data"
Pkg.activate(project_dir)
Pkg.instantiate()

# exp_dir = pwd() * "exp_data"
exp_dir = "Physics-Informed_ADoK/physics_informed_SR/Hydrodealkylation_of_Toluene/exp_data"
hof_dir = "Physics-Informed_ADoK/physics_informed_SR/Hydrodealkylation_of_Toluene/hof_files"
rate_dir = "Physics-Informed_ADoK/physics_informed_SR/Hydrodealkylation_of_Toluene/const_data"

using IterTools: ncycle
using SymbolicRegression
using Infiltrator
using DelimitedFiles

tspan = (0e0, 1e1)
num_timepoints = 15

times_per_dataset=collect(range(tspan[begin], tspan[end]; length=num_timepoints))

ini_T = [1e0, 5e0, 5e0, 1e0, 1e0, 5e0, 5e0]
ini_H = [8e0, 8e0, 3e0, 3e0, 8e0, 6.95383183e0, 8e0]
ini_B = [2e0, 0e0, 0e0, 0e0, 2e0, 2e0, 0.696312e0]
ini_M = [3e0, 0.5e0, 0.5e0, 3e0, 0.5e0, 2.65975052e0, 3e0]

num_datasets = length(ini_T)
num_states = 4

# # NOTE only used for development
# # start dev
# # i = 1
# # j = 1
# # datasets = readdlm(exp_dir*"/exp_$i.csv", ',', Float64, '\n')
# # X = reshape(times_per_dataset, 1, :)
# # y = reshape(datasets[j, :], 1, :)
# # name = "hof_files/hall_of_fame_T$i.csv"


# # options = Options(; # NOTE add new constraint here
# # binary_operators=[+, *, /, -],
# # unary_operators=[exp],
# # constraint_initial_condition=true,
# # constraint_concentration_equilibrium=true,
# # hofFile=name
# # )

# # hall_of_fame = equation_search(
# # X, y, niterations=2, options=options, parallelism=:serial
# # )
# # end dev

function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    return sum((prediction .- dataset.y) .^ 2)
end


# READ THE DATASET FROM PYTHON
for i in num_datasets:num_datasets
    datasets = readdlm(exp_dir*"/exp_$i.csv", ',', Float64, '\n')
    #------------------------------#

    for j in 1:num_states
        X = reshape(times_per_dataset, 1, :)
        y = reshape(datasets[j, :], 1, :)

        if j == 1
            name = hof_dir*"/hall_of_fame_T$i.csv"
            options = Options(; # NOTE add new constraint here
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
            name =  hof_dir*"/hall_of_fame_H$i.csv"
            options = Options(; # NOTE add new constraint here
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

        elseif j == 3
            name =  hof_dir*"/hall_of_fame_B$i.csv"
            options = Options(; # NOTE add new constraint here
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

        elseif j == 4
            name =  hof_dir*"/hall_of_fame_M$i.csv"
            options = Options(; # NOTE add new constraint here
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

        hall_of_fame = equation_search(
            X, y, niterations=200, options=options, parallelism=:serial, variable_names=["t"]
        )
    end
end

# NOTE model selection done in python to generate the following file...


# conc_data = readdlm(rate_dir*"/conc_data_for_rate_models.csv", ',', Float64, '\n')

# for j in 1:num_states
#     X = reshape(conc_data, num_states - 1, :)
#     i = num_datasets

#     if j == 1
#         name = hof_dir*"/hall_of_fame_rate_T$i.csv"
#         a = readdlm(rate_dir*"/rate_data_T.csv", ',', Float64, '\n')
#         y = reshape(a, 1, :)
#         num = 200
#         options = Options(; # NOTE add new constraint here
#             binary_operators=[+, *, /, -],
#             loss_function=my_loss,
#             maxsize=25,
#             parsimony=0.00001,
#             timeout_in_seconds=600,
#             constraint_initial_condition=false,
#             constraint_concentration_equilibrium=false,
#             constraint_always_positive=false,
#             constraint_always_negative=true,
#             constraint_always_increasing=true,
#             constraint_always_decreasing=false,
#             hofFile=name
#         )

#     elseif j == 2
#         name = hof_dir*"/hall_of_fame_rate_H$i.csv"
#         a = readdlm(rate_dir*"/rate_data_H.csv", ',', Float64, '\n')
#         y = reshape(a, 1, :)
#         num = 2000
#         options = Options(; # NOTE add new constraint here
#             binary_operators=[+, *, /, -],
#             loss_function=my_loss,
#             maxsize=25,
#             parsimony=0.00001,
#             timeout_in_seconds=600,
#             constraint_initial_condition=false,
#             constraint_concentration_equilibrium=false,
#             constraint_always_positive=false,
#             constraint_always_negative=true,
#             constraint_always_increasing=true,
#             constraint_always_decreasing=false,
#             hofFile=name
#         )

#     elseif j == 3
#         name = hof_dir*"/hall_of_fame_rate_B$i.csv"
#         a = readdlm(rate_dir*"/rate_data_B.csv", ',', Float64, '\n')
#         y = reshape(a, 1, :)
#         num = 200
#         options = Options(; # NOTE add new constraint here
#             binary_operators=[+, *, /, -],
#             loss_function=my_loss,
#             maxsize=25,
#             parsimony=0.00001,
#             timeout_in_seconds=600,
#             constraint_initial_condition=false,
#             constraint_concentration_equilibrium=false,
#             constraint_always_positive=true,
#             constraint_always_negative=false,
#             constraint_always_increasing=false,
#             constraint_always_decreasing=true,
#             hofFile=name
#         )

#     elseif j == 4
#         name = hof_dir*"/hall_of_fame_rate_M$i.csv"
#         a = readdlm(rate_dir*"/rate_data_M.csv", ',', Float64, '\n')
#         y = reshape(a, 1, :)
#         num = 200
#         options = Options(; # NOTE add new constraint here
#             binary_operators=[+, *, /, -],
#             loss_function=my_loss,
#             maxsize=25,
#             parsimony=0.00001,
#             timeout_in_seconds=600,
#             constraint_initial_condition=false,
#             constraint_concentration_equilibrium=false,
#             constraint_always_positive=true,
#             constraint_always_negative=false,
#             constraint_always_increasing=false,
#             constraint_always_decreasing=true,
#             hofFile=name
#         )

#     end

#     hall_of_fame = equation_search(
#         X, y, niterations=num, options=options, parallelism=:serial, variable_names=["T", "H", "B"]
#     )
# end


#TODO: declare the name of the variables for the models for easier manipulation
