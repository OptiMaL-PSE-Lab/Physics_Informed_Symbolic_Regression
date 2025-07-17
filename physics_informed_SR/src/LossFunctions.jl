module LossFunctionsModule

import Random: MersenneTwister
using StatsBase: StatsBase
import DynamicExpressions: Node
using LossFunctions: LossFunctions
import LossFunctions: SupervisedLoss
import ..InterfaceDynamicExpressionsModule: eval_tree_array
import ..CoreModule: Options, Dataset, DATA_TYPE, LOSS_TYPE
import ..ComplexityModule: compute_complexity
import ..DimensionalAnalysisModule: violates_dimensional_constraints

using Statistics: mean
using Infiltrator: @infiltrate

function _loss(
    x::AbstractArray{T}, y::AbstractArray{T}, loss::LT
) where {T<:DATA_TYPE,LT<:Union{Function,SupervisedLoss}}
    if LT <: SupervisedLoss
        return LossFunctions.mean(loss, x, y)
    else
        l(i) = loss(x[i], y[i])
        return LossFunctions.mean(l, eachindex(x))
    end
end

function _weighted_loss(
    x::AbstractArray{T}, y::AbstractArray{T}, w::AbstractArray{T}, loss::LT
) where {T<:DATA_TYPE,LT<:Union{Function,SupervisedLoss}}
    if LT <: SupervisedLoss
        return LossFunctions.sum(loss, x, y, w; normalize=true)
    else
        l(i) = loss(x[i], y[i], w[i])
        return sum(l, eachindex(x)) / sum(w)
    end
end

"""If any of the indices are `nothing`, just return."""
@inline function maybe_getindex(v, i...)
    if any(==(nothing), i)
        return v
    else
        return getindex(v, i...)
    end
end

# Evaluate the loss of a particular expression on the input dataset.
function _eval_loss(
    tree::Node{T}, dataset::Dataset{T,L}, options::Options, regularization::Bool, idx
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}

    (prediction, completion) = eval_tree_array(
        tree, maybe_getindex(dataset.X, :, idx), options
    )

    if !completion
        return L(Inf)
    end

    loss_val = if dataset.weighted
        _weighted_loss(
            prediction,
            maybe_getindex(dataset.y, idx),
            maybe_getindex(dataset.weights, idx),
            options.elementwise_loss,
        )
    else
        _loss(prediction, maybe_getindex(dataset.y, idx), options.elementwise_loss)
    end

    if regularization
        loss_val += dimensional_regularization(tree, dataset, options)
    end

    # NOTE this will track all our custom penalizations
    penalties = 0.0

    # NOTE: design
    # options.constraints.initial.active = true
    # options.constraints.initial.penalty = 1e1

    # options.constraints.equilibrium.active = true
    # options.constraints.equilibrium.penalty = 1e2

    # println(prediction)
    # @infiltrate
    # options.constraint_initial_condition = false
    # tol = 1e-1
    diff_initial = 0.0
    # println(options.constraint_initial_condition)
    if options.constraint_initial_condition
        diff_initial = abs(dataset.y[1] - prediction[1])
        # if diff < tol
        #     penalty = 1e1 * (1 + diff)
        #     return L(penalty)
        # else
        #     return L(Inf)
        # end
        penalty_diff_initial = 1e1 * diff_initial
        penalties += penalty_diff_initial
    end

    if options.constraint_concentration_equilibrium
        last_time = dataset.X[end]
        future_first = 5 * last_time
        future_second =  6 * last_time
        time_lapse = future_second - future_first
        futures = hcat(future_first, future_second)
        (future_prediction, future_completion) = eval_tree_array(
            tree, futures, options
        )
        diff_equilibrium = abs2(future_prediction[2] - future_prediction[1])
        if diff_equilibrium > 1e-2 * time_lapse
            penalty_equilibrum = 1e1 * diff_equilibrium
            # @infiltrate
            penalties += penalty_equilibrum
        end
    end

    how_negative(x) = max(0, -x)
    how_positive(x) = max(0, x)

    if options.constraint_always_positive
        mean_diversion = mean(map(how_negative, prediction))
        max_value = maximum(abs.(dataset.y))
        penalty_always_positive = 1e1 * mean_diversion / max_value
        penalties += penalty_always_positive
    end
    if options.constraint_always_negative
        mean_diversion = mean(map(how_positive, prediction))
        max_value = maximum(abs.(dataset.y))
        penalty_always_negative = 1e1 * mean_diversion / max_value
        penalties += penalty_always_negative
    end
    if options.constraint_always_increasing
        # @infiltrate
        penalty_increasing = 0.0
        # for predicted in prediction[:, 2:end]:
        for i in 2:length(prediction)
            predicted_prev = prediction[i-1]
            predicted_after = prediction[i]
            change = predicted_after - predicted_prev
            negative_change = how_negative(change)
            penalty_increasing += 1e1 * negative_change
        end
        penalties += penalty_increasing
    end
    if options.constraint_always_decreasing
        penalty_decreasing = 0.0
        # for predicted in prediction[:, 2:end]:
        for i in 2:length(prediction)
            predicted_prev = prediction[i-1]
            predicted_after = prediction[i]
            change = predicted_after - predicted_prev
            positive_change = how_positive(change)
            penalty_decreasing += 1e1 * positive_change
        end
        penalties += penalty_decreasing
    end

    # TODO monotonically increasing/decreasing constraint
    # evaluate this on the directly available prediction array

    # TODO
    # add flag and penalty to options
    return loss_val + L(penalties)
end

# This evaluates function F:
function evaluator(
    f::F, tree::Node{T}, dataset::Dataset{T,L}, options::Options, idx
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE,F}
    if hasmethod(f, typeof((tree, dataset, options, idx)))
        # If user defines method that accepts batching indices:
        return f(tree, dataset, options, idx)
    elseif options.batching
        error(
            "User-defined loss function must accept batching indices if `options.batching == true`. " *
            "For example, `f(tree, dataset, options, idx)`, where `idx` " *
            "is `nothing` if full dataset is to be used, " *
            "and a vector of indices otherwise.",
        )
    else
        return f(tree, dataset, options)
    end
end

# Evaluate the loss of a particular expression on the input dataset.
function eval_loss(
    tree::Node{T},
    dataset::Dataset{T,L},
    options::Options;
    regularization::Bool=true,
    idx=nothing,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    loss_val = if options.loss_function === nothing
        _eval_loss(tree, dataset, options, regularization, idx)
    else
        f = options.loss_function::Function
        evaluator(f, tree, dataset, options, idx)
    end

    return loss_val
end

function eval_loss_batched(
    tree::Node{T},
    dataset::Dataset{T,L},
    options::Options;
    regularization::Bool=true,
    idx=nothing,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    _idx = idx === nothing ? batch_sample(dataset, options) : idx
    return eval_loss(tree, dataset, options; regularization=regularization, idx=_idx)
end

function batch_sample(dataset, options)
    return StatsBase.sample(1:(dataset.n), options.batch_size; replace=true)::Vector{Int}
end

# Just so we can pass either PopMember or Node here:
get_tree(t::Node) = t
get_tree(m) = m.tree
# Beware: this is a circular dependency situation...
# PopMember is using losses, but then we also want
# losses to use the PopMember's cached complexity for trees.
# TODO!

# Compute a score which includes a complexity penalty in the loss
function loss_to_score(
    loss::L,
    use_baseline::Bool,
    baseline::L,
    member,
    options::Options,
    complexity::Union{Int,Nothing}=nothing,
)::L where {L<:LOSS_TYPE}
    # TODO: Come up with a more general normalization scheme.
    normalization = if baseline >= L(0.01) && use_baseline
        baseline
    else
        L(0.01)
    end
    loss_val = loss / normalization
    size = complexity === nothing ? compute_complexity(member, options) : complexity
    parsimony_term = size * options.parsimony
    loss_val += L(parsimony_term)

    return loss_val
end

# Score an equation
function score_func(
    dataset::Dataset{T,L}, member, options::Options; complexity::Union{Int,Nothing}=nothing
)::Tuple{L,L} where {T<:DATA_TYPE,L<:LOSS_TYPE}
    result_loss = eval_loss(get_tree(member), dataset, options)
    score = loss_to_score(
        result_loss,
        dataset.use_baseline,
        dataset.baseline_loss,
        member,
        options,
        complexity,
    )
    return score, result_loss
end

# Score an equation with a small batch
function score_func_batched(
    dataset::Dataset{T,L},
    member,
    options::Options;
    complexity::Union{Int,Nothing}=nothing,
    idx=nothing,
)::Tuple{L,L} where {T<:DATA_TYPE,L<:LOSS_TYPE}
    result_loss = eval_loss_batched(get_tree(member), dataset, options; idx=idx)
    score = loss_to_score(
        result_loss,
        dataset.use_baseline,
        dataset.baseline_loss,
        member,
        options,
        complexity,
    )
    return score, result_loss
end

"""
    update_baseline_loss!(dataset::Dataset{T,L}, options::Options) where {T<:DATA_TYPE,L<:LOSS_TYPE}

Update the baseline loss of the dataset using the loss function specified in `options`.
"""
function update_baseline_loss!(
    dataset::Dataset{T,L}, options::Options
) where {T<:DATA_TYPE,L<:LOSS_TYPE}
    example_tree = Node(T; val=dataset.avg_y)
    baseline_loss = eval_loss(example_tree, dataset, options)
    if isfinite(baseline_loss)
        dataset.baseline_loss = baseline_loss
        dataset.use_baseline = true
    else
        dataset.baseline_loss = one(L)
        dataset.use_baseline = false
    end
    return nothing
end

function dimensional_regularization(
    tree::Node{T}, dataset::Dataset{T,L}, options::Options
) where {T<:DATA_TYPE,L<:LOSS_TYPE}
    if !violates_dimensional_constraints(tree, dataset, options)
        return zero(L)
    elseif options.dimensional_constraint_penalty === nothing
        return L(1000)
    else
        return L(options.dimensional_constraint_penalty::Float32)
    end
end

end
