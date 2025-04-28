#  Copyright (c) 2017-25, Oscar Dowson and SDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

mutable struct Cut
    intercept::Float64
    coefficients::Dict{Symbol,Float64}
    obj_y::Union{Nothing,NTuple{N,Float64} where {N}}
    belief_y::Union{Nothing,Dict{T,Float64} where {T}}
    non_dominated_count::Int
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
end

mutable struct SampledState
    state::Dict{Symbol,Float64}
    obj_y::Union{Nothing,NTuple{N,Float64} where {N}}
    belief_y::Union{Nothing,Dict{T,Float64} where {T}}
    dominating_cut::Cut
    best_objective::Float64
end

mutable struct ConvexApproximation
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    objective_states::Union{Nothing,NTuple{N,JuMP.VariableRef} where {N}}
    belief_states::Union{Nothing,Dict{T,JuMP.VariableRef} where {T}}
    # Storage for cut selection
    cuts::Vector{Cut}
    sampled_states::Vector{SampledState}
    cuts_to_be_deleted::Vector{Cut}
    deletion_minimum::Int

    function ConvexApproximation(
        theta::JuMP.VariableRef,
        states::Dict{Symbol,JuMP.VariableRef},
        objective_states,
        belief_states,
        deletion_minimum::Int,
    )
        return new(
            theta,
            states,
            objective_states,
            belief_states,
            Cut[],
            SampledState[],
            Cut[],
            deletion_minimum,
        )
    end
end

_magnitude(x) = abs(x) > 0 ? log10(abs(x)) : 0

function _dynamic_range_warning(intercept, coefficients)
    lo = hi = _magnitude(intercept)
    lo_v = hi_v = intercept
    for v in values(coefficients)
        i = _magnitude(v)
        if v < lo_v
            lo, lo_v = i, v
        elseif v > hi_v
            hi, hi_v = i, v
        end
    end
    if hi - lo > 10
        @warn(
            """Found a cut with a mix of small and large coefficients.
          The order of magnitude difference is $(hi - lo).
          The smallest cofficient is $(lo_v).
          The largest coefficient is $(hi_v).

      You can ignore this warning, but it may be an indication of numerical issues.

      Consider rescaling your model by using different units, e.g, kilometers instead
      of meters. You should also consider reducing the accuracy of your input data (if
      you haven't already). For example, it probably doesn't make sense to measure the
      inflow into a reservoir to 10 decimal places.""",
            maxlog = 1,
        )
    end
    return
end

function _add_cut(
    model::PolicyGraph{T},
    node::Node{T},
    V::ConvexApproximation,
    θᵏ::Float64,
    shift::Float64,
    πᵏ::Dict{Symbol,Float64},
    xᵏ::Dict{Symbol,Float64},
    obj_y::Union{Nothing,NTuple{N,Float64}},
    belief_y::Union{Nothing,Dict{T,Float64}};
    cut_selection::Bool = true,
) where {N,T}
    for (key, x) in xᵏ
        θᵏ -= πᵏ[key] * x
    end
    _dynamic_range_warning(θᵏ, πᵏ)
    cut = Cut(θᵏ, πᵏ, obj_y, belief_y, 1, nothing)
    _add_cut_constraint_to_model(model, node, V, cut, shift)
    if cut_selection
        _cut_selection_update(V, cut, xᵏ)
    end
    return
end

#Mathis
function _add_cut_to_two_stage_model(
    node::Node{T},   
    θᵏ::Float64,
    πᵏ::Dict{Symbol,Float64},
    xᵏ::Dict{Symbol,Float64},
    obj_y::Union{Nothing,NTuple{N,Float64}},
    belief_y::Union{Nothing,Dict{T,Float64}}
) where {N,T}

    for (key, x) in xᵏ
        θᵏ -= πᵏ[key] * x
    end
    _dynamic_range_warning(θᵏ, πᵏ)
    cut = Cut(θᵏ, πᵏ, obj_y, belief_y, 1, nothing)
    
    two_stage_model = node.two_stage.model
    S=length(node.noise_terms)
    expr = @expression(
        two_stage_model,
        [j=1:S],
        node.two_stage.bellman_variables[j]-sum(cut.coefficients[i] * node.two_stage.states[i][j] for (i, x) in node.states)
    )
    if JuMP.objective_sense(two_stage_model) == MOI.MIN_SENSE
        @constraint(two_stage_model, [j in 1:S], expr[j] >= cut.intercept-shift)
    else
        #Mathis Quel shift quand on maximise ?
        @constraint(two_stage_model, [j in 1:S], expr[j] <= cut.intercept)
    end
    return
end

function _add_cut_constraint_to_model(
    model::PolicyGraph{T}, 
    node::Node{T}, 
    V::ConvexApproximation, 
    cut::Cut, 
    shift::Float64
) where {T}
    mod = JuMP.owner_model(V.theta)
    yᵀμ = JuMP.AffExpr(0.0)
    if V.objective_states !== nothing
        for (y, μ) in zip(cut.obj_y, V.objective_states)
            JuMP.add_to_expression!(yᵀμ, y, μ)
        end
    end
    if V.belief_states !== nothing
        for (k, μ) in V.belief_states
            JuMP.add_to_expression!(yᵀμ, cut.belief_y[k], μ)
        end
    end
    expr = @expression(
        mod,
        V.theta + yᵀμ - sum(cut.coefficients[i] * x for (i, x) in V.states)
    )
    cut.constraint_ref = if JuMP.objective_sense(mod) == MOI.MIN_SENSE
        csp = @constraint(mod, expr >= cut.intercept-shift)
        _update_value_function(model, node, cut, shift, csp)
    else
        @constraint(mod, expr <= cut.intercept)
    end

    #Get cst in node
    return
end

function _update_value_function(
    model::PolicyGraph{T}, 
    node::Node{T}, 
    cut::Cut, 
    shift::Float64,
    csp::JuMP.ConstraintRef
) where {T}

    # TV
    for child in node.children
        child_node=model[child.term]
        child_vf=child_node.value_function
        md_TV = child_vf.model_TV
        @constraint(md_TV, child_vf.theta_TV -sum(cut.coefficients[i]*x for (i,x) in child_vf.states_TV)>=cut.intercept)
        
        md_V = child_vf.model
        cV = @constraint(md_V, child_vf.theta-sum(cut.coefficients[i]*x for (i,x) in child_vf.states)>=cut.intercept-shift)
        
        cutV = Cut2(
            cut.intercept-shift,
            cut.coefficients,
            shift,
            cV,
            csp,
        )
    
        push!(child_node.value_function.cut_V, cutV)
    end

    #Get cst in node
    return
end

"""
Internal function: calculate the height of `cut` evaluated at `state`.
"""
function _eval_height(cut::Cut, sampled_state::SampledState)
    height = cut.intercept
    for (key, value) in cut.coefficients
        height += value * sampled_state.state[key]
    end
    return height
end

"""
Internal function: check if the candidate point dominates the incumbent.
"""
function _dominates(candidate, incumbent, minimization::Bool)
    return minimization ? candidate >= incumbent : candidate <= incumbent
end

function _cut_selection_update(
    V::ConvexApproximation,
    cut::Cut,
    state::Dict{Symbol,Float64},
)
    model = JuMP.owner_model(V.theta)
    is_minimization = JuMP.objective_sense(model) == MOI.MIN_SENSE
    sampled_state = SampledState(state, cut.obj_y, cut.belief_y, cut, NaN)
    sampled_state.best_objective = _eval_height(cut, sampled_state)
    # Loop through previously sampled states and compare the height of the most
    # recent cut against the current best. If this new cut is an improvement,
    # store this one instead.
    for old_state in V.sampled_states
        # Only compute cut selection at same points in concave space.
        if old_state.obj_y != cut.obj_y || old_state.belief_y != cut.belief_y
            continue
        end
        height = _eval_height(cut, old_state)
        if _dominates(height, old_state.best_objective, is_minimization)
            old_state.dominating_cut.non_dominated_count -= 1
            cut.non_dominated_count += 1
            old_state.dominating_cut = cut
            old_state.best_objective = height
        end
    end
    push!(V.sampled_states, sampled_state)
    # Now loop through previously discovered cuts and compare their height at
    # `sampled_state`. If a cut is an improvement, add it to a queue to be
    # added.
    for old_cut in V.cuts
        if old_cut.constraint_ref !== nothing
            # We only care about cuts not currently in the model.
            continue
        elseif old_cut.obj_y != sampled_state.obj_y
            # Only compute cut selection at same points in objective space.
            continue
        elseif old_cut.belief_y != sampled_state.belief_y
            # Only compute cut selection at same points in belief space.
            continue
        end
        height = _eval_height(old_cut, sampled_state)
        if _dominates(height, sampled_state.best_objective, is_minimization)
            sampled_state.dominating_cut.non_dominated_count -= 1
            old_cut.non_dominated_count += 1
            sampled_state.dominating_cut = old_cut
            sampled_state.best_objective = height
            _add_cut_constraint_to_model(V, old_cut)
        end
    end
    push!(V.cuts, cut)
    # Delete cuts that need to be deleted.
    for cut in V.cuts
        if cut.non_dominated_count < 1
            if cut.constraint_ref !== nothing
                push!(V.cuts_to_be_deleted, cut)
            end
        end
    end
    if length(V.cuts_to_be_deleted) >= V.deletion_minimum
        for cut in V.cuts_to_be_deleted
            JuMP.delete(model, cut.constraint_ref)
            cut.constraint_ref = nothing
            cut.non_dominated_count = 0
        end
    end
    empty!(V.cuts_to_be_deleted)
    return
end

@enum(CutType, SINGLE_CUT, MULTI_CUT)

# Internal struct: this struct is just a cache for arguments until we can build
# an actual instance of the type T at a later point.
struct InstanceFactory{T}
    args::Any
    kwargs::Any
    InstanceFactory{T}(args...; kwargs...) where {T} = new{T}(args, kwargs)
end

"""
    BellmanFunction

A representation of the value function. SDDP.jl uses the following unique
representation of the value function that is undocumented in the literature.

It supports three types of state variables:

 1) x - convex "resource" states
 2) b - concave "belief" states
 3) y - concave "objective" states

In addition, we have three types of cuts:

 1) Single-cuts (also called "average" cuts in the literature), which involve
    the risk-adjusted expectation of the cost-to-go.
 2) Multi-cuts, which use a different cost-to-go term for each realization w.
 3) Risk-cuts, which correspond to the facets of the dual interpretation of a
    convex risk measure.

Therefore, ValueFunction returns a JuMP model of the following form:

```
V(x, b, y) =
    min: μᵀb + νᵀy + θ
    s.t. # "Single" / "Average" cuts
         μᵀb(j) + νᵀy(j) + θ >= α(j) + xᵀβ(j),          ∀ j ∈ J
         # "Multi" cuts
         μᵀb(k) + νᵀy(k) + φ(w) >= α(k, w) + xᵀβ(k, w), ∀w ∈ Ω, k ∈ K
         # "Risk-set" cuts
         θ ≥ Σ{p(k, w) * φ(w)}_w - μᵀb(k) - νᵀy(k),     ∀ k ∈ K
```
"""
mutable struct BellmanFunction
    cut_type::CutType
    global_theta::ConvexApproximation
    local_thetas::Vector{ConvexApproximation}
    # Cuts defining the dual representation of the risk measure.
    risk_set_cuts::Set{Vector{Float64}}
end

"""
    BellmanFunction(;
        lower_bound = -Inf,
        upper_bound = Inf,
        deletion_minimum::Int = 1,
        cut_type::CutType = MULTI_CUT,
    )
"""
function BellmanFunction(;
    lower_bound = -Inf,
    upper_bound = Inf,
    deletion_minimum::Int = 1,
    cut_type::CutType = MULTI_CUT,
)
    return InstanceFactory{BellmanFunction}(;
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        deletion_minimum = deletion_minimum,
        cut_type = cut_type,
    )
end

function bellman_term(bellman_function::BellmanFunction)
    return bellman_function.global_theta.theta
end

function initialize_bellman_function(
    factory::InstanceFactory{BellmanFunction},
    model::PolicyGraph{T},
    node::Node{T},
) where {T}
    lower_bound, upper_bound, deletion_minimum, cut_type =
        -Inf, Inf, 0, SINGLE_CUT
    if length(factory.args) > 0
        error(
            "Positional arguments $(factory.args) ignored in BellmanFunction.",
        )
    end
    for (kw, value) in factory.kwargs
        if kw == :lower_bound
            lower_bound = value
        elseif kw == :upper_bound
            upper_bound = value
        elseif kw == :deletion_minimum
            deletion_minimum = value
        elseif kw == :cut_type
            cut_type = value
        else
            error(
                "Keyword $(kw) not recognised as argument to BellmanFunction.",
            )
        end
    end
    if lower_bound == -Inf && upper_bound == Inf
        error("You must specify a finite bound on the cost-to-go term.")
    end
    if length(node.children) == 0
        lower_bound = upper_bound = 0.0
    end
    #Mathis
    Θᴳ = @variable(node.subproblem, base_name = "V_"*string(node.index))
    lower_bound > -Inf && JuMP.set_lower_bound(Θᴳ, lower_bound)
    upper_bound < Inf && JuMP.set_upper_bound(Θᴳ, upper_bound)

    cV= @constraint(node.value_function.model, node.value_function.theta >= lower_bound)
    csp= @constraint(node.subproblem, Θᴳ >= lower_bound)

    cutV = Cut2(
        0.0,
        Dict{Symbol,Float64}(i => 0.0 for (i,x) in node.states),
        0.0,
        cV,
        csp,
    )
    push!(node.value_function.cut_V, cutV)



    # Initialize bounds for the objective states. If objective_state==nothing,
    # this check will be skipped by dispatch.
    _add_initial_bounds(node.objective_state, Θᴳ)
    x′ = Dict(key => var.out for (key, var) in node.states)
    obj_μ = node.objective_state !== nothing ? node.objective_state.μ : nothing
    belief_μ = node.belief_state !== nothing ? node.belief_state.μ : nothing
    return BellmanFunction(
        cut_type,
        ConvexApproximation(Θᴳ, x′, obj_μ, belief_μ, deletion_minimum),
        ConvexApproximation[],
        Set{Vector{Float64}}(),
    )
end

# Internal function: helper used in _add_initial_bounds.
function _add_objective_state_constraint(
    theta::JuMP.VariableRef,
    y::NTuple{N,Float64},
    μ::NTuple{N,JuMP.VariableRef},
) where {N}
    is_finite = [-Inf < y[i] < Inf for i in 1:N]
    model = JuMP.owner_model(theta)
    lower_bound = JuMP.has_lower_bound(theta) ? JuMP.lower_bound(theta) : -Inf
    upper_bound = JuMP.has_upper_bound(theta) ? JuMP.upper_bound(theta) : Inf
    if lower_bound ≈ upper_bound ≈ 0.0
        @constraint(model, [i = 1:N], μ[i] == 0.0)
        return
    end
    expr = @expression(
        model,
        sum(y[i] * μ[i] for i in 1:N if is_finite[i]) + theta
    )
    if lower_bound > -Inf
        @constraint(model, expr >= lower_bound)
    end
    if upper_bound < Inf
        @constraint(model, expr <= upper_bound)
    end
    return
end

# Internal function: When created, θ has bounds of [-M, M], but, since we are
# adding these μ terms, we really want to bound <y, μ> + θ ∈ [-M, M]. We need to
# consider all possible values for `y`. Because the domain of `y` is
# rectangular, we want to add a constraint at each extreme point. This involves
# adding 2^N constraints where N = |μ|. This is only feasible for
# low-dimensional problems, e.g., N < 5.
_add_initial_bounds(::Nothing, ::Any) = nothing

function _add_initial_bounds(obj_state::ObjectiveState, theta)
    if length(obj_state.μ) < 5
        for y in
            Base.product(zip(obj_state.lower_bound, obj_state.upper_bound)...)
            _add_objective_state_constraint(theta, y, obj_state.μ)
        end
    else
        _add_objective_state_constraint(
            theta,
            obj_state.lower_bound,
            obj_state.μ,
        )
        _add_objective_state_constraint(
            theta,
            obj_state.upper_bound,
            obj_state.μ,
        )
    end
    return
end

#Mathis
function refine_value_function(
    model::PolicyGraph{T},
    node::Node{T},
    outgoing_state::Dict{Symbol,Float64},
    items::BackwardPassItems,
    shift::Float64,
) where {T}
    N = length(node.children)
    index_child = Dict(c.term => i for (i, c) in enumerate(node.children))
    πᵏ = [Dict(key => 0.0 for key in keys(model[child.term].states)) for child in node.children]
    θᵏ = zeros(N)
    for (j, c) in enumerate(items.nodes)
        p = items.probability[j]
        θᵏ[index_child[c]] += p * items.objectives[j]
        for (key, dual) in items.duals[j]
            πᵏ[index_child[c]][key] += p * dual
        end
    end

    for (key, x) in outgoing_state
        for c in keys(πᵏ)
            θᵏ[c] -= πᵏ[c][key] * x
        end
    end

    # TV
    for child in node.children
        child_node=model[child.term]
        child_vf=child_node.value_function
        md = child_vf.model_TV
        @constraint(md, child_vf.theta_TV>=θᵏ[index_child[child.term]]+sum(πᵏ[index_child[child.term]][i]*x for (i,x) in child_vf.states_TV))
        # push!(child_node.value_function.cut_TV, Cut2(θᵏ[index_child[child.term]], πᵏ[index_child[child.term]], shift))
    end

    #Attention: quel shift si plusieurs enfants ?
    θᵏ.-= shift

    # println(("cut added to value function :",θᵏ, πᵏ, outgoing_state))

    for child in node.children
        child_node=model[child.term]
        child_vf=child_node.value_function
        md = child_vf.model
        @constraint(md, child_vf.theta>=θᵏ[index_child[child.term]]+sum(πᵏ[index_child[child.term]][i]*x for (i,x) in child_vf.states))
        # push!(child_node.value_function.cut_V, Cut2(θᵏ[index_child[child.term]], πᵏ[index_child[child.term]], shift))
    end
end

function refine_bellman_function(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    risk_measure::AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    cut_selection::Bool,
    shift::Float64,
) where {T}
    lock(node.lock)
    try
        return _refine_bellman_function_no_lock(
            model,
            node,
            bellman_function,
            risk_measure,
            outgoing_state,
            dual_variables,
            noise_supports,
            nominal_probability,
            objective_realizations,
            cut_selection,
            shift,
        )
    finally
        unlock(node.lock)
    end
end

function _refine_bellman_function_no_lock(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    risk_measure::AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    cut_selection::Bool,
    shift::Float64,
) where {T}
    # Sanity checks.
    @assert length(dual_variables) ==
            length(noise_supports) ==
            length(nominal_probability) ==
            length(objective_realizations)
    # Preliminaries that are common to all cut types.
    risk_adjusted_probability = similar(nominal_probability)
    offset = adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        nominal_probability,
        noise_supports,
        objective_realizations,
        model.objective_sense == MOI.MIN_SENSE,
    )
    # The meat of the function.
    if bellman_function.cut_type == SINGLE_CUT
        return _add_average_cut(
            model,
            node,
            outgoing_state,
            risk_adjusted_probability,
            objective_realizations,
            dual_variables,
            offset,
            cut_selection,
            shift,
        )
    else  # Add a multi-cut
        @assert bellman_function.cut_type == MULTI_CUT
        _add_locals_if_necessary(node, bellman_function, length(dual_variables))
        return _add_multi_cut(
            node,
            outgoing_state,
            risk_adjusted_probability,
            objective_realizations,
            dual_variables,
            offset,
        )
    end
end

#Mathis attention il faudrait un shift pour chaque enfant
function inf_shift(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64,
) where {T}
    res=0.0
    for child in node.children
        child_node=model[child.term]
        inf_TV, sol = compute_inf_TV(child_node.two_stage)
        V_x=compute_V(child_node.value_function, sol)

        res+=inf_TV-V_x
    end
    return res
end

function approx_inf_shift(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64,
) where {T}
    res=0.0
    for child in node.children
        child_node=model[child.term]
        _, sol = compute_inf_approx_TV(child_node.value_function)
        V_x=compute_V(child_node.value_function, sol)
        
        # inf_TV, sol = compute_inf_TV(child_node.two_stage)
        inf_TV=compute_TV(child_node, sol)

        DCAs = DCA_shift(model, node, bellman_function, state, θᵏ)
        res+=min(DCAs, inf_TV-V_x, θᵏ-compute_V(child_node.value_function, state))
        # if child_node.index == 1
        #     # DCAs2 = DCA_shift(model, node, bellman_function, sol, θᵏ)
        #     println((DCAs, inf_TV-V_x, res, sol, state))
        # end
    end
    return res
end

function current_shift(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64
) where {T}
    res=0.0
    for child in node.children
        child_node=model[child.term]

        TVx=θᵏ
        Vx=compute_V(child_node.value_function, state)

        res+=TVx-Vx
    end
    return res
end

function update_shift(
    model::PolicyGraph{T},
    node::Node{T},
    shift_k::Float64,
) where {T}
    for cut in node.value_function.cut_V
        if shift_k<= cut.shift
            cut.intercept += cut.shift-shift_k
            cut.shift = shift_k
            set_normalized_rhs(cut.constraint_V, cut.intercept)
            set_normalized_rhs(cut.constraint_subproblem, cut.intercept)
        end
    end
end

function current_shift2(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64
) where {T}
    res=0.0
    for child in node.children
        child_node=model[child.term]

        TVx=θᵏ
        Vx=compute_V(child_node.value_function, state)

        res+=TVx-Vx
        update_shift(model, child_node, res)
        # println(res)
    end
    return res
end

function DCA_shift(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64
) where {T}
    res=0.0
    for child in node.children
        child_node=model[child.term]
        sol = DCA(child_node.value_function, state)
        TVx=compute_TV(child_node, sol)
        Vx=compute_V(child_node.value_function, sol)

        # if node.index == 3
        #     println((TVx-Vx, state, sol))
        # end

        res+=TVx-Vx
    end
    return res
end

function random_shift(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64,
) where {T}
    res=0.0
    sol=Dict{Symbol,Float64}()

    for child in node.children
        child_node=model[child.term]
        two_stage=child_node.two_stage
        for (i,x) in state
            lb=two_stage.lower_bounds[i]
            ub=two_stage.upper_bounds[i]
            sol[i]=rand()*(ub-lb)+lb
        end
        # println("000")
        # DCAs = DCA_shift(model, node, bellman_function, state, θᵏ)
        # prinln(111)
        # res+=DCAs
        TVx=compute_TV(child_node, sol)
        Vx=compute_V(child_node.value_function, sol)

        # res+=min(DCAs,TVx-Vx)
        res+=min(TVx-Vx)
        # res+=min(TVx-Vx, θᵏ-compute_V(child_node.value_function, state)) 

    end
    return res
end

function random_shift2(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64,
) where {T}
    res=0.0
    sol=Dict{Symbol,Float64}()

    for child in node.children
        child_node=model[child.term]
        two_stage=child_node.two_stage
        for (i,x) in state
            lb=two_stage.lower_bounds[i]
            ub=two_stage.upper_bounds[i]
            sol[i]=rand()*(ub-lb)+lb
        end
        # println("000")
        # DCAs = DCA_shift(model, node, bellman_function, state, θᵏ)
        # prinln(111)
        # res+=DCAs
        TVx=compute_TV(child_node, sol)
        Vx=compute_V(child_node.value_function, sol)

        # res+=min(DCAs,TVx-Vx)
        res+=min(TVx-Vx, θᵏ-compute_V(child_node.value_function, state)) 
        update_shift(model, child_node, res)

    end
    return res
end

function best_shift(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64,
) where {T}
    # if node.index == 3
    #     (a,b,c) = (compute_current_shift(model, node, bellman_function, state), compute_inf_shift(model, node, bellman_function, state), compute_random_shift(model, node, bellman_function, state))
    #     # println((a,b,c))
    # end
    a = compute_current_shift(model, node, bellman_function, state, θᵏ)
    # b = compute_inf_shift(model, node, bellman_function, state, θᵏ)
    c = compute_random_shift(model, node, bellman_function, state, θᵏ)
    if a<=c
        # println((node.index, "current shift is the best: $(c-a)", a, c, state))
    end
    return minimum((a, c))
end

function no_shift(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
    θᵏ::Float64,
) where {T}
    return 0.0
end

function _add_average_cut(
    model::PolicyGraph{T},
    node::Node{T},
    outgoing_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
    cut_selection::Bool,
    shift::Float64,
) where {T}
    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables)
    # Calculate the expected intercept and dual variables with respect to the
    # risk-adjusted probability distribution.
    πᵏ = Dict(key => 0.0 for key in keys(outgoing_state))
    θᵏ = offset
    for i in 1:length(objective_realizations)
        p = risk_adjusted_probability[i]
        θᵏ += p * objective_realizations[i]
        for (key, dual) in dual_variables[i]
            πᵏ[key] += p * dual
        end
    end
    # Now add the average-cut to the subproblem. We include the objective-state
    # component μᵀy and the belief state (if it exists).
    obj_y =
        node.objective_state === nothing ? nothing : node.objective_state.state
    belief_y =
        node.belief_state === nothing ? nothing : node.belief_state.belief
    _add_cut(
        model,
        node, 
        node.bellman_function.global_theta,
        θᵏ,
        shift,
        πᵏ,
        outgoing_state,
        obj_y,
        belief_y,
        cut_selection=cut_selection,
    )
    # Mathis faudrait pouvoir le désactiver
    # _add_cut_to_two_stage_model(
    #     node,
    #     θᵏ,
    #     πᵏ,
    #     outgoing_state,
    #     obj_y,
    #     belief_y,
    # )
    return (
        theta = θᵏ,
        pi = πᵏ,
        x = outgoing_state,
        obj_y = obj_y,
        belief_y = belief_y,
    )
end

function _add_multi_cut(
    node::Node,
    outgoing_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
)
    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables)
    bellman_function = node.bellman_function
    μᵀy = get_objective_state_component(node)
    JuMP.add_to_expression!(μᵀy, get_belief_state_component(node))
    for i in 1:length(dual_variables)
        _add_cut(
            bellman_function.local_thetas[i],
            objective_realizations[i],
            dual_variables[i],
            outgoing_state,
            node.objective_state === nothing ? nothing :
            node.objective_state.state,
            node.belief_state === nothing ? nothing : node.belief_state.belief,
        )
    end
    model = JuMP.owner_model(bellman_function.global_theta.theta)
    cut_expr = @expression(
        model,
        sum(
            risk_adjusted_probability[i] *
            bellman_function.local_thetas[i].theta for i in 1:N
        ) - (1 - sum(risk_adjusted_probability)) * μᵀy + offset
    )
    # TODO(odow): should we use `cut_expr` instead?
    ξ = copy(risk_adjusted_probability)
    if !(ξ in bellman_function.risk_set_cuts) || μᵀy != JuMP.AffExpr(0.0)
        push!(bellman_function.risk_set_cuts, ξ)
        if JuMP.objective_sense(model) == MOI.MIN_SENSE
            @constraint(model, bellman_function.global_theta.theta >= cut_expr)
        else
            @constraint(model, bellman_function.global_theta.theta <= cut_expr)
        end
    end
    return
end

# If we are adding a multi-cut for the first time, then the local θ variables
# won't have been added.
# TODO(odow): a way to set different bounds for each variable in the multi-cut.
function _add_locals_if_necessary(
    node::Node,
    bellman_function::BellmanFunction,
    N::Int,
)
    num_local_thetas = length(bellman_function.local_thetas)
    if num_local_thetas == N
        return # Do nothing. Already initialized.
    elseif num_local_thetas > 0
        error(
            "Expected $(N) local θ variables but there were " *
            "$(num_local_thetas).",
        )
    end
    global_theta = bellman_function.global_theta
    model = JuMP.owner_model(global_theta.theta)
    local_thetas = @variable(model, [1:N])
    if JuMP.has_lower_bound(global_theta.theta)
        JuMP.set_lower_bound.(
            local_thetas,
            JuMP.lower_bound(global_theta.theta),
        )
    end
    if JuMP.has_upper_bound(global_theta.theta)
        JuMP.set_upper_bound.(
            local_thetas,
            JuMP.upper_bound(global_theta.theta),
        )
    end
    for local_theta in local_thetas
        push!(
            bellman_function.local_thetas,
            ConvexApproximation(
                local_theta,
                global_theta.states,
                node.objective_state === nothing ? nothing :
                node.objective_state.μ,
                node.belief_state === nothing ? nothing : node.belief_state.μ,
                global_theta.deletion_minimum,
            ),
        )
    end
    return
end

"""
    write_cuts_to_file(
        model::PolicyGraph{T},
        filename::String;
        kwargs...,
    ) where {T}

Write the cuts that form the policy in `model` to `filename` in JSON format.

## Keyword arguments

 - `node_name_parser` is a function which converts the name of each node into a
    string representation. It has the signature: `node_name_parser(::T)::String`.

 - `write_only_selected_cuts` write only the selected cuts to the json file.
    Defaults to false.

See also [`SDDP.read_cuts_from_file`](@ref).
"""
function write_cuts_to_file(
    model::PolicyGraph{T},
    filename::String;
    node_name_parser::Function = string,
    write_only_selected_cuts::Bool = false,
) where {T}
    cuts = Dict{String,Any}[]
    for (node_name, node) in model.nodes
        if node.objective_state !== nothing || node.belief_state !== nothing
            error(
                "Unable to write cuts to file because model contains " *
                "objective states or belief states.",
            )
        end
        node_cuts = Dict(
            "node" => node_name_parser(node_name),
            "single_cuts" => Dict{String,Any}[],
            "multi_cuts" => Dict{String,Any}[],
            "risk_set_cuts" => Vector{Float64}[],
        )
        oracle = node.bellman_function.global_theta
        for (cut, state) in zip(oracle.cuts, oracle.sampled_states)
            if write_only_selected_cuts && cut.constraint_ref === nothing
                continue
            end
            intercept = cut.intercept
            for (key, π) in cut.coefficients
                intercept += π * state.state[key]
            end
            push!(
                node_cuts["single_cuts"],
                Dict(
                    "intercept" => intercept,
                    "coefficients" => copy(cut.coefficients),
                    "state" => copy(state.state),
                ),
            )
        end
        for (i, theta) in enumerate(node.bellman_function.local_thetas)
            for (cut, state) in zip(theta.cuts, theta.sampled_states)
                if write_only_selected_cuts && cut.constraint_ref === nothing
                    continue
                end
                intercept = cut.intercept
                for (key, π) in cut.coefficients
                    intercept += π * state.state[key]
                end
                push!(
                    node_cuts["multi_cuts"],
                    Dict(
                        "realization" => i,
                        "intercept" => intercept,
                        "coefficients" => copy(cut.coefficients),
                        "state" => copy(state.state),
                    ),
                )
            end
        end
        for p in node.bellman_function.risk_set_cuts
            push!(node_cuts["risk_set_cuts"], p)
        end
        push!(cuts, node_cuts)
    end
    open(filename, "w") do io
        return write(io, JSON.json(cuts))
    end
    return
end

_node_name_parser(::Type{Int}, name::String) = parse(Int, name)

_node_name_parser(::Type{Symbol}, name::String) = Symbol(name)

function _node_name_parser(::Type{NTuple{N,Int}}, name::String) where {N}
    keys = parse.(Int, strip.(split(name[2:end-1], ",")))
    if length(keys) != N
        error("Unable to parse node called $(name). Expected $N elements.")
    end
    return tuple(keys...)
end

function _node_name_parser(::Any, name)
    return error(
        "Unable to read name $(name). Provide a custom parser to " *
        "`read_cuts_from_file` using the `node_name_parser` keyword.",
    )
end

"""
    read_cuts_from_file(
        model::PolicyGraph{T},
        filename::String;
        kwargs...,
    ) where {T}

Read cuts (saved using [`SDDP.write_cuts_to_file`](@ref)) from `filename` into
`model`.

Since `T` can be an arbitrary Julia type, the conversion to JSON is lossy. When
reading, `read_cuts_from_file` only supports `T=Int`, `T=NTuple{N, Int}`, and
`T=Symbol`. If you have manually created a policy graph with a different node
type `T`, provide a function `node_name_parser` with the signature

## Keyword arguments

 - `node_name_parser(T, name::String)::T where {T}` that returns the name of each
    node given the string name `name`.
    If `node_name_parser` returns `nothing`, those cuts are skipped.

 - `cut_selection::Bool` run or not the cut selection algorithm when adding the
    cuts to the model.

See also [`SDDP.write_cuts_to_file`](@ref).
"""
function read_cuts_from_file(
    model::PolicyGraph{T},
    filename::String;
    node_name_parser::Function = _node_name_parser,
    cut_selection::Bool = true,
) where {T}
    cuts = JSON.parsefile(filename; use_mmap = false)
    for node_cuts in cuts
        node_name = node_name_parser(T, node_cuts["node"])::Union{Nothing,T}
        if node_name === nothing
            continue  # Skip reading these cuts
        end
        node = model[node_name]
        bf = node.bellman_function
        # Loop through and add the single-cuts.
        for json_cut in node_cuts["single_cuts"]
            has_state = haskey(json_cut, "state")
            state = if has_state
                Dict(Symbol(k) => v for (k, v) in json_cut["state"])
            else
                Dict(Symbol(k) => 0.0 for k in keys(json_cut["coefficients"]))
            end
            _add_cut(
                bf.global_theta,
                json_cut["intercept"],
                Dict(Symbol(k) => v for (k, v) in json_cut["coefficients"]),
                state,
                nothing,
                nothing;
                cut_selection = (cut_selection && has_state),
            )
        end
        # Loop through and add the multi-cuts. There are two parts:
        #  (i) the cuts w.r.t. the state variable x
        # (ii) the cuts that define the risk set
        # There is one additional complication: if these cuts are being read
        # into a new model, the local theta variables may not exist yet.
        if length(node_cuts["risk_set_cuts"]) > 0
            _add_locals_if_necessary(
                node,
                bf,
                length(first(node_cuts["risk_set_cuts"])),
            )
        end
        for json_cut in node_cuts["multi_cuts"]
            has_state = haskey(json_cut, "state")
            state = if has_state
                Dict(Symbol(k) => v for (k, v) in json_cut["state"])
            else
                Dict(Symbol(k) => 0.0 for k in keys(json_cut["coefficients"]))
            end
            _add_cut(
                bf.local_thetas[json_cut["realization"]],
                json_cut["intercept"],
                Dict(Symbol(k) => v for (k, v) in json_cut["coefficients"]),
                state,
                nothing,
                nothing;
                cut_selection = (cut_selection && has_state),
            )
        end
        # Here is part (ii): adding the constraints that define the risk-set
        # representation of the risk measure.
        for json_cut in node_cuts["risk_set_cuts"]
            expr = @expression(
                node.subproblem,
                bf.global_theta.theta - sum(
                    p * V.theta for (p, V) in zip(json_cut, bf.local_thetas)
                )
            )
            if JuMP.objective_sense(node.subproblem) == MOI.MIN_SENSE
                @constraint(node.subproblem, expr >= 0)
            else
                @constraint(node.subproblem, expr <= 0)
            end
        end
    end
    return
end

"""
    add_all_cuts(model::PolicyGraph)

Add all cuts that may have been deleted back into the model.

## Explanation

During the solve, SDDP.jl may decide to remove cuts for a variety of reasons.

These can include cuts that define the optimal value function, particularly
around the extremes of the state-space (e.g., reservoirs empty).

This function ensures that all cuts discovered are added back into the model.

You should call this after [`train`](@ref) and before [`simulate`](@ref).
"""
function add_all_cuts(model::PolicyGraph)
    for node in values(model.nodes)
        global_theta = node.bellman_function.global_theta
        for cut in global_theta.cuts
            if cut.constraint_ref === nothing
                _add_cut_constraint_to_model(global_theta, cut)
            end
        end
        for approximation in node.bellman_function.local_thetas
            for cut in approximation.cuts
                if cut.constraint_ref === nothing
                    _add_cut_constraint_to_model(approximation, cut)
                end
            end
        end
    end
    return
end
