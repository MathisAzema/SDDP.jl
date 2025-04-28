#Mathis

function add_scenario_to_two_stage(
    model::JuMP.Model,
    two_stage::TwoStage,
    child::ScenarioTreeNode,
    s::Int64,
    check_time_limit::Function,
)
    check_time_limit()
    node = child.node
    parameterize(node, child.noise)
    # Add variables:
    src_variables = JuMP.all_variables(node.subproblem)
    x = @variable(model, [1:length(src_variables)])
    var_src_to_dest = Dict{JuMP.VariableRef,JuMP.VariableRef}()
    for (src, dest) in zip(src_variables, x)
        var_src_to_dest[src] = dest
        name = JuMP.name(src)
        if !isempty(name)
            # append node index to original variable name
            JuMP.set_name(dest, string(name, "#", s))
            if name==JuMP.name(node.bellman_function.global_theta.theta)
                two_stage.bellman_variables[s] = dest
            end
        else
            # append node index to original variable index
            var_name = string("_[", index(src).value, "]")
            JuMP.set_name(dest, string(var_name, "#", s))
        end
        for (i,x) in node.states
            if name==JuMP.name(x.out)
                two_stage.states[i][s]=dest
            end
        end
    end
    # Add constraints:
    for (F, S) in JuMP.list_of_constraint_types(node.subproblem)
        for con in JuMP.all_constraints(node.subproblem, F, S)
            obj = JuMP.constraint_object(con)
            new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
            @constraint(model, new_func in obj.set)
        end
    end
    # Add objective:
    current = JuMP.objective_function(model)
    subproblem_objective =
        copy_and_replace_variables(objective_function(node.subproblem), var_src_to_dest)
    JuMP.set_objective_function(
        model,
        current + child.probability * subproblem_objective,
    )
    return
end

function add_next_node_to_scenario_tree(
    parent::Vector{ScenarioTreeNode{T}},
    pg::PolicyGraph{T},
    node::Node{T},
    check_time_limit::Function,
) where {T}
    if node.objective_state !== nothing
        throw_detequiv_error("Objective states detected!")
    elseif node.belief_state !== nothing
        throw_detequiv_error("Belief states detected!")
    elseif length(node.bellman_function.global_theta.cuts) > 0
        throw_detequiv_error(
            "Model has been used for training. Can only form deterministic " *
            "equivalent on a fresh model.",
        )
    else
        check_time_limit()
    end
    for noise in node.noise_terms
        scenario_node = ScenarioTreeNode(
            node,
            noise.term,
            noise.probability,
            ScenarioTreeNode{T}[],
            Dict{Symbol,State{JuMP.VariableRef}}(),
        )
        push!(parent, scenario_node)
    end
    return
end

function initialize_value_function(
    sense::Symbol,
    optimizer = nothing
)
    model = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)
    set_silent(model)
    theta = @variable(model, V)
    if sense == :Min
        @objective(model, Min, theta)
    else
        @objective(model, Max, theta)
    end

    #TV
    model_TV = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)
    set_silent(model_TV)
    theta_TV = @variable(model_TV, TV)
    if sense == :Min
        @objective(model_TV, Min, theta_TV)
    else
        @objective(model_TV, Max, theta_TV)
    end

    return Value_Function(
        model,
        Cut2[],
        theta,
        Dict{Symbol,JuMP.VariableRef}(),
        model_TV,
        theta_TV,
        Dict{Symbol,JuMP.VariableRef}(),
    )
end

function add_state_variables_to_value_function(
    node::Node
)
    value_function=node.value_function

    y = @variable(value_function.model, [1:length(node.states)])
    state_variables=[(key,var.out) for (key,var) in node.states]
    for (src, dest) in zip(state_variables, y)
        name = JuMP.name(src[2]) #MATHIS: checker si name exist
        JuMP.set_name(dest, name)
        value_function.states[src[1]]=dest
        lb=node.two_stage.lower_bounds[src[1]]
        ub=node.two_stage.upper_bounds[src[1]]
        @constraint(value_function.model, dest >= lb)
        @constraint(value_function.model, dest <= ub)
    end

    # cV= @constraint(value_function.model, value_function.theta >= 0.0)

    # cutV = Cut2(
    #     0.0,
    #     Dict{Symbol,Float64}(i => 0.0 for (i,x) in value_function.states),
    #     0.0,
    #     cV,
    # )
    # push!(value_function.cut_V, cutV)

    #TV
    y_TV = @variable(value_function.model_TV, [1:length(node.states)])
    state_variables=[(key,var.out) for (key,var) in node.states]
    for (src, dest) in zip(state_variables, y_TV)
        name = JuMP.name(src[2]) #MATHIS: checker si name exist
        JuMP.set_name(dest, name)
        value_function.states_TV[src[1]]=dest
        lb=node.two_stage.lower_bounds[src[1]]
        ub=node.two_stage.upper_bounds[src[1]]
        @constraint(value_function.model_TV, dest >= lb)
        @constraint(value_function.model_TV, dest <= ub)
    end

    # cTV= @constraint(value_function.model_TV, value_function.theta_TV >= 0.0)

    # cutTV = Cut2(
    #     0.0,
    #     Dict{Symbol,Float64}(i => 0.0 for (i,x) in value_function.states),
    #     0.0,
    #     cTV,
    # )
    # push!(value_function.cut_TV, cutTV)
end

function initialize_two_stage(
    pg::PolicyGraph{T},
    node::Node,
    optimizer = nothing;
    time_limit::Union{Real,Nothing} = 60.0,
) where {T}
    two_stage=node.two_stage
    start_time = time()
    time_limit = time_limit === nothing ? typemax(Float64) : Float64(time_limit)
    function check_time_limit()
        if time() - start_time > time_limit::Float64
            throw_detequiv_error("Time limit exceeded!")
        end
    end
    tree = ScenarioTree{T}(ScenarioTreeNode{T}[])
    add_next_node_to_scenario_tree(
        tree.children,
        pg,
        node,
        check_time_limit,
    )
    for (i,x) in node.states
        two_stage.states[i]=Vector{VariableRef}(undef, length(tree.children))
    end
    two_stage.bellman_variables=Vector{VariableRef}(undef, length(tree.children))
    
    model = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)
    set_silent(model)
    set_objective_sense(model, pg.objective_sense)

    for (s, child) in enumerate(tree.children)
        add_scenario_to_two_stage(model, two_stage, child, s, check_time_limit)
    end

    y = @variable(model, [1:length(node.states)])
    incoming_variables=[(key,var) for (key,var) in node.states]
    for (src, dest) in zip(incoming_variables, y)
        name = JuMP.name(src[2].in) #MATHIS: checker si name exist
        JuMP.set_name(dest, string("non_anticipative_", name))
        two_stage.non_anticipative_variables[src[1]]=dest
        lb=lower_bound(src[2].out)
        ub=upper_bound(src[2].out)
        @constraint(model, dest >= lb)
        @constraint(model, dest <= ub) # Mathis : borne in et out sont les mêmes ?
        two_stage.lower_bounds[src[1]]=lb
        two_stage.upper_bounds[src[1]]=ub
        for (s, child) in enumerate(tree.children)
            var_s=variable_by_name(model, string(name, "#", s))
            expr=JuMP.GenericAffExpr(
                0.0,
                Pair{VariableRef,Float64}[
                    dest => 1.0,
                    var_s => -1.0,
                ],
            )
            @constraint(model, expr==0.0)
        end
    end
    two_stage.model=model
    return 
end

function compute_inf_TV(
    two_stage::TwoStage,
)
    mod=two_stage.model
    JuMP.optimize!(mod)
    inf_TV=JuMP.objective_value(mod)
    solution=Dict(i => value.(x) for (i,x) in two_stage.non_anticipative_variables)
    return (inf_TV, solution)
end

# function compute_TV(
#     two_stage::TwoStage,
#     incoming_state::Dict{Symbol,Float64}
# )
#     model=two_stage.model
#     for (i, value) in incoming_state
#         JuMP.fix(two_stage.non_anticipative_variables[i], value; force=true)
#     end
#     JuMP.optimize!(model)
#     obj=JuMP.objective_value(model)
#     for (i, value) in incoming_state
#         JuMP.unfix(two_stage.non_anticipative_variables[i])
#     end
#     return obj
# end

function compute_TV(
    node::Node,
    incoming_state::Dict{Symbol,Float64}
)
    TVx=0.0
    model=node.subproblem
    set_incoming_state(node, incoming_state)
    for noise in node.noise_terms
        parameterize(node, noise.term)
        JuMP.optimize!(model)
        TVx += noise.probability * JuMP.objective_value(model)
    end
    return TVx
end

# function compute_V(
#     vf::Value_Function,
#     incoming_state::Dict{Symbol,Float64}
# )
#     model=vf.model
#     for (i, value) in incoming_state
#         JuMP.fix(vf.states[i], value; force=true)
#     end
#     JuMP.optimize!(model)
#     obj=JuMP.objective_value(model)
#     for (i, value) in incoming_state
#         JuMP.unfix(vf.states[i])
#     end
#     return obj
# end

function compute_V(
    vf::Value_Function,
    incoming_state::Dict{Symbol,Float64}
)
    val = maximum([cut.intercept + sum(cut.coefficients[i] * x for (i,x) in incoming_state) for cut in vf.cut_V])
    return val
end


# function compute_approx_TV(
#     vf::Value_Function,
#     incoming_state::Dict{Symbol,Float64}
# )
#     model=vf.model_TV
#     for (i, value) in incoming_state
#         JuMP.fix(vf.states_TV[i], value; force=true)
#     end
#     JuMP.optimize!(model)
#     obj=JuMP.objective_value(model)
#     for (i, value) in incoming_state
#         JuMP.unfix(vf.states_TV[i])
#     end
#     return obj
# end

function compute_approx_TV(
    vf::Value_Function,
    incoming_state::Dict{Symbol,Float64}
)
    val = maximum([cut.intercept+cut.shift + sum(cut.coefficients[i] * x for (i,x) in incoming_state) for cut in vf.cut_V])
    return val
end

function compute_inf_approx_TV(
    vf::Value_Function,
)
    mod=vf.model_TV
    JuMP.optimize!(mod)
    inf_TV_k=JuMP.objective_value(mod)
    solution=Dict(i => value.(x) for (i,x) in vf.states_TV)
    return (inf_TV_k, solution)
end

# function compute_V2(
#     vf::Value_Function,
#     incoming_state::Dict{Symbol,Float64}
# )
#     model=vf.model
#     for (i, value) in incoming_state
#         JuMP.fix(vf.states[i], value; force=true)
#     end
#     JuMP.optimize!(model)
#     obj=JuMP.objective_value(model)

#     dual_sign = JuMP.objective_sense(model) == MOI.MIN_SENSE ? 1 : -1
#     λ = Dict{Symbol,Float64}(
#         name => dual_sign * JuMP.dual(JuMP.FixRef(state)) for
#         (name, state) in vf.states
#     )


#     for (i, value) in incoming_state
#         JuMP.unfix(vf.states[i])
#     end
#     return obj, λ
# end

function compute_V2(
    vf::Value_Function,
    incoming_state::Dict{Symbol,Float64}
)
    val, idx = findmax([cut.intercept + sum(cut.coefficients[i] * x for (i,x) in incoming_state) for cut in vf.cut_V])
    return val, vf.cut_V[idx].coefficients
end

function DCA(
    vf::Value_Function,
    incoming_state::Dict{Symbol,Float64}
)
    xk=incoming_state
    vk, gh_k = compute_V2(vf, xk)
    tvk = SDDP.compute_approx_TV(vf, xk)
    k=0
    model = vf.model_TV
    primal_obj = JuMP.objective_function(model)
    while true
        JuMP.set_objective_function(
            model,
            @expression(model, primal_obj - sum(gh_k[name] * state for (name, state) in vf.states_TV)),
        )
        JuMP.optimize!(model)
        tvk=JuMP.objective_value(model)
        xk1=Dict(i => value.(x) for (i,x) in vf.states_TV)
        vk, gh_k = compute_V2(vf, xk1)
        tvk = SDDP.compute_approx_TV(vf, xk1)
        k+=1
        if sum(abs.(xk1[i] - xk[i]) for i in keys(xk)) < 1e-4 || k > 0
            # if k>2
            #     println(k)
            # end
            JuMP.set_objective_function(
                model,
                @expression(model, primal_obj),
            )
            return xk
        else
            xk = xk1
        end
    end
end