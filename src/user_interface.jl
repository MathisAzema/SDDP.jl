#  Copyright (c) 2017-25, Oscar Dowson and SDDP.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct Graph{T}
    # The root node of the policy graph.
    root_node::T
    # nodes[x] returns a vector of the children of node x and their
    # probabilities.
    nodes::Dict{T,Vector{Tuple{T,Float64}}}
    # A partition of the nodes into ambiguity sets.
    belief_partition::Vector{Vector{T}}
    belief_lipschitz::Vector{Vector{Float64}}
end

"""
    Graph(root_node::T) where T

Create an empty graph struture with the root node `root_node`.

## Example

```jldoctest
julia> graph = SDDP.Graph(0)
Root
 0
Nodes
 {}
Arcs
 {}

julia> graph = SDDP.Graph(:root)
Root
 root
Nodes
 {}
Arcs
 {}

julia> graph = SDDP.Graph((0, 0))
Root
 (0, 0)
Nodes
 {}
Arcs
 {}
```
"""
function Graph(root_node::T) where {T}
    return Graph{T}(
        root_node,
        Dict{T,Vector{Tuple{T,Float64}}}(root_node => Tuple{T,Float64}[]),
        Vector{T}[],
        Vector{Float64}[],
    )
end

# Helper utilities to sort the nodes for printing. This helps linear and
# Markovian policy graphs where the nodes might be stored in an unusual ordering
# in the dictionary.
sort_nodes(nodes::Vector{Int}) = sort!(nodes)
sort_nodes(nodes::Vector{Tuple{Int,Int}}) = sort!(nodes)
sort_nodes(nodes::Vector{Tuple{Int,Float64}}) = sort!(nodes)
sort_nodes(nodes::Vector{Symbol}) = sort!(nodes)
sort_nodes(nodes) = nodes

function Base.show(io::IO, graph::Graph)
    println(io, "Root")
    println(io, " ", graph.root_node)
    println(io, "Nodes")
    nodes = sort_nodes(collect(keys(graph.nodes)))
    if first(nodes) != graph.root_node
        splice!(nodes, findfirst(isequal(graph.root_node), nodes))
        prepend!(nodes, [graph.root_node])
    end
    tree_nodes = filter(n -> n != graph.root_node, nodes)
    if isempty(tree_nodes)
        println(io, " {}")
    else
        for node in tree_nodes
            println(io, " ", node)
        end
    end
    print(io, "Arcs")
    has_arc = false
    for node in nodes
        for (child, probability) in graph.nodes[node]
            print(io, "\n ", node, " => ", child, " w.p. ", probability)
            has_arc = true
        end
    end
    if !has_arc
        print(io, "\n {}")
    end
    if length(graph.belief_partition) > 0
        print(io, "\nPartitions")
        for element in graph.belief_partition
            print(io, "\n {", join(string.(sort_nodes(element)), ", "), "}")
        end
    end
    return
end

# Internal function used to validate the structure of a graph
function _validate_graph(graph::Graph)
    for (node, children) in graph.nodes
        if length(children) > 0
            probability = sum(child[2] for child in children)
            if !(-1e-8 <= probability <= 1.0 + 1e-8)
                error(
                    "Probability on edges leaving node $(node) sum to " *
                    "$(probability), but this must be in [0.0, 1.0]",
                )
            end
        end
    end
    if length(graph.belief_partition) > 0
        # The -1 accounts for the root node, which shouldn't be in the
        # partition.
        if graph.root_node in union(graph.belief_partition...)
            error(
                "Belief partition $(graph.belief_partition) cannot contain " *
                "the root node $(graph.root_node).",
            )
        end
        if length(graph.nodes) - 1 != length(union(graph.belief_partition...))
            error(
                "Belief partition $(graph.belief_partition) does not form a" *
                " valid partition of the nodes in the graph.",
            )
        end
    end
    return
end

"""
    add_node(graph::Graph{T}, node::T) where {T}

Add a node to the graph `graph`.

## Examples

```jldoctest
julia> graph = SDDP.Graph(:root);

julia> SDDP.add_node(graph, :A)

julia> graph
Root
 root
Nodes
 A
Arcs
 {}
```

```jldoctest
julia> graph = SDDP.Graph(0);

julia> SDDP.add_node(graph, 2)

julia> graph
Root
 0
Nodes
 2
Arcs
 {}
```
"""
function add_node(graph::Graph{T}, node::T) where {T}
    if haskey(graph.nodes, node) || node == graph.root_node
        error("Node $(node) already exists!")
    end
    graph.nodes[node] = Tuple{T,Float64}[]
    return
end

function add_node(graph::Graph{T}, node) where {T}
    return error("Unable to add node $(node). Nodes must be of type $(T).")
end

function _add_node_if_missing(graph::Graph{T}, node::T) where {T}
    if haskey(graph.nodes, node) || node == graph.root_node
        return
    end
    return add_node(graph, node)
end

"""
    add_edge(graph::Graph{T}, edge::Pair{T, T}, probability::Float64) where {T}

Add an edge to the graph `graph`.

## Examples

```jldoctest
julia> graph = SDDP.Graph(0);

julia> SDDP.add_node(graph, 1)

julia> SDDP.add_edge(graph, 0 => 1, 0.9)

julia> graph
Root
 0
Nodes
 1
Arcs
 0 => 1 w.p. 0.9
```

```jldoctest
julia> graph = SDDP.Graph(:root);

julia> SDDP.add_node(graph, :A)

julia> SDDP.add_edge(graph, :root => :A, 1.0)

julia> graph
Root
 root
Nodes
 A
Arcs
 root => A w.p. 1.0
```
"""
function add_edge(
    graph::Graph{T},
    edge::Pair{T,T},
    probability::Float64,
) where {T}
    (parent, child) = edge
    if !(parent == graph.root_node || haskey(graph.nodes, parent))
        error("Node $(parent) does not exist.")
    elseif !haskey(graph.nodes, child)
        error("Node $(child) does not exist.")
    elseif child == graph.root_node
        error("Cannot have an edge entering the root node.")
    else
        push!(graph.nodes[parent], (child, probability))
    end
    return
end

function _add_to_or_create_edge(
    graph::Graph{T},
    edge::Pair{T,T},
    probability::Float64,
) where {T}
    for (i, (child, p)) in enumerate(graph.nodes[edge[1]])
        if child == edge[2]
            graph.nodes[edge[1]][i] = (edge[2], p + probability)
            return
        end
    end
    return add_edge(graph, edge, probability)
end

"""
    add_ambiguity_set(
        graph::Graph{T},
        set::Vector{T},
        lipschitz::Vector{Float64},
    ) where {T}

Add `set` to the belief partition of `graph`.

`lipschitz` is a vector of Lipschitz constants, with one element for each node
in `set`. The Lipschitz constant is the maximum slope of the cost-to-go function
with respect to the belief state associated with each node at any point in the
state-space.

## Examples

```julia
julia> graph = SDDP.LinearGraph(3)
Root
 0
Nodes
 1
 2
 3
Arcs
 0 => 1 w.p. 1.0
 1 => 2 w.p. 1.0
 2 => 3 w.p. 1.0

julia> SDDP.add_ambiguity_set(graph, [1, 2], [1e3, 1e2])

julia> SDDP.add_ambiguity_set(graph, [3], [1e5])

julia> graph
Root
 0
Nodes
 1
 2
 3
Arcs
 0 => 1 w.p. 1.0
 1 => 2 w.p. 1.0
 2 => 3 w.p. 1.0
Partitions
 {1, 2}
 {3}
```
"""
function add_ambiguity_set(
    graph::Graph{T},
    set::Vector{T},
    lipschitz::Vector{Float64},
) where {T}
    if any(l -> l < 0.0, lipschitz)
        error("Cannot provide negative Lipschitz constant: $(lipschitz)")
    elseif length(set) != length(lipschitz)
        error(
            "You must provide on Lipschitz contsant for every element in " *
            "the ambiguity set.",
        )
    end
    push!(graph.belief_partition, set)
    push!(graph.belief_lipschitz, lipschitz)
    return
end

"""
    add_ambiguity_set(graph::Graph{T}, set::Vector{T}, lipschitz::Float64)

Add `set` to the belief partition of `graph`.

`lipschitz` is a Lipschitz constant for each node in `set`. The Lipschitz
constant is the maximum slope of the cost-to-go function with respect to the
belief state associated with each node at any point in the state-space.

## Examples

```julia
julia> graph = SDDP.LinearGraph(3);

julia> SDDP.add_ambiguity_set(graph, [1, 2], 1e3)

julia> SDDP.add_ambiguity_set(graph, [3], 1e5)

julia> graph
Root
 0
Nodes
 1
 2
 3
Arcs
 0 => 1 w.p. 1.0
 1 => 2 w.p. 1.0
 2 => 3 w.p. 1.0
Partitions
 {1, 2}
 {3}
```
"""
function add_ambiguity_set(
    graph::Graph{T},
    set::Vector{T},
    lipschitz::Float64 = 1e5,
) where {T}
    return add_ambiguity_set(graph, set, fill(lipschitz, length(set)))
end

function Graph(
    root_node::T,
    nodes::Vector{T},
    edges::Vector{Tuple{Pair{T,T},Float64}};
    belief_partition::Vector{Vector{T}} = Vector{T}[],
    belief_lipschitz::Vector{Vector{Float64}} = Vector{Float64}[],
) where {T}
    graph = Graph(root_node)
    add_node.(Ref(graph), nodes)
    for (edge, probability) in edges
        add_edge(graph, edge, probability)
    end
    add_ambiguity_set.(Ref(graph), belief_partition, belief_lipschitz)
    return graph
end

"""
    LinearGraph(stages::Int)

Create a linear graph with `stages` number of nodes.

## Examples

```jldoctest
julia> graph = SDDP.LinearGraph(3)
Root
 0
Nodes
 1
 2
 3
Arcs
 0 => 1 w.p. 1.0
 1 => 2 w.p. 1.0
 2 => 3 w.p. 1.0
```
"""
function LinearGraph(stages::Int)
    edges = Tuple{Pair{Int,Int},Float64}[]
    for t in 1:stages
        push!(edges, (t - 1 => t, 1.0))
    end
    return Graph(0, collect(1:stages), edges)
end

#Mathis
function InfiniteLinearGraph(stages::Int)
    edges = Tuple{Pair{Int,Int},Float64}[]
    for t in 1:stages
        push!(edges, (t - 1 => t, 1.0))
    end
    push!(edges, (stages => 1, 1.0))
    return Graph(0, collect(1:stages), edges)
end

"""
    MarkovianGraph(transition_matrices::Vector{Matrix{Float64}})

Construct a Markovian graph from the vector of transition matrices.

`transition_matrices[t][i, j]` gives the probability of transitioning from
Markov state `i` in stage `t - 1` to Markov state `j` in stage `t`.

The dimension of the first transition matrix should be `(1, N)`, and
`transition_matrics[1][1, i]` is the probability of transitioning from the root
node to the Markov state `i`.

## Examples

```jldoctest
julia> graph = SDDP.MarkovianGraph([ones(1, 1), [0.5 0.5], [0.8 0.2; 0.2 0.8]])
Root
 (0, 1)
Nodes
 (1, 1)
 (2, 1)
 (2, 2)
 (3, 1)
 (3, 2)
Arcs
 (0, 1) => (1, 1) w.p. 1.0
 (1, 1) => (2, 1) w.p. 0.5
 (1, 1) => (2, 2) w.p. 0.5
 (2, 1) => (3, 1) w.p. 0.8
 (2, 1) => (3, 2) w.p. 0.2
 (2, 2) => (3, 1) w.p. 0.2
 (2, 2) => (3, 2) w.p. 0.8
```
"""
function MarkovianGraph(transition_matrices::Vector{Matrix{Float64}})
    if size(transition_matrices[1], 1) != 1
        error(
            "Expected the first transition matrix to be of size (1, N). It " *
            "is of size $(size(transition_matrices[1])).",
        )
    end
    node_type = Tuple{Int,Int}
    root_node = (0, 1)
    nodes = node_type[]
    edges = Tuple{Pair{node_type,node_type},Float64}[]
    for (stage, transition) in enumerate(transition_matrices)
        if !all(transition .>= 0.0)
            error("Entries in the transition matrix must be non-negative.")
        end
        if !all(0.0 - 1e-8 .<= sum(transition; dims = 2) .<= 1.0 + 1e-8)
            error(
                "Rows in the transition matrix must sum to between 0.0 and 1.0.",
            )
        end
        if stage > 1
            if size(transition_matrices[stage-1], 2) != size(transition, 1)
                error("Transition matrix for stage $(stage) is the wrong size.")
            end
        end
        for markov_state in 1:size(transition, 2)
            push!(nodes, (stage, markov_state))
        end
        for markov_state in 1:size(transition, 2)
            for last_markov_state in 1:size(transition, 1)
                probability = transition[last_markov_state, markov_state]
                edge = (stage - 1, last_markov_state) => (stage, markov_state)
                push!(edges, (edge, probability))
            end
        end
    end
    return Graph(root_node, nodes, edges)
end

"""
    MarkovianGraph(;
        stages::Int,
        transition_matrix::Matrix{Float64},
        root_node_transition::Vector{Float64},
    )

Construct a Markovian graph object with `stages` number of stages and
time-independent Markov transition probabilities.

`transition_matrix` must be a square matrix, and the probability of
transitioning from Markov state `i` in stage `t` to Markov state `j` in stage
`t + 1` is given by `transition_matrix[i, j]`.

`root_node_transition[i]` is the probability of transitioning from the root node
to Markov state `i` in the first stage.

## Examples

```jldoctest
julia> graph = SDDP.MarkovianGraph(;
           stages = 3,
           transition_matrix = [0.8 0.2; 0.2 0.8],
           root_node_transition = [0.5, 0.5],
       )
Root
 (0, 1)
Nodes
 (1, 1)
 (1, 2)
 (2, 1)
 (2, 2)
 (3, 1)
 (3, 2)
Arcs
 (0, 1) => (1, 1) w.p. 0.5
 (0, 1) => (1, 2) w.p. 0.5
 (1, 1) => (2, 1) w.p. 0.8
 (1, 1) => (2, 2) w.p. 0.2
 (1, 2) => (2, 1) w.p. 0.2
 (1, 2) => (2, 2) w.p. 0.8
 (2, 1) => (3, 1) w.p. 0.8
 (2, 1) => (3, 2) w.p. 0.2
 (2, 2) => (3, 1) w.p. 0.2
 (2, 2) => (3, 2) w.p. 0.8
```
"""
function MarkovianGraph(;
    stages::Int = 1,
    transition_matrix::Matrix{Float64} = [1.0],
    root_node_transition::Vector{Float64} = [1.0],
)
    @assert size(transition_matrix, 1) == size(transition_matrix, 2)
    @assert length(root_node_transition) == size(transition_matrix, 1)
    return MarkovianGraph(
        vcat(
            [
                Base.reshape(
                    root_node_transition,
                    1,
                    length(root_node_transition),
                ),
            ],
            [transition_matrix for stage in 1:(stages-1)],
        ),
    )
end

"""
    UnicyclicGraph(discount_factor::Float64; num_nodes::Int = 1)

Construct a graph composed of `num_nodes` nodes that form a single cycle, with a
probability of `discount_factor` of continuing the cycle.

## Examples

```jldoctest
julia> graph = SDDP.UnicyclicGraph(0.9; num_nodes = 2)
Root
 0
Nodes
 1
 2
Arcs
 0 => 1 w.p. 1.0
 1 => 2 w.p. 1.0
 2 => 1 w.p. 0.9
```
"""
function UnicyclicGraph(discount_factor::Float64; num_nodes::Int = 1)
    @assert 0 < discount_factor < 1
    @assert num_nodes > 0
    graph = LinearGraph(num_nodes)
    add_edge(graph, num_nodes => 1, discount_factor)
    return graph
end

"""
    Noise(support, probability)

An atom of a discrete random variable at the point of support `support` and
associated probability `probability`.
"""
struct Noise{T}
    # The noise term.
    term::T
    # The probability of sampling the noise term.
    probability::Float64
end

struct State{T}
    # The incoming state variable.
    in::T
    # The outgoing state variable.
    out::T
end

mutable struct ObjectiveState{N}
    update::Function
    initial_value::NTuple{N,Float64}
    state::NTuple{N,Float64}
    lower_bound::NTuple{N,Float64}
    upper_bound::NTuple{N,Float64}
    μ::NTuple{N,JuMP.VariableRef}
end

# Storage for belief-related things.
struct BeliefState{T}
    partition_index::Int
    belief::Dict{T,Float64}
    μ::Dict{T,JuMP.VariableRef}
    updater::Function
end

mutable struct TwoStage 
    model::JuMP.Model
    non_anticipative_variables::Dict{Symbol, VariableRef}
    states::Dict{Symbol, Vector{VariableRef}}
    bellman_variables::Vector{VariableRef}
    lower_bounds::Dict{Symbol, Float64}
    upper_bounds::Dict{Symbol, Float64}
end

mutable struct Cut2
    iteration::Int64
    time::Float64
    intercept::Float64
    coefficients::Dict{Symbol,Float64}
    shift::Vector{Float64}
    constraint_V::JuMP.ConstraintRef
    constraint_subproblem::JuMP.ConstraintRef
    state::Dict{Symbol,Float64}
end

mutable struct Cut3
    intercept::Float64
    coefficients::Dict{Symbol,Float64}
end

mutable struct Value_Function
    model::JuMP.Model
    cut_V::Vector{Cut2}
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    model_TV::JuMP.Model
    theta_TV::JuMP.VariableRef
    states_TV::Dict{Symbol,JuMP.VariableRef}
    heuristic_state::Dict{Symbol, Float64}
    cut_TV::Vector{Cut3}
end

mutable struct Node{T}
    # The index of the node in the policy graph.
    index::T
    # The JuMP subproblem.
    subproblem::JuMP.Model
    # Mathis.
    two_stage::TwoStage
    value_function::Value_Function
    states_two_stage::Dict{Symbol,VariableRef}
    constraints::Vector{ConstraintRef}
    # A vector of the child nodes.
    children::Vector{Noise{T}}
    # A vector of the discrete stagewise-independent noise terms.
    noise_terms::Vector{Noise}
    # A function parameterize(model::JuMP.Model, noise) that modifies the JuMP
    # model based on the observation of the noise.
    parameterize::Function  # TODO(odow): make this a concrete type?
    # A list of the state variables in the model.
    states::Dict{Symbol,State{JuMP.VariableRef}}
    # Stage objective
    stage_objective::Any  # TODO(odow): make this a concrete type?
    stage_objective_set::Bool
    # Bellman function
    bellman_function::Any  # TODO(odow): make this a concrete type?
    # For dynamic interpolation of objective states.
    objective_state::Union{Nothing,ObjectiveState}
    # For dynamic interpolation of belief states.
    belief_state::Union{Nothing,BeliefState{T}}
    # An over-loadable hook for the JuMP.optimize! function.
    pre_optimize_hook::Union{Nothing,Function}
    post_optimize_hook::Union{Nothing,Function}
    # Approach for handling discrete variables.
    has_integrality::Bool
    # The user's optimizer. We use this in asynchronous mode.
    optimizer::Any
    # An extension dictionary. This is a useful place for packages that extend
    # SDDP.jl to stash things.
    ext::Dict{Symbol,Any}
    # Lock for threading
    lock::ReentrantLock
    # (lower, upper, is_integer)
    incoming_state_bounds::Dict{Symbol,Tuple{Float64,Float64,Bool}}
    #Mathis
    discount_factor::Float64
end

function Base.show(io::IO, node::Node)
    println(io, "Node $(node.index)")
    println(io, "  # State variables : ", length(node.states))
    println(io, "  # Children        : ", length(node.children))
    println(io, "  # Noise terms     : ", length(node.noise_terms))
    return
end

function pre_optimize_hook(f::Function, node::Node)
    node.pre_optimize_hook = f
    return
end

function post_optimize_hook(f::Function, node::Node)
    node.post_optimize_hook = f
    return
end

struct Log
    iteration::Int
    bound::Float64
    simulation_value::Float64
    time::Float64
    pid::Int
    total_solves::Int
    duality_key::String
    serious_numerical_issue::Bool
    iter_cuts
end

struct TrainingResults
    status::Symbol
    log::Vector{Log}
end

mutable struct PolicyGraph{T}
    # Must be MOI.MIN_SENSE or MOI.MAX_SENSE
    objective_sense::MOI.OptimizationSense
    # Index of the root node.
    root_node::T
    # Children of the root node. child => probability.
    root_children::Vector{Noise{T}}
    # Starting value of the state variables.
    initial_root_state::Dict{Symbol,Float64}
    # All nodes in the graph.
    nodes::Dict{T,Node{T}}
    # Belief partition.
    belief_partition::Vector{Set{T}}
    # Storage for the most recent training results.
    most_recent_training_results::Union{Nothing,TrainingResults}
    # An extension dictionary. This is a useful place for packages that extend
    # SDDP.jl to stash things.
    ext::Dict{Symbol,Any}
    timer_output::TimerOutputs.TimerOutput
    lock::ReentrantLock

    function PolicyGraph(sense::Symbol, root_node::T) where {T}
        if sense != :Min && sense != :Max
            error(
                "The optimization sense must be `:Min` or `:Max`. It is $(sense).",
            )
        end
        optimization_sense = sense == :Min ? MOI.MIN_SENSE : MOI.MAX_SENSE
        return new{T}(
            optimization_sense,
            root_node,
            Noise{T}[],
            Dict{Symbol,Float64}(),
            Dict{T,Node{T}}(),
            Set{T}[],
            nothing,
            Dict{Symbol,Any}(),
            TimerOutputs.TimerOutput(),
            ReentrantLock(),
        )
    end
end

function Base.show(io::IO, graph::PolicyGraph)
    N = length(graph.nodes)
    println(io, "A policy graph with $(N) nodes.")
    nodes = sort_nodes(collect(keys(graph.nodes)))
    if N < 10
        println(io, " Node indices: ", join(nodes, ", "))
    else
        println(io, " Node indices: ", nodes[1], ", ..., ", nodes[end])
    end
    return
end

# So we can query nodes in the graph as graph[node].
function Base.getindex(graph::PolicyGraph{T}, index::T) where {T}
    return graph.nodes[index]
end

# Work around different JuMP modes (Automatic / Manual / Direct).
function construct_subproblem(optimizer_factory, direct_mode::Bool)
    if direct_mode
        model = JuMP.direct_model(MOI.instantiate(optimizer_factory))
        set_silent(model)
        return model
    end
    return JuMP.Model()
end

# Work around different JuMP modes (Automatic / Manual / Direct).
function construct_subproblem(::Nothing, direct_mode::Bool)
    if direct_mode
        error(
            "You must specify an optimizer in the form:\n" *
            "    with_optimizer(Module.Opimizer, args...) if " *
            "direct_mode=true.",
        )
    end
    return JuMP.Model()
end

"""
    LinearPolicyGraph(builder::Function; stages::Int, kwargs...)

Create a linear policy graph with `stages` number of stages.

## Keyword arguments

 - `stages`: the number of stages in the graph

 - `kwargs`: other keyword arguments are passed to [`SDDP.PolicyGraph`](@ref).

## Examples

```jldoctest
julia> SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0.0) do sp, t
    # ... build model ...
end
A policy graph with 2 nodes.
Node indices: 1, 2
```
is equivalent to
```jldoctest
julia> graph = SDDP.LinearGraph(2);

julia> SDDP.PolicyGraph(graph; lower_bound = 0.0) do sp, t
    # ... build model ...
end
A policy graph with 2 nodes.
Node indices: 1, 2
```
"""
function LinearPolicyGraph(builder::Function; stages::Int, kwargs...)
    if stages < 1
        error("You must create a LinearPolicyGraph with `stages >= 1`.")
    end
    return PolicyGraph(builder, LinearGraph(stages); kwargs...)
end

"""
    MarkovianPolicyGraph(
        builder::Function;
        transition_matrices::Vector{Array{Float64,2}},
        kwargs...
    )

Create a Markovian policy graph based on the transition matrices given in
`transition_matrices`.

## Keyword arguments

 - `transition_matrices[t][i, j]` gives the probability of transitioning from
   Markov state `i` in stage `t - 1` to Markov state `j` in stage `t`.
   The dimension of the first transition matrix should be `(1, N)`, and
   `transition_matrics[1][1, i]` is the probability of transitioning from the
   root node to the Markov state `i`.

 - `kwargs`: other keyword arguments are passed to [`SDDP.PolicyGraph`](@ref).

## See also

See [`SDDP.MarkovianGraph`](@ref) for other ways of specifying a Markovian
policy graph.

See [`SDDP.PolicyGraph`](@ref) for the other keyword arguments.

## Examples

```jldoctest
julia> SDDP.MarkovianPolicyGraph(;
           transition_matrices = [ones(1, 1), [0.5 0.5], [0.8 0.2; 0.2 0.8]],
           lower_bound = 0.0,
       ) do sp, node
           # ... build model ...
       end
A policy graph with 5 nodes.
 Node indices: (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)
```
is equivalent to
```jldoctest
julia> graph = SDDP.MarkovianGraph([ones(1, 1), [0.5 0.5], [0.8 0.2; 0.2 0.8]]);

julia> SDDP.PolicyGraph(graph; lower_bound = 0.0) do sp, t
    # ... build model ...
end
A policy graph with 5 nodes.
 Node indices: (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)
```
"""
function MarkovianPolicyGraph(
    builder::Function;
    transition_matrices::Vector{Array{Float64,2}},
    kwargs...,
)
    return PolicyGraph(builder, MarkovianGraph(transition_matrices); kwargs...)
end

"""
    PolicyGraph(
        builder::Function,
        graph::Graph{T};
        sense::Symbol = :Min,
        lower_bound = -Inf,
        upper_bound = Inf,
        optimizer = nothing,
    ) where {T}

Construct a policy graph based on the graph structure of `graph`. (See
[`SDDP.Graph`](@ref) for details.)

## Keyword arguments

 - `sense`: whether we are minimizing (`:Min`) or maximizing (`:Max`).

 - `lower_bound`: if mimimizing, a valid lower bound for the cost to go in all
   subproblems.

 - `upper_bound`: if maximizing, a valid upper bound for the value to go in all
   subproblems.

 - `optimizer`: the optimizer to use for each of the subproblems

## Examples

```julia
function builder(subproblem::JuMP.Model, index)
    # ... subproblem definition ...
end

model = PolicyGraph(
    builder,
    graph;
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
)
```

Or, using the Julia `do ... end` syntax:

```julia
model = PolicyGraph(
    graph;
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
) do subproblem, index
    # ... subproblem definitions ...
end
```
"""
#Mathis
function PolicyGraph(
    builder::Function,
    graph::Graph{T};
    sense::Symbol = :Min,
    lower_bound = -Inf,
    upper_bound = Inf,
    optimizer = nothing,
    # These arguments are deprecated
    bellman_function = nothing,
    direct_mode::Bool = false,
    discount_factor::Float64 = 1.0,
) where {T}
    # Spend a one-off cost validating the graph.
    _validate_graph(graph)
    # Construct a basic policy graph. We will add to it in the remainder of this
    # function.
    policy_graph = PolicyGraph(sense, graph.root_node)
    # Create a Bellman function if one is not given.
    if bellman_function === nothing
        if sense == :Min && lower_bound === -Inf
            error(
                "You must specify a finite lower bound on the objective value" *
                " using the `lower_bound = value` keyword argument.",
            )
        elseif sense == :Max && upper_bound === Inf
            error(
                "You must specify a finite upper bound on the objective value" *
                " using the `upper_bound = value` keyword argument.",
            )
        else
            bellman_function = BellmanFunction(;
                lower_bound = lower_bound,
                upper_bound = upper_bound,
            )
        end
    end
    # Initialize nodes.
    for (node_index, children) in graph.nodes
        if node_index == graph.root_node
            continue
        end
        subproblem = construct_subproblem(optimizer, direct_mode)
        twostage=TwoStage(
            construct_subproblem(optimizer, direct_mode), 
            Dict{Symbol,VariableRef}(), 
            Dict{Symbol, Vector{VariableRef}}(),
            VariableRef[],
            Dict{Symbol, Float64}(),
            Dict{Symbol, Float64}()
        )

        valuefunction = initialize_value_function(sense, optimizer)
        node = Node(
            node_index,
            subproblem,
            twostage,
            valuefunction,
            Dict{Symbol, VariableRef}(),
            JuMP.ConstraintRef[],
            Noise{T}[],
            Noise[],
            (ω) -> nothing,
            Dict{Symbol,State{JuMP.VariableRef}}(),
            0.0,
            false,
            # Delay initializing the bellman function until later so that it can
            # use information about the children and number of
            # stagewise-independent noise realizations.
            nothing,
            # Likewise for the objective states.
            nothing,
            # And for belief states.
            nothing,
            # The optimize hook defaults to nothing.
            nothing,
            nothing,
            false,
            direct_mode ? nothing : optimizer,
            # The extension dictionary.
            Dict{Symbol,Any}(),
            ReentrantLock(),
            Dict{Symbol,Tuple{Float64,Float64,Bool}}(),
            discount_factor,
        )
        subproblem.ext[:sddp_policy_graph] = policy_graph
        policy_graph.nodes[node_index] = subproblem.ext[:sddp_node] = node
        JuMP.set_objective_sense(subproblem, policy_graph.objective_sense)
        builder(subproblem, node_index, discount_factor)
        # Add a dummy noise here so that all nodes have at least one noise term.
        if length(node.noise_terms) == 0
            push!(node.noise_terms, Noise(nothing, 1.0))
        end
        ctypes = JuMP.list_of_constraint_types(subproblem)
        node.has_integrality =
            (JuMP.VariableRef, MOI.Integer) in ctypes ||
            (JuMP.VariableRef, MOI.ZeroOne) in ctypes
        
    end
    # Loop back through and add the arcs/children.
    for (node_index, children) in graph.nodes
        if node_index == graph.root_node
            continue
        end
        node = policy_graph.nodes[node_index]
        for (child, probability) in children
            push!(node.children, Noise(child, probability))
        end
        # Intialize the bellman function. (See note in creation of Node above.)
        node.bellman_function =
            initialize_bellman_function(bellman_function, policy_graph, node)
    end
    # Add root nodes
    for (child, probability) in graph.nodes[graph.root_node]
        push!(policy_graph.root_children, Noise(child, probability))
        # We check the feasibility of the initial point here. It is a really
        # tricky feasibility bug to diagnose otherwise. See #387 for details.
        for (k, v) in policy_graph.initial_root_state
            x_out = policy_graph[child].states[k].out
            if JuMP.has_lower_bound(x_out) && JuMP.lower_bound(x_out) > v
                error("Initial point $(v) violates lower bound on state $k")
            elseif JuMP.has_upper_bound(x_out) && JuMP.upper_bound(x_out) < v
                error("Initial point $(v) violates upper bound on state $k")
            end
        end
    end
    # Initialize belief states.
    if length(graph.belief_partition) > 0
        initialize_belief_states(policy_graph, graph)
    end
    domain = _get_incoming_domain(policy_graph)
    for (node_name, node) in policy_graph.nodes
        for (k, v) in domain[node_name]
            node.incoming_state_bounds[k] = something(v, (-Inf, Inf, false))
        end
    end
    _initialize_solver(policy_graph; throw_error = false)

    for (node_index, children) in graph.nodes
        if node_index == graph.root_node
            continue
        end
        node = policy_graph.nodes[node_index]
        initialize_two_stage(policy_graph, node, optimizer)
        add_state_variables_to_value_function(node)
    end
    return policy_graph
end

function _get_incoming_domain(model::PolicyGraph{T}) where {T}
    function _bounds(x)
        l, u = -Inf, Inf
        if has_lower_bound(x)
            l = lower_bound(x)
        end
        if has_upper_bound(x)
            u = upper_bound(x)
        end
        if is_fixed(x)
            l = u = fix_value(x)
        end
        is_int = is_integer(x)
        if is_binary(x)
            l, u = max(something(l, 0.0), 0.0), min(something(u, 1.0), 1.0)
            is_int = true
        end
        return l, u, is_int
    end
    outgoing_bounds = Dict{Tuple{T,Symbol},Any}(
        (k, state_name) => nothing for (k, node) in model.nodes for
        (state_name, _) in node.states
    )
    for (k, node) in model.nodes
        for noise in node.noise_terms
            parameterize(node, noise.term)
            for (state_name, state) in node.states
                domain = outgoing_bounds[(k, state_name)]
                l_new, u_new, is_int_new = _bounds(state.out)
                outgoing_bounds[(k, state_name)] = if domain === nothing
                    (l_new, u_new, is_int_new)
                else
                    l, u, is_int = domain::Tuple{Float64,Float64,Bool}
                    (min(l, l_new), max(u, u_new), is_int & is_int_new)
                end
            end
        end
    end
    incoming_bounds = Dict{T,Dict{Symbol,Any}}()
    for (k, node) in model.nodes
        incoming_bounds[k] = Dict{Symbol,Any}(
            state_name => nothing for (state_name, _) in node.states
        )
    end
    for (parent_name, parent) in model.nodes
        for (state_name, state) in parent.states
            domain_new = outgoing_bounds[(parent_name, state_name)]
            l_new, u_new, is_int_new = domain_new::Tuple{Float64,Float64,Bool}
            for child in parent.children
                domain = incoming_bounds[child.term][state_name]
                incoming_bounds[child.term][state_name] = if domain === nothing
                    (l_new, u_new, is_int_new)
                else
                    l, u, is_int = domain::Tuple{Float64,Float64,Bool}
                    (min(l, l_new), max(u, u_new), is_int & is_int_new)
                end
            end
        end
    end
    # The incoming state from the root node can be anything
    for (state_name, value) in model.initial_root_state
        for child in model.root_children
            incoming_bounds[child.term][state_name] = (-Inf, Inf, false)
        end
    end
    return incoming_bounds
end

# Internal function: set up ::BeliefState for each node.
function initialize_belief_states(
    policy_graph::PolicyGraph{T},
    graph::Graph{T},
) where {T}
    # Pre-compute the function `belief_updater`. See `construct_belief_update`
    # for details.
    belief_updater =
        construct_belief_update(policy_graph, Set.(graph.belief_partition))
    # Initialize a belief dictionary (containing one element for each node in
    # the graph).
    belief = Dict{T,Float64}(keys(graph.nodes) .=> 0.0)
    delete!(belief, graph.root_node)
    # Now for each element in the partition...
    for (partition_index, partition) in enumerate(graph.belief_partition)
        # Store the partition in the `policy_graph` object.
        push!(policy_graph.belief_partition, Set(partition))
        # Then for each node in the partition.
        for node_index in partition
            # Get the `::Node` object.
            node = policy_graph[node_index]
            # Add the dual variable μ for the cut:
            # <b, μ> + θ ≥ α + <β, x>
            # We need one variable for each non-zero belief state.
            μ = Dict{T,JuMP.VariableRef}()
            for (node_name, L) in
                zip(partition, graph.belief_lipschitz[partition_index])
                μ[node_name] = @variable(
                    node.subproblem,
                    lower_bound = -L,
                    upper_bound = L
                )
            end
            add_initial_bounds(node, μ)
            # Attach the belief state as an extension.
            node.belief_state =
                BeliefState{T}(partition_index, copy(belief), μ, belief_updater)

            node.bellman_function.global_theta.belief_states = μ
            for theta in node.bellman_function.local_thetas
                theta.belief_states = μ
            end
        end
    end
    return
end

# Internal function: When created, θ has bounds of [-M, M], but, since we are
# adding these μ terms, we really want to bound <b, μ> + θ ∈ [-M, M]. Keeping in
# mind that ∑b = 1, we really only need to add these constraints at the corners
# of the box where one element in b is 1, and all the rest are 0.
function add_initial_bounds(node, μ::Dict)
    θ = bellman_term(node.bellman_function)
    lower_bound = JuMP.has_lower_bound(θ) ? JuMP.lower_bound(θ) : -Inf
    upper_bound = JuMP.has_upper_bound(θ) ? JuMP.upper_bound(θ) : Inf
    for (_, variable) in μ
        if lower_bound > -Inf
            @constraint(node.subproblem, variable + θ >= lower_bound)
        end
        if upper_bound < Inf
            @constraint(node.subproblem, variable + θ <= upper_bound)
        end
    end
    return
end

# Internal function: helper to get the node given a subproblem.
function get_node(subproblem::JuMP.Model)
    return subproblem.ext[:sddp_node]::Node
end

# Internal function: helper to get the policy graph given a subproblem.
function get_policy_graph(subproblem::JuMP.Model)
    return subproblem.ext[:sddp_policy_graph]::PolicyGraph
end

"""
    parameterize(
        modify::Function,
        subproblem::JuMP.Model,
        realizations::Vector{T},
        probability::Vector{Float64} = fill(1.0 / length(realizations))
    ) where {T}

Add a parameterization function `modify` to `subproblem`. The `modify` function
takes one argument and modifies `subproblem` based on the realization of the
noise sampled from `realizations` with corresponding probabilities
`probability`.

In order to conduct an out-of-sample simulation, `modify` should accept
arguments that are not in realizations (but still of type T).

## Examples

```julia
SDDP.parameterize(subproblem, [1, 2, 3], [0.4, 0.3, 0.3]) do ω
    JuMP.set_upper_bound(x, ω)
end
```
"""
function parameterize(
    modify::Function,
    subproblem::JuMP.Model,
    realizations::AbstractVector{T},
    probability::AbstractVector{Float64} = fill(
        1.0 / length(realizations),
        length(realizations),
    ),
) where {T}
    node = get_node(subproblem)
    if length(node.noise_terms) != 0
        error("Duplicate calls to SDDP.parameterize detected.")
    end
    for (realization, prob) in zip(realizations, probability)
        push!(node.noise_terms, Noise(realization, prob))
    end
    node.parameterize = modify
    return
end

"""
    set_stage_objective(
        subproblem::JuMP.Model,
        stage_objective::Union{Real,JuMP.AbstractJuMPScalar},
    )

Set the stage-objective of `subproblem` to `stage_objective`.

## Examples

```julia
SDDP.set_stage_objective(subproblem, 2x + 1)
```
"""
function set_stage_objective(
    subproblem::JuMP.Model,
    stage_objective::Union{Real,JuMP.AbstractJuMPScalar},
)
    node = get_node(subproblem)
    node.stage_objective = stage_objective
    node.stage_objective_set = false
    return
end

function set_stage_objective(::JuMP.Model, f)
    return error(
        "Unable to set the stage-objective of type $(typeof(f)). It must be " *
        "a scalar function.",
    )
end

"""
    @stageobjective(subproblem, expr)

Set the stage-objective of `subproblem` to `expr`.

## Examples

```julia
@stageobjective(subproblem, 2x + y)
```
"""
macro stageobjective(subproblem, expr)
    code = MutableArithmetics.rewrite_and_return(expr)
    return quote
        SDDP.set_stage_objective($(esc(subproblem)), $code)
    end
end

"""
    add_objective_state(update::Function, subproblem::JuMP.Model; kwargs...)

Add an objective state variable to `subproblem`.

Required `kwargs` are:

 - `initial_value`: The initial value of the objective state variable at the
    root node.
 - `lipschitz`: The lipschitz constant of the objective state variable.

Setting a tight value for the lipschitz constant can significantly improve the
speed of convergence.

Optional `kwargs` are:

 - `lower_bound`: A valid lower bound for the objective state variable. Can be
    `-Inf`.
 - `upper_bound`: A valid upper bound for the objective state variable. Can be
    `+Inf`.

Setting tight values for these optional variables can significantly improve the
speed of convergence.

If the objective state is `N`-dimensional, each keyword argument must be an
`NTuple{N,Float64}`. For example, `initial_value = (0.0, 1.0)`.
"""
function add_objective_state(
    update::Function,
    subproblem::JuMP.Model;
    initial_value::Union{Real,Tuple},
    lipschitz::Union{Real,Tuple},
    lower_bound::Union{Real,Tuple} = -Inf,
    upper_bound::Union{Real,Tuple} = Inf,
)
    tup_initial_value = _to_tuple(initial_value)
    N = length(tup_initial_value)
    return add_objective_state(
        update,
        subproblem,
        tup_initial_value,
        _to_tuple(lower_bound, N),
        _to_tuple(upper_bound, N),
        _to_tuple(lipschitz, N),
    )
end

_to_tuple(x::Real, N::Int = 1) = ntuple(i -> Float64(x), N)

function _to_tuple(x::Tuple, N::Int = length(x))
    if length(x) != N
        error(
            "Invalid dimension in the input to `add_objective_state`. Got: ",
            "`$x`, but expected it to have length `$N`.",
        )
    end
    return Float64.(x)
end

# Internal function: add_objective_state with positional NTuple arguments.
function add_objective_state(
    update::Function,
    subproblem::JuMP.Model,
    initial_value::NTuple{N,Float64},
    lower_bound::NTuple{N,Float64},
    upper_bound::NTuple{N,Float64},
    lipschitz::NTuple{N,Float64},
) where {N}
    node = get_node(subproblem)
    if node.objective_state !== nothing
        error("add_objective_state can only be called once.")
    end
    μ = @variable(
        subproblem,
        [i = 1:N],
        lower_bound = -lipschitz[i],
        upper_bound = lipschitz[i]
    )
    node.objective_state = ObjectiveState(
        update,
        initial_value,
        initial_value,
        lower_bound,
        upper_bound,
        tuple(μ...),
    )
    return
end

"""
    objective_state(subproblem::JuMP.Model)

Return the current objective state of the problem.

Can only be called from [`SDDP.parameterize`](@ref).
"""
function objective_state(subproblem::JuMP.Model)
    objective_state = get_node(subproblem).objective_state
    if objective_state === nothing
        error("No objective state defined.")
    elseif length(objective_state.state) == 1
        return objective_state.state[1]
    else
        return objective_state.state
    end
end

# Internal function: calculate <y, μ>.
function get_objective_state_component(node::Node)
    objective_state_component = JuMP.AffExpr(0.0)
    objective_state = node.objective_state
    if objective_state !== nothing
        for (y, μ) in zip(objective_state.state, objective_state.μ)
            JuMP.add_to_expression!(objective_state_component, y, μ)
        end
    end
    return objective_state_component
end

function build_Φ(graph::PolicyGraph{T}) where {T}
    Φ = Dict{Tuple{T,T},Float64}()
    for (node_index_1, node_1) in graph.nodes
        for child in node_1.children
            Φ[(node_index_1, child.term)] = child.probability
        end
    end
    for child in graph.root_children
        Φ[(graph.root_node, child.term)] = child.probability
    end
    return Φ
end

"""
    construct_belief_update(graph::PolicyGraph{T}, partition::Vector{Set{T}})

Returns a function that calculates the belief update. That function has the
following signature and returns the outgoing belief:

    belief_update(
        incoming_belief::Dict{T, Float64},
        observed_partition::Int,
        observed_noise
    )::Dict{T,Float64}

We use Bayes theorem: P(X′ | Y) = P(Y | X′) × P(X′) / P(Y), where P(Xᵢ′ | Y) is
the probability of being in node i given the observation of ω. In addition

 - P(Xⱼ′) = ∑ᵢ P(Xᵢ) × Φᵢⱼ
 - P(Y|Xᵢ′) = P(ω ∈ Ωᵢ)
 - P(Y) = ∑ᵢ P(Xᵢ′) × P(ω ∈ Ωᵢ)
"""
function construct_belief_update(
    graph::SDDP.PolicyGraph{T},
    partition::Vector{Set{T}},
) where {T}
    # TODO: check that partition is proper.
    Φ = build_Φ(graph)  # Dict{Tuple{T, T}, Float64}
    Ω = Dict{T,Dict{Any,Float64}}()
    for (index, node) in graph.nodes
        Ω[index] = Dict{Any,Float64}()
        for noise in node.noise_terms
            Ω[index][noise.term] = noise.probability
        end
    end
    function belief_updater(
        outgoing_belief::Dict{T,Float64},
        incoming_belief::Dict{T,Float64},
        observed_partition::Int,
        observed_noise,
    )::Dict{T,Float64}
        # P(Y) = ∑ᵢ Xᵢ × ∑ⱼ P(i->j) × P(ω ∈ Ωⱼ)
        PY = 0.0
        for (node_i, belief) in incoming_belief
            probability = 0.0
            for (node_j, Ωj) in Ω
                p_ij = get(Φ, (node_i, node_j), 0.0)
                p_ω = get(Ωj, observed_noise, 0.0)
                probability += p_ij * p_ω
            end
            PY += belief * probability
        end
        if PY ≈ 0.0
            error(
                "Unable to update belief in partition ",
                observed_partition,
                " after observing ",
                observed_noise,
                ".The incoming belief ",
                "is:\n  ",
                incoming_belief,
            )
        end
        # Now update each belief.
        for (node_i, belief) in incoming_belief
            PX = sum(
                belief * get(Φ, (node_j, node_i), 0.0) for
                (node_j, belief) in incoming_belief
            )
            PY_X = 0.0
            if node_i in partition[observed_partition]
                PY_X += get(Ω[node_i], observed_noise, 0.0)
            end
            outgoing_belief[node_i] = PY_X * PX / PY
        end
        if length(outgoing_belief) == 2
            for (node_i, belief) in incoming_belief
                if belief < 1e-6
                    incoming_belief[node_i] = 0.0
                elseif belief > 1 - 1e-6
                    incoming_belief[node_i] = 1.0
                end
            end
        end
        return outgoing_belief
    end
    return belief_updater
end

# Internal function: calculate <b, μ>.
function get_belief_state_component(node::Node)
    belief_component = JuMP.AffExpr(0.0)
    if node.belief_state !== nothing
        belief = node.belief_state
        for (key, μ) in belief.μ
            JuMP.add_to_expression!(belief_component, belief.belief[key], μ)
        end
    end
    return belief_component
end
