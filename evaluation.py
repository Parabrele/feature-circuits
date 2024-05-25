"""
This file contains evaluations for compuational graphs, including :

faithfulness :
    - override forward fct to keep only edges defined by some graph
    - measure recovered metric for some graph

sparsity :
    - get edges / nodes as a function of nodes (n^2, nln(n), O(n), etc.) (1)
    - get minimum edges (and edges / nodes) (per layer / total / etc.) requiered to recover 1 +- eps of some metric (e.g. accuracy, CE, etc.)
    - plot recovered metric as a function of (1) (mostly O(n), but also include n^2, nsqrt(n), nln(n), etc.) in the plot as vertical lines as an indication
    - do as marks (for completeness) and patch nodes instead of edges, and plot how much nodes are required to recover some metric with the complete graph between those nodes
    
modularity :
    - cluster the nodes st degree(intra) / degree(inter) is maximized, or degree(inter) is minimized, or etc.
    - Use the Louvain algorithm (or Leiden, which is an improvement of Louvain) to cluster the nodes
    - Cluster either on weighted graph, log weighted graph, or binary graph
    - measure the separation and the quality of the communities
    - do communities of communities and iteratively to estimate the degree of complexity of the model
        (this would require communities to be quite small though, and assume that community graph is still sparse)
        EDIT : this is in fact just more or less iterations of the algo.
        
    - maybe cluster nodes using GNN ?
        
is_subgraph :
    given a circuit, e.g. from the IOI task, check if it is a subgraph of the full graph of computation

is_module :
    given a circuit, check that it is a module/community

module_score :
    given a circuit, compute edge_in / edge_out or something to measure the quality of this circuit as a module in the full graph

"""

import torch
import networkx as nx
import networkit as nk

from utils import SparseAct

##########
# faithfulness & model evaluation
##########

@torch.no_grad()
def get_mask(circuit, threshold):
    """
    circuit :
    tuple (nodes, edges)
        nodes : dict of SparseAct with keys layer names
        edges : dict of dict of sparse_coo tensors, [name_upstream][name_downstream] -> edge weights

    returns a dict similar to edges but binary sparse coo tensors : only weights above threshold are kept
    """

    nodes, edges = circuit
    mask = {}

    for upstream in edges:
        mask[upstream] = {}
        for downstream in edges[upstream]:
            weights = edges[upstream][downstream].coalesce()
            if threshold == -1:
                mask[upstream][downstream] = torch.sparse_coo_tensor([], [], weights.size())
            else:
                value_mask = weights.values() > threshold
                mask[upstream][downstream] = torch.sparse_coo_tensor(weights.indices()[value_mask], weights.values()[value_mask], weights.size())
        
    return mask

@torch.no_grad()
def run_graph(
        model,
        submodules,
        sae_dict,
        name_dict,
        clean,
        patch,
        circuit,
        metric_fn,
        metric_fn_kwargs,
        ablation_fn,
        complement=False,
    ):
    """
    model : nnsight model
    submodules : list of model submodules
        Should be ordered by appearance in the sequencial model
    sae_dict : dict
        dict [submodule] -> SAE
    name_dict : dict
        dict [submodule] -> str
    clean : str, list of str or tensor (batch, seq_len)
        the input to the model
    patch : None, str, list of str or tensor (batch, seq_len)
        the counterfactual input to the model to ablate edges
    circuit : edges
        the graph to use. Accept any type of sparse_coo tensor, but preferably bool.
        Any index in these tensors will be used as an edge.
    metric_fn : callable
        the function to evaluate the model.
        It can be CE, accuracy, logit for target token, etc.
    metric_fn_kwargs : dict
        the kwargs to pass to metric_fn. E.g. target token.
    ablation_fn : callable
        the function used to get the patched states.
        E.g. : mean ablation means across batch and sequence length or only across one of them.
               zero ablation : patch states are just zeros
               id ablation : patch states are computed from the patch input and left unchanged

    returns the metric on the model with the graph
    """

    sub_dict = {v : k for k, v in name_dict.items()}

    if patch is None:
        patch = clean
    patch_states = {}

    with model.trace(patch):
        for submodule in submodules:
            downstream_dict = sae_dict[submodule]
            x = submodule.output
            if type(x) == tuple:
                x = x[0]
            x_hat, f = downstream_dict(x, output_features=True)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    
    patch_states = {k : ablation_fn(v.value) for k, v in patch_states.items()}

    # For each downstream module, get it's potentially alive features by reachability from previously alive ones
    # Then, for each of these features, get it's masked input, and compute a forward pass to get this particular feature.
    # This gives the new state for this downstream output.

    with model.trace(clean):
        hidden_states = {}
        for downstream in submodules:
            # get downstream dict, output, ...
            downstream_dict = sae_dict[downstream]
            down_name = name_dict[downstream]

            input_shape = downstream.input.size()
            x = downstream.output
            is_tuple = type(x) == tuple
            if is_tuple:
                x = x[0]
            x_hat, f = downstream_dict(x, output_features=True)
            res = x - x_hat

            # if downstream is embed, there is no upstream and the result stays unchanged
            if down_name == 'embed':
                hidden_states[downstream] = SparseAct(act=f, res=res)
                continue

            # otherwise, we have to do the computation as the graph describes it, and each downstream
            # feature, including the res, is computed from a different set of upstream features

            # TODO : this is wrong, I have to compute all features, even those who have no predecessors
            # alive as it may be important that they are all asleep for this particular function
            # However, it is computationally unreasonable to compute tens of thousands of forward passes,
            # so until a better solution is found, I will do this.
            # TODO : check that the result between this and all features being potentially alive are the same
            #        or close enough on a few examples to validate this choice.
            # TODO : of course, mention it in the paper.
            # TODO?: for features whose masks are all zeros, skip and keep only the patch state
            # TODO?: compress the model and its graph by deleting SAE features that are neither reachable nor co reachable
            # TODO?: graph weights should be absolute values
            # TODO : mask[:-1] @ (cat(..., 1).aggregate(dim=(0, 1)))
            potentially_alive = torch.sparse_coo_tensor([], [], f.size())
            for up_name in circuit:
                if down_name in circuit[up_name]:
                    upstream = sub_dict[up_name]
                    mask = circuit[up_name][down_name] # shape (f_down + 1, f_up + 1)
                    potentially_alive += (
                        mask[:-1, :-1] @ hidden_states[upstream].act.to_sparse()
                    ).to(potentially_alive.dtype)

            for f_ in potentially_alive.indices()[0]:
                edge_ablated_input = torch.zeros(input_shape)
                for up_name in circuit:
                    if down_name not in circuit[up_name]:
                        continue
                    upstream = sub_dict[up_name]
                    upstream_dict = sae_dict[upstream]
                    
                    mask = circuit[up_name][down_name][f_] # shape (f_down + 1, f_up + 1)

                    edge_ablated_upstream = SparseAct(
                        act = hidden_states[upstream].act[:, :, mask[:-1]] + patch_states[upstream].act[:, :, ~mask[:-1]],
                        res = hidden_states[upstream].res if mask[-1] else patch_states[upstream].res
                    )

                    edge_ablated_input += upstream_dict.decode(edge_ablated_upstream.act) + edge_ablated_upstream.res

                edge_ablated_out = downstream.forward(edge_ablated_input)
                if f_ < f.size(-1):
                    # TODO : add option in sae forward to get only one feature to fasten this
                    #        only after testing that it works like this first and then checking that it
                    #        is faster and doesn't break anything
                    f[..., f_] = downstream_dict.encode(edge_ablated_out)[..., f_]
                else:
                    res = edge_ablated_out - downstream_dict(edge_ablated_out)

            hidden_states[downstream] = SparseAct(act=f, res=res)

            if is_tuple:
                downstream.output[0][:] = downstream_dict.decode(f) + res
            else:
                downstream.output = downstream_dict.decode(f) + res

        metric = metric_fn(model, **metric_fn_kwargs).save()

    return metric.value

@torch.no_grad()
def faithfulness(
        model,
        submodules,
        sae_dict,
        name_dict,
        clean,
        circuit,
        thresholds,
        metric_fn,
        metric_fn_kwargs={},
        patch=None,
        complement=False,
        ablation_fn=None,
        default_ablation='mean',
    ):
    """
    model : nnsight model
    submodules : list of model submodules
    sae_dict : dict
        dict [submodule] -> SAE
    name_dict : dict
        dict [submodule] -> str
    clean : str, list of str or tensor (batch, seq_len)
        the input to the model
    patch : None, str, list of str or tensor (batch, seq_len)
        the counterfactual input to the model to ablate edges.
        If None, ablation_fn is default to mean
        Else, ablation_fn is default to identity
    circuit : tuple (nodes, edges)
    thresholds : float or list of float
        the thresholds to discard edges based on their weights
    metric_fn : callable
        the function to evaluate the model.
        It can be CE, accuracy, logit for target token, etc.
    metric_fn_kwargs : dict
        the kwargs to pass to metric_fn. E.g. target token.
    complement : bool
        if True, the complement of the graph is used
    ablation_fn : callable
        the function used to get the patched states.
        E.g. : mean ablation means across batch and sequence length or only across one of them.
               zero ablation : patch states are just zeros
               id ablation : patch states are computed from the patch input and left unchanged
    default_ablation : str
        the default ablation function to use if patch is None
        Available : mean, zero
        Mean is across batch and sequence length by default.

    returns a dict
        threshold -> dict ('TODO:nedges/nnodes/avgdegre/anyothermetriconthegraph', 'metric', 'metric_comp', 'faithfulness', 'completeness')
            -> float
        'complete' -> float (metric on the original model)
        'empty' -> float (metric on the fully ablated model, no edges)
    """
    if isinstance(thresholds, float):
        thresholds = [thresholds]
    if patch is None and ablation_fn is None:
        if default_ablation == 'mean':
            ablation_fn = lambda x: x.mean(dim=(0, 1)).expand_as(x)
        elif default_ablation == 'zero':
            ablation_fn = lambda x: torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown default ablation function : {default_ablation}")
        
    results = {}

    # get metric on original model
    with model.trace(clean):
        metric = metric_fn(model, **metric_fn_kwargs).save()
    metric = metric.value.mean().item()
    results['complete'] = metric

    # get metric on empty graph
    mask = get_mask(circuit, -1)
    empty = run_graph(
        model,
        submodules,
        sae_dict,
        clean,
        patch,
        mask,
        metric_fn,
        metric_fn_kwargs,
        ablation_fn,
    ).mean().item()
    results['empty'] = empty

    # get metric on thresholded graph
    for threshold in thresholds:
        mask = get_mask(circuit, threshold)

        # TODO : get graph informations
        #results[threshold][TODO] = TODO
            
        threshold_result = run_graph(
            model,
            submodules,
            sae_dict,
            name_dict,
            clean,
            patch,
            mask,
            metric_fn,
            metric_fn_kwargs,
            ablation_fn,
        ).mean().item()
        results[threshold]['metric'] = threshold_result

        complement_result = run_graph(
            model,
            submodules,
            sae_dict,
            name_dict,
            clean,
            patch,
            mask,
            metric_fn,
            metric_fn_kwargs,
            ablation_fn,
            complement=True,
        ).mean().item()
        results[threshold]['metric_comp'] = complement_result

        results[threshold]['faithfulness'] = (threshold_result - empty) / (metric - empty)
        results[threshold]['completeness'] = (complement_result - empty) / (metric - empty)
    
    return results

##########
# sparsity, modularity and graph properties
##########

@torch.no_grad()
def to_Digraph(circuit, discard_res=False):
    """
    circuit : tuple (nodes, edges) or nk.Graph
    returns a networkx DiGraph
    """
    if isinstance(circuit, nx.DiGraph):
        return circuit
    elif isinstance(circuit, nk.Graph):
        return nk.nxadapter.nk2nx(circuit)
    elif isinstance(circuit, tuple):
        G = nx.DiGraph()

        nodes, edges = circuit

        for upstream in edges:
            for downstream in edges[upstream]:
                for d, u in edges[upstream][downstream].coalesce().indices():
                    # this weight matrix has shape (f_down + 1, f_up + 1)
                    # reconstruction error nodes are the last ones
                    if discard_res and (
                        d == edges[upstream][downstream].size(0) - 1
                        or u == edges[upstream][downstream].size(1) - 1
                    ):
                        continue
                    
                    upstream_name = f"{upstream}_{u}"
                    downstream_name = f"{downstream}_{d}"
                    G.add_edge(upstream_name, downstream_name)

        return G

def to_tuple(G):
    """
    G : nx.DiGraph or nk.Graph
    returns a tuple (nodes, edges)
    """
    raise NotImplementedError

@torch.no_grad()
def prune(
    G,
    return_tuple=False,
):
    """
    circuit : nx.DiGraph
    returns a new nx.DiGraph
    """

    G = G.copy()

    # save the 'embed' nodes and their edges to restore them later
    save = []
    for node in G.nodes:
        if False:# TODO
            save += G.edges(node)

    # merge nodes from embedding into a single 'embed' node, like 'y' is single.
    to_relable = {}
    for node in G.nodes:
        if False:# TODO
            to_relable[node] = 'embed'
    G = nx.relabel_nodes(G, to_relable)

    # do reachability from v -> 'y' for all v, remove all nodes that are not reachable
    reachable = nx.ancestors(G, 'y')
    reachable.add('y')

    complement = set(G.nodes) - reachable

    G.remove_nodes_from(complement)

    # do reachability from 'embed' -> v for all v, remove all nodes that are not reachable
    reachable = nx.descendants(G, 'embed')
    reachable.add('embed')

    complement = set(G.nodes) - reachable

    G.remove_nodes_from(complement)

    # untangle the 'embed' node into its original nodes and return the new graph
    G.remove_node('embed')
    for edge in save['edges']:
        G.add_edge(edge)

    if return_tuple:
        return to_tuple(G)
    return G

def get_avg_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()

def get_connected_components(G):
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected()
    return nx.number_connected_components(G)

def get_density(G):
    return nx.density(G)

def get_global_clustering_coefficient(G):
    return nx.average_clustering(G)

def get_transitivity(G):
    return nx.transitivity(G)

def get_degree_distribution(G):
    return nx.degree_histogram(G)

def get_s_metric(G):
    return nx.s_metric(G)

@torch.no_grad()
def sparsity(
    circuit,
    prune=False,
):
    """
    circuit : tuple (nodes, edges) or nx.DiGraph or nk.Graph
    returns a dict
        'nedges' -> int
        'nnodes' -> int
        'avgdegree' -> float
        ...
        'pruned' -> dict
            'nedges' -> int
            'nnodes' -> int
            'avgdegree' -> float
            ...
    
    other interesting metrics :
        - number of connected components
        - Global clustering coefficient # should be close or equal to 0, as the graph is layered
        - density
        - degree distribution
        - transitivity
        - s metric
    """

    if isinstance(circuit, tuple) or isinstance(circuit, nk.Graph):
        G = to_Digraph(circuit)
    else:
        G = circuit

    results = {
        'nedges' : G.number_of_edges(),
        'nnodes' : G.number_of_nodes(),
        'avgdegree' : get_avg_degree(G),
        'connected_components' : get_connected_components(G),
        'density' : get_density(G),
        'global_clustering_coefficient' : get_global_clustering_coefficient(G),
        'transitivity' : get_transitivity(G),
        'degree_distribution' : get_degree_distribution(G),
        's_metric' : get_s_metric(G),
    }

    if prune:
        pruned_circuit = prune(G)
        pruned = sparsity(pruned_circuit)
        results['pruned'] = pruned

    return results

def single_community_modularity(G, C, weighted=False):
    """
    G : nk.Graph
    C : set of nodes
    weighted : bool
    returns float

    Returns Sum_{e in E | e = (u in C, v in C)} w(e) / Sum_{e in E | e = (u in C, v in V) or (u in V, v in C)} w(e)
    """

    if not weighted:
        subgraph = nk.graphtools.subgraphFromNodes(G, C)
        sum1 = subgraph.numberOfEdges()

        sum2 = 0
        for node in C:
            sum2 += G.degree(node)
        sum2 -= sum1 # edges from C to C are counted twice, but not those from C to V \ C

        return sum1 / sum2
    else:
        raise NotImplementedError

# TODO : plot leiden results wrt iterations
@torch.no_grad()
def modularity(
    circuit,
    method='Leiden',
    iterations=10,
    gamma=1.0,
    weighted=False,
    log_weighted=False,
):
    """
    circuit : tuple (nodes, edges) or nx.DiGraph
    method : str
        only 'Leiden' is available for now
    iterations : int
        number of iterations for the algorithm
    gamma : float
        resolution parameter. Higher values lead to smaller communities
    weighted : bool
        if True, the graph is weighted
    log_weighted : bool
        if True, the weights are log( _ / eps )
    returns (dict : subset -> score), float)
    """


    if isinstance(circuit, tuple):
        G = to_Digraph(circuit)
        G = nk.nxadapter.nx2nk(G)
    else:
        G = nk.nxadapter.nx2nk(circuit)
        
    if weighted:
        raise NotImplementedError
        if log_weighted:
            G = nk.graphtools.toWeighted(G, edge_weight_log=True)
        else:
            G = nk.graphtools.toWeighted(G)

    # get communities
    if method == 'Leiden':
        partition = nk.community.ParallelLeiden(
            G,
            gamma=gamma,
            iterations=iterations,
        )
        partition.run()
        communities = partition.getPartition()
    else:
        raise NotImplementedError(f"Method {method} is not implemented")

    # get quality
    quality = nk.community.Modularity().getQuality(communities, G)

    subset_ids = communities.getSubsetIds()
    dict_communities = {
        subset_id : single_community_modularity(G, communities.getMembers(subset_id), weighted=weighted)
        for subset_id in subset_ids
    }
    
    return dict_communities, quality