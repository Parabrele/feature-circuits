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

from utils import SparseAct, get_hidden_states, sparse_coo_truncate

##########
# faithfulness & model evaluation
##########

def get_layer_idx(submod_name):
    module_split = submod_name.split('_')
    if module_split[0] in ['resid', 'attn', 'mlp']:
        return int(module_split[1])
    elif module_split[0] == 'embed':
        return 0
    else:
        raise ValueError(f"Unknown module : {submod_name}")

@torch.no_grad()
def get_mask(circuit, threshold):
    """
    circuit :
    edges : dict of dict of sparse_coo tensors, [name_upstream][name_downstream] -> edge weights

    returns a dict similar to edges but binary sparse coo tensors : only weights above threshold are kept
    """
    if isinstance(circuit, tuple):
        circuit = circuit[1]

    edges = circuit
    mask = {}

    for upstream in edges:
        mask[upstream] = {}
        for downstream in edges[upstream]:
            weights = edges[upstream][downstream].coalesce()
            if threshold == -1:
                mask[upstream][downstream] = torch.sparse_coo_tensor(
                    [[]] if downstream == 'y' else [[], []],
                    [],
                    weights.size()
                )
            else:
                value_mask = weights.values() > threshold
                mask[upstream][downstream] = torch.sparse_coo_tensor(
                    weights.indices()[:, value_mask],
                    torch.ones(value_mask.sum(), device=weights.device, dtype=torch.bool),
                    weights.size(),
                    dtype=torch.bool
                )
    return mask

def reorder_mask(edges):
    """
    mask : dict of dict of sparse_coo tensors
    returns a dict of dict of sparse_coo tensors
    """
    new_mask = {}
    for up in edges:
        for down in edges[up]:
            if down not in new_mask:
                new_mask[down] = {}
            new_mask[down][up] = edges[up][down]
    return new_mask

@torch.jit.script
def compiled_loop_pot_ali(mask_idx, potentially_alive, up_nz):
    # in the mask indices has shape (2, ...), potentially alive downstream features are indices in [0] st indices[1] is in up_nz
    for i, idx in enumerate(mask_idx[1]):
        if up_nz[idx]:
            potentially_alive[mask_idx[0][i]] = True

@torch.no_grad()
def run_graph(
        model,
        submodules,
        sae_dict,
        mod2name,
        clean,
        patch,
        circuit,
        metric_fn,
        metric_fn_kwargs,
        ablation_fn,
        complement=False,
        clean_logits=None,
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

    # various initializations
    circuit = reorder_mask(circuit)

    if complement:
        raise NotImplementedError("Complement is not implemented yet")

    name2mod = {v : k for k, v in mod2name.items()}
    
    is_tuple = {}
    input_is_tuple = {}
    with model.trace("_"), torch.no_grad():
        for submodule in submodules:
            input_is_tuple[submodule] = type(submodule.input.shape) == tuple
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    if patch is None:
        patch = clean

    # get patch hidden states
    patch_states = get_hidden_states(model, submodules, sae_dict, is_tuple, patch)
    patch_states = {k : ablation_fn(v) for k, v in patch_states.items()}

    # forward through the model by computing each node as described by the graph and not as the original model does

    # For each downstream module, get it's potentially alive features (heuristic to not compute one forward pass per node
    # as there are too many of them) by reachability from previously alive ones
    # Then, for each of these features, get it's masked input, and compute a forward pass to get this particular feature.
    # This gives the new state for this downstream output.

    # with model.trace(clean):
    hidden_states = {}
    for downstream in submodules:
        # get downstream dict, output, ...
        downstream_dict = sae_dict[downstream]
        down_name = mod2name[downstream]

        # TOD? : this surely can be replaced by a single call to trace, or none at all
        with model.trace(clean):
            if input_is_tuple[downstream]:
                input_shape = downstream.input[0].shape
                input_dict = downstream.input[1:].save()
            else:
                input_shape = downstream.input.shape
            
            x = downstream.output
            if is_tuple[downstream]:
                x = x[0]
            x = x.save()
        
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]
            
        x_hat, f = downstream_dict(x, output_features=True)
        res = x - x_hat

        # if downstream is embed, there is no upstream and the result stays unchanged
        if down_name == 'embed' or downstream == submodules[0]:
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
        # TOD? : for features whose masks are all zeros, skip and keep only the patch state
        
        potentially_alive = torch.zeros(f.shape[-1] + 1, device=f.device, dtype=torch.bool)
        for up_name in circuit[down_name]:
            upstream = name2mod[up_name]
            mask = circuit[down_name][up_name] # shape (f_down + 1, f_up + 1)

            upstream_hidden = hidden_states[upstream].act # shape (batch, seq_len, f_up)
            # reduce to (f_up) by maxing over batch and seq_len (should only have positive entries, but a .abs() can't hurt)
            upstream_hidden = upstream_hidden.abs().amax(dim=(0, 1)) # shape (f_up)
            up_nz = torch.cat([upstream_hidden > 0, torch.tensor([True], device=f.device)]) # shape (f_up + 1). Always keep the res feature alive
            
            compiled_loop_pot_ali(mask.indices(), potentially_alive, up_nz)

        potentially_alive = potentially_alive.nonzero().squeeze(1)

        f[...] = patch_states[downstream].act # shape (batch, seq_len, f_down)
        for f_ in potentially_alive:
            edge_ablated_input = torch.zeros(tuple(input_shape)).to(f.device)
            for up_name in circuit[down_name]:
                upstream = name2mod[up_name]
                upstream_dict = sae_dict[upstream]
                
                mask = circuit[down_name][up_name][f_].to_dense() # shape (f_up + 1)

                edge_ablated_upstream = SparseAct(
                    act = patch_states[upstream].act,
                    res = hidden_states[upstream].res if mask[-1] else patch_states[upstream].res
                )
                edge_ablated_upstream.act[:, :, mask[:-1]] = hidden_states[upstream].act[:, :, mask[:-1]]

                edge_ablated_input += upstream_dict.decode(edge_ablated_upstream.act) + edge_ablated_upstream.res

            module_type = down_name.split('_')[0]
            # TODO : batch these forward passes to speed up the process
            if module_type == 'resid':
                # if resid only, do this, othewise, should be literally the identity as the sum gives resid_post already.
                if input_is_tuple[downstream]:
                    edge_ablated_out = downstream.forward(edge_ablated_input, **input_dict.value[0])
                else:
                    edge_ablated_out = downstream.forward(edge_ablated_input)
                if is_tuple[downstream]:
                    edge_ablated_out = edge_ablated_out[0]
            else:
                # if attn or mlp, use corresponding LN
                raise NotImplementedError(f"Support for module type {module_type} is not implemented yet")
            if f_ < f.shape[-1]:
                # TODO : add option in sae forward to get only one feature to fasten this
                #        only after testing that it works like this first and then checking that it
                #        is faster and doesn't break anything
                #        more generally, try to compress each node function as much as possible.
                f[..., f_] = downstream_dict.encode(edge_ablated_out)[..., f_] # replace by ", f_)"
            else:
                res = edge_ablated_out - downstream_dict(edge_ablated_out)

        hidden_states[downstream] = SparseAct(act=f, res=res)

        # if is_tuple[downstream]:
        #     downstream.output[0][:] = downstream_dict.decode(f) + res
        # else:
        #     downstream.output = downstream_dict.decode(f) + res

    last_layer = submodules[-1]
    with model.trace(clean):
        if is_tuple[last_layer]:
            last_layer.output[0][:] = sae_dict[last_layer].decode(hidden_states[last_layer].act) + hidden_states[last_layer].res
        else:
            last_layer.output = sae_dict[last_layer].decode(hidden_states[last_layer].act) + hidden_states[last_layer].res
        
        if isinstance(metric_fn, dict):
            metric = {}
            for name, fn in metric_fn.items():
                if name == "KL":
                    metric[name] = fn(model, clean_logits=clean_logits, **metric_fn_kwargs).save()
                else:
                    metric[name] = fn(model, **metric_fn_kwargs).save()
        else:
            metric = metric_fn(model, **metric_fn_kwargs).save()

    if isinstance(metric, dict):
        return {name : value.value.mean().item() for name, value in metric.items()}
    return metric.value.mean().item()

# TODO : compute statistics on the graph (nb of edges, nb of nodes)
# TODO : use several metric fcts to evaluate the model (logit difference, CE, KL, accuracy, 1/rank, etc.)
#        compute them all at once to save compute time
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
    circuit : edges
    thresholds : float or list of float
        the thresholds to discard edges based on their weights
    metric_fn : callable or dict name -> callable
        the function(s) to evaluate the model.
        It can be CE, accuracy, logit for target token, etc.
    metric_fn_kwargs : dict
        the kwargs to pass to metric_fn. E.g. target token.
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
            ablation_fn = lambda x: SparseAct(torch.zeros_like(x.act), torch.zeros_like(x.res))
        else:
            raise ValueError(f"Unknown default ablation function : {default_ablation}")
        
    results = {}

    # get metric on original model
    with model.trace(clean):
        clean_logits = model.output[0][:,-1,:].save()
        if isinstance(metric_fn, dict):
            metric = {}
            for name, fn in metric_fn.items():
                if name == "KL":
                    metric[name] = fn(model, clean_logits=clean_logits, **metric_fn_kwargs).save()
                else:
                    metric[name] = fn(model, **metric_fn_kwargs).save()
        else:
            metric = metric_fn(model, **metric_fn_kwargs).save()
    if isinstance(metric, dict):
        results['complete'] = {}
        for name, value in metric.items():
            results['complete'][name] = value.value.mean().item()
    else:
        results['complete'] = metric.value.mean().item()

    # get metric on empty graph
    mask = get_mask(circuit, -1)
    empty = run_graph(
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
        clean_logits=clean_logits
    )
    results['empty'] = empty

    # get metric on thresholded graph
    for threshold in thresholds:
        results[threshold] = {}

        mask = get_mask(circuit, threshold)
        pruned = prune(mask)

        results[threshold]['n_nodes'] = get_n_nodes(pruned)
        results[threshold]['n_edges'] = get_n_edges(pruned)
        results[threshold]['avg_degree'] = results[threshold]['n_edges'] / results[threshold]['n_nodes']
        results[threshold]['density'] = get_density(pruned)
        # TODO : results[threshold]['modularity'] = modularity, as in kaarel, as in modularity in NN paper, as in me
        #        results[threshold]['z_score'] = Z_score(pruned)
            
        threshold_result = run_graph(
            model,
            submodules,
            sae_dict,
            name_dict,
            clean,
            patch,
            pruned,
            metric_fn,
            metric_fn_kwargs,
            ablation_fn,
            clean_logits=clean_logits
        )
        results[threshold]['metric'] = threshold_result

        # complement_result = run_graph(
        #     model,
        #     submodules,
        #     sae_dict,
        #     name_dict,
        #     clean,
        #     patch,
        #     pruned,
        #     metric_fn,
        #     metric_fn_kwargs,
        #     ablation_fn,
        #     complement=True,
        # ).mean().item()
        # results[threshold]['metric_comp'] = complement_result

        if isinstance(metric, dict):
            results[threshold]['faithfulness'] = {}
            for name, value in metric.items():
                if "rank" in name or "acc" in name:
                    results[threshold]['faithfulness'][name] = threshold_result[name] / results['complete'][name]
                elif name == "KL":
                    results[threshold]['faithfulness'][name] = threshold_result[name]
                else:
                    results[threshold]['faithfulness'][name] = (threshold_result[name] - empty[name]) / (metric[name] - empty[name])
        else:
            results[threshold]['faithfulness'] = (threshold_result - empty) / (metric - empty)
    
    return results

##########
# sparsity, modularity and graph properties
##########

@torch.no_grad()
def to_Digraph(circuit, discard_res=False, discard_y=False):
    """
    circuit : tuple (nodes, edges), dict or nk.Graph
    returns a networkx DiGraph
    """
    if isinstance(circuit, nx.DiGraph):
        return circuit
    elif isinstance(circuit, nk.Graph):
        return nk.nxadapter.nk2nx(circuit)
    elif isinstance(circuit, tuple) or isinstance(circuit, dict):
        G = nx.DiGraph()

        if isinstance(circuit, tuple):
            nodes, edges = circuit
        else:
            edges = circuit

        for upstream in edges:
            for downstream in edges[upstream]:
                if downstream == 'y':
                    if discard_y:
                        continue
                    else:
                        for u in edges[upstream][downstream].coalesce().indices().t():
                            u = u.item()
                            if discard_res and u == edges[upstream][downstream].size(0) - 1:
                                continue
                            upstream_name = f"{upstream}_{u}"
                            G.add_edge(upstream_name, downstream)
                        continue
                for d, u in edges[upstream][downstream].coalesce().indices().t():
                    d = d.item()
                    u = u.item()
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

def prune(
    circuit
):
    """
    circuit : nx.DiGraph or dict of dict of sparse_coo tensors
    returns a new nx.DiGraph or dict of dict of sparse_coo tensors
    """
    if isinstance(circuit, nx.DiGraph):
        return prune_nx(circuit)
    else:
        return prune_sparse_coos(circuit)

def reorder_upstream(edges):
    """
    edges : dict of dict of sparse_coo tensors
    returns a dict of dict of sparse_coo tensors
    """
    new_edges = {}
    for up in reversed(list(edges.keys())):
        new_edges[up] = edges[up]
    return new_edges

def coalesce_edges(edges):
    """
    coalesce all edges sparse coo weights
    """
    for up in edges:
        for down in edges[up]:
            edges[up][down] = edges[up][down].coalesce()
    return edges

@torch.no_grad()
def prune_sparse_coos(
    circuit
):
    """
    circuit : edges
    returns a new edges

    Assumes edges is a dict of bool sparse coo tensors. If not, it will just forget the values.
    """
    circuit = coalesce_edges(reorder_upstream(circuit))

    # build a dict module_name -> sparse vector of bools, where True means that the feature is reachable from one embed node
    size = {}
    reachable = {}

    for upstream in circuit:
        for downstream in circuit[upstream]:
            if downstream == 'y':
                size[upstream] = circuit[upstream][downstream].size(0)
                size[downstream] = 1
            else:
                size[upstream] = circuit[upstream][downstream].size(1)
                size[downstream] = circuit[upstream][downstream].size(0)                    
    
    # build a sparse_coo_tensor of ones with size (size['embed']) :
    reachable['embed'] = torch.sparse_coo_tensor(
        torch.arange(size['embed']).unsqueeze(0),
        torch.ones(size['embed'], device=circuit[upstream][downstream].device, dtype=torch.bool),
        (size['embed'],),
        device=circuit[upstream][downstream].device,
        dtype=torch.bool
    ).coalesce()

    for upstream in circuit:
        for downstream in circuit[upstream]:
            if upstream not in reachable:
                raise ValueError(f"Upstream {upstream} reachability not available. Check the order of the keys.")
            if downstream not in reachable:
                reachable[downstream] = torch.sparse_coo_tensor(
                    [[]], [],
                    (size[downstream],),
                    device=circuit[upstream][downstream].device
                )

            idx1 = circuit[upstream][downstream].indices() # (2, n1)
            idx2 = reachable[upstream].indices() # (1, n2)

            # keep only rows of circuit[upstream][downstream] at idx in idx2
            new_edges = torch.sparse_coo_tensor(
                [[]] if downstream == 'y' else [[], []],
                [],
                circuit[upstream][downstream].size(),
                device=circuit[upstream][downstream].device,
                dtype=torch.bool
            ).coalesce()
            for u in idx2[0]:
                mask = (idx1[0] == u if downstream == 'y' else idx1[1] == u)
                new_edges = torch.sparse_coo_tensor(
                    torch.cat([new_edges.indices(), idx1[:, mask]], dim=1),
                    torch.cat([new_edges.values(), circuit[upstream][downstream].values()[mask]]),
                    new_edges.size(),
                    device=circuit[upstream][downstream].device,
                    dtype=torch.bool
                ).coalesce()

            circuit[upstream][downstream] = new_edges

            # now look at what downstream features are reachable, as just the indices in the new_edges and add them to reachable[downstream]
            idx = new_edges.indices()[0].unique()
            reachable[downstream] += torch.sparse_coo_tensor(
                idx.unsqueeze(0),
                torch.ones(idx.size(0)),
                (size[downstream],),
                device=circuit[upstream][downstream].device
            )
            reachable[downstream] = reachable[downstream].coalesce()
    
    return circuit

@torch.no_grad()
def prune_nx(
    G
):
    """
    circuit : nx.DiGraph
    returns a new nx.DiGraph
    """

    G = G.copy()

    # save the 'embed' nodes and their edges to restore them later
    save = []
    to_relabel = {}
    for node in G.nodes:
        if 'embed' in node:
            save += G.edges(node)
            to_relabel[node] = 'embed'

    # merge nodes from embedding into a single 'embed' node, like 'y' is single.
    G = nx.relabel_nodes(G, to_relabel)

    # do reachability from v -> 'y' for all v, remove all nodes that are not reachable
    reachable = nx.ancestors(G, 'y')
    reachable.add('y')
    complement = set(G.nodes) - reachable

    G.remove_nodes_from(complement)

    # do reachability from 'embed' -> v for all v, remove all nodes that are not reachable
    reachable = nx.descendants(G, 'embed')
    complement = set(G.nodes) - reachable

    G.remove_nodes_from(complement)

    # untangle the 'embed' node into its original nodes and return the new graph
    G.add_edges_from(save)

    return G

def get_n_nodes(G):
    if isinstance(G, nx.DiGraph):
        return G.number_of_nodes()
    elif isinstance(G, dict):
        n_nodes = 0
        for up in G:
            up_nodes = torch.empty(0, dtype=torch.long)
            for down in G[up]:
                nodes = G[up][down].indices()[1].unique()
                up_nodes = torch.cat([up_nodes, nodes])
            n_nodes += up_nodes.unique().size(0)
        return n_nodes
    else :
        raise ValueError("Unknown graph type")

def get_n_edges(G):
    if isinstance(G, nx.DiGraph):
        return G.number_of_edges()
    elif isinstance(G, dict):
        n_edges = 0
        for up in G:
            for down in G[up]:
                n_edges += G[up][down].values().size(0)
        return n_edges
    else :
        raise ValueError("Unknown graph type")

def get_avg_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()

def get_connected_components(G):
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected()

    G = G.copy()
    G.remove_node('y')

    return nx.number_connected_components(G)

def get_density(edges):
    # edges is a dict of dict of sparse_coo tensors
    if isinstance(edges, nx.DiGraph):
        return nx.density(edges)
    n_edges = 0
    max_edges = 0
    for up in edges:
        for down in edges[up]:
            n_edges += edges[up][down].values().size(0)
            max_edges += edges[up][down].size(0) * edges[up][down].size(1)
    return n_edges / max_edges

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
    circuit : tuple (nodes, edges), dict or nx.DiGraph or nk.Graph
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

    if isinstance(circuit, tuple) or isinstance(circuit, dict) or isinstance(circuit, nk.Graph):
        G = to_Digraph(circuit)
    else:
        G = circuit

    results = {
        'nedges' : G.number_of_edges(),
        'nnodes' : G.number_of_nodes(),
        'avgdegree' : get_avg_degree(G),
        'connected_components' : get_connected_components(G),
        'density' : get_density(circuit),
        'degree_distribution' : get_degree_distribution(G),
    }

    if prune:
        pruned_circuit = prune(G)
        pruned = sparsity(pruned_circuit)
        results['pruned'] = pruned

    return results

def shuffle_edges(edges):
    """
    edges : dict of dict of sparse_coo tensors
    returns a dict of dict of sparse_coo tensors, with edges uniformly randomly assigned to the active nodes
    """
    new_edges = {}
    for up in edges:
        new_edges[up] = {}
        for down in edges[up]:
            up_alive = edges[up][down].indices()[1].unique()
            down_alive = edges[up][down].indices()[0].unique()
            n_edges = edges[up][down].values().size(0)
            new_edges[up][down] = torch.sparse_coo_tensor(
                torch.stack([
                    down_alive[torch.randint(down_alive.size(0), (n_edges,))],
                    up_alive[torch.randint(up_alive.size(0), (n_edges,))]
                ]),
                torch.ones(n_edges, device=edges[up][down].device, dtype=torch.bool),
                edges[up][down].size(),
                device=edges[up][down].device,
                dtype=torch.bool
            ).coalesce()
    return new_edges

# TODO : to compare to kaarel's work, use [2] (quality, as defined by nx, optimized by leiden). Not good results on LIB even on toy models,
#        no modularity in language models graphs. Ask them quantitative results to be able to compare.
#        to compare to clusterability in NN paper, use their metric. They use spectral clustering, but I
#        have really huge but sparse matrices, so I don't know if it is possible. This includes weights so it's nice.
#        use my metric just to see if it is nice.
def mean_std_clusterability(edges, n_samples=50):
    """
    edges : dict of dict of sparse_coo tensors
    returns a tuple (mean, std) of the clusterability of the graph
    """
    clusterabilities = []
    for _ in range(n_samples):
        shuffled_edges = shuffle_edges(edges)
        clusterabilities.append(modularity(shuffled_edges)[1])
    return torch.tensor(clusterabilities).mean().item(), torch.tensor(clusterabilities).std().item()

def Z_score(edges, n_samples=50):
    """
    edges : dict of dict of sparse_coo tensors
    returns a tuple (mean, std) of the clusterability of the graph
    """
    mean, std = mean_std_clusterability(edges, n_samples)
    return (modularity(edges)[1] - mean) / std

# TODO : look into normalised cut paper to define a global metric similar to that.
# TODO : do that from clusterability in NN :
"""
To measure the relative clusterability of an MLP, we sample 50 random networks by randomly shuffling the weight
matrix of each layer of the trained network. We convert these
networks to graphs, cluster them, and find their n-cuts. For
CNNs, we shuffle the edge weights between channels once
the network has been turned into a graph, which is equivalent to shuffling which two channels are connected by each
spatial kernel slice. We then compare the n-cut of the trained
network to the sampled n-cuts, estimating the left one-sided
p-value (North, Curtis, and Sham 2002) and the Z-score: the
number of standard deviations the network’s n-cut lies below or above the mean of the shuffle distribution. This determines whether the trained network is more clusterable than
one would p
"""
# TODO : he does it with raw weights : try to rethink a way to do raw weight graph in transformers.
# TODO : https://openreview.net/pdf?id=HJghQ7YU8H already showed success in identifying functional modules in NNs
# TODO : do the same as they did, instead of having a graph, have a matrix of (nfeatures * nsamples) and bicluster this.
# TODO : they did not try to kill all neurons except for those in a module to see if the model is still working.
#        they only found neurons that fire in the same patterns. Link to neuro science ?
# TODO : Maybe this can help reduce the cost of finding the graph : fisrt look at what nodes are likely to be in the same module
#        then define nodes inside this module and voilà, here is the circuit for this unsupervisedly found behavior.
# TODO : in fact they do it for a single layer, they basically look at the correlation between neurons.
#        this is more or less what I dit with SAE dict correlation
#        maybe this could lead to another measure of the quality of the graph : nodes with high correlation should belong to the same module
#        and maybe modules as defined in this article should be the communities given by leiden, or be close, or at least should be good communities.
#        even if not really related to the one found by leiden. Compare the functional interpretablitity of the two methods.
#
#       Technical detail : to get the big vector of active features, don't just concatenate feature vectors of one token, aggregate over sequence
#       to get all features involved.
# TODO : speak to martin ester.
def single_community_modularity(G, C, weighted=False, output_n_edges=False):
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

        if sum2 == 0:
            if output_n_edges:
                return sum1, 0
            return 0
        if output_n_edges:
            return sum1, sum1 / sum2
        return sum1 / sum2
    else:
        raise NotImplementedError

# TOD? : plot leiden results wrt iterations (or plot [quality o (communities o to_merged_graph)^n](communities)) : do communities of communities
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
    circuit : tuple (nodes, edges), dict or nx.DiGraph
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
    returns (dict : subset -> score), float, float
    """

    if isinstance(circuit, tuple) or isinstance(circuit, dict):
        circuit = to_Digraph(circuit)
    
    try :
        circuit = circuit.copy()
        circuit.remove_node('y')
    except:
        pass

    G = nk.nxadapter.nx2nk(circuit)
    G = nk.graphtools.toUndirected(G)
        
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
        subset_id : {
            'n_nodes' : len(communities.getMembers(subset_id)),
            'score' : single_community_modularity(G, communities.getMembers(subset_id), weighted=weighted, output_n_edges=True)
        }
        for subset_id in subset_ids
    }

    to_del = []
    for subset_id in dict_communities:
        if dict_communities[subset_id]['n_nodes'] == 1:
            to_del.append(subset_id)
            continue
        dict_communities[subset_id]['n_edges'], dict_communities[subset_id]['score'] = dict_communities[subset_id]['score']
    
    for subset_id in to_del:
        del dict_communities[subset_id]

    # normalize by number of nodes ? larger communities are more likely to have better scores... choose whether to do it or not based on dict_communities
    avg_single_community_modularity = sum([v['score'] * v['n_nodes'] for v in dict_communities.values()]) / sum([v['n_nodes'] for v in dict_communities.values()])

    return dict_communities, avg_single_community_modularity, quality