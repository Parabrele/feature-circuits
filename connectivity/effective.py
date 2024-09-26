import gc

from collections import defaultdict

import torch as t

from connectivity.attribution import y_effect, get_effect
from utils.activation import get_hidden_states
from utils.sparse_coo_helper import rearrange_weights, aggregate_weights

# TODO : module-only

def get_circuit(
    clean,
    patch,
    model,
    dictionaries,
    metric_fn,
    embed,
    resids,
    attns=None,
    mlps=None,
    metric_kwargs=dict(),
    ablation_fn=None,
    aggregation='sum', # or 'none' for not aggregating across sequence position
    node_threshold=0.1,
    edge_threshold=0.01,
    steps=10,
    nodes_only=False,
    dump_all=False,
    save_path=None,
):
    return get_circuit_resid_only(
        clean,
        patch,
        model,
        embed,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        ablation_fn=ablation_fn,
        normalise_edges=False, # old test, not used anymore
        use_start_at_layer=False,
        aggregation=aggregation,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        nodes_only=nodes_only,
        steps=steps,
        dump_all=dump_all,
        save_path=save_path,
    )

def get_circuit_resid_only(
        clean,
        patch,
        model,
        embed,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        ablation_fn=None,
        normalise_edges=False, # whether to divide the edges entering a node by their sum
        use_start_at_layer=False, # Whether to compute the layer-wise effects with the start at layer argument to save computation
        aggregation='max', # or 'none' for not aggregating across sequence position
        node_threshold=0.1,
        edge_threshold=0.01,
        nodes_only=False,
        steps=10,
        dump_all=False,
        save_path=None,
):
    if dump_all and save_path is None:
        raise ValueError("If dump_all is True, save_path must be provided.")
    
    all_submods = [embed] + [submod for submod in resids]
    last_layer = resids[-1]
    n_layers = len(resids)
    
    # dummy forward pass to get shapes of outputs
    is_tuple = {}
    with model.trace("_"), t.no_grad():
        for submodule in all_submods:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    if use_start_at_layer:
        raise NotImplementedError
    
    # get encoding and reconstruction errors for clean and patch
    hidden_states_clean = get_hidden_states(model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=clean)

    if patch is None:
        patch = clean
    
    hidden_states_patch = get_hidden_states(model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=patch)
    hidden_states_patch = {k : ablation_fn(v) for k, v in hidden_states_patch.items()}
    
    features_by_submod = {}

    # start by the effect of the last layer to the metric
    edge_effect, nodes_attr = y_effect(
        model,
        clean, hidden_states_clean, hidden_states_patch,
        last_layer, all_submods,
        dictionaries, is_tuple,
        steps, metric_fn, metric_kwargs,
        normalise_edges, node_threshold, edge_threshold,
        features_by_submod
    )
    nodes = {}
    # print(f'resid_{len(resids)-1}')
    nodes[f'resid_{len(resids)-1}'] = nodes_attr[last_layer]

    if nodes_only:
        for layer in reversed(range(n_layers)):
            if layer > 0:
                # print(f'resid_{layer-1}')
                nodes[f'resid_{layer-1}'] = nodes_attr[resids[layer-1]]
            else:
                # print('embed')
                nodes['embed'] = nodes_attr[embed]

        print(nodes.keys())
        return nodes, {}
    
    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids)-1}'] = {'y' : edge_effect}
    
    # Now, backward through the model to get the effects of each layer on its successor.
    for layer in reversed(range(n_layers)):
        # print("Layer", layer, "threshold", edge_threshold)
        # print("Number of downstream features:", len(features_by_submod[resids[layer]]))
        
        resid = resids[layer]
        if layer > 0:
            prev_resid = resids[layer-1]
        else:
            prev_resid = embed

        RR_effect = get_effect(
            model,
            clean, hidden_states_clean, hidden_states_patch,
            dictionaries,
            layer, prev_resid, resid,
            features_by_submod,
            is_tuple, steps, normalise_edges,
            nodes_attr, node_threshold, edge_threshold,
        )
    
        if layer > 0:
            nodes[f'resid_{layer-1}'] = nodes_attr[prev_resid]
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect
        else:
            nodes['embed'] = nodes_attr[prev_resid]
            edges['embed'][f'resid_0'] = RR_effect
        
        gc.collect()
        t.cuda.empty_cache()

    rearrange_weights(nodes, edges)
    aggregate_weights(nodes, edges, aggregation, dump_all=dump_all, save_path=save_path)

    return nodes, edges
