import gc

from collections import defaultdict

import torch as t
from einops import rearrange

from connectivity.attribution import jvp, patching_effect, y_effect, get_effect
from utils.activation import SparseAct, get_hidden_states
from utils.sparse_coo_helper import rearrange_weights, aggregate_weights

# TODO : module-only

available_methods = ['resid', 'marks', 'resid_topk']

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
    method='resid',
    aggregation='max', # or 'none' for not aggregating across sequence position
    node_threshold=0.1,
    edge_threshold=0.01,
    steps=10,
    nodes_only=False,
    dump_all=False,
    save_path=None,
):
    if method not in available_methods:
        raise ValueError(f"Unknown circuit discovery method: {method}")
    if method == 'marks':
        if (attns is None or mlps is None):
            raise ValueError("Original marks method requires attns and mlps to be provided")
        else:
            return get_circuit_marks(
                clean,
                patch,
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                metric_fn,
                metric_kwargs=metric_kwargs,
                aggregation=aggregation,
                nodes_only=nodes_only,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
    else:
        return get_circuit_resid_only(
            clean,
            patch,
            model,
            embed,
            resids,
            dictionaries,
            metric_fn,
            metric_kwargs=metric_kwargs,
            normalise_edges=(method == 'resid_topk'),
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
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
    else:
        hidden_states_patch = get_hidden_states(model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=patch)
    
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
    print(f'resid_{len(resids)-1}')
    nodes[f'resid_{len(resids)-1}'] = nodes_attr[last_layer]

    if nodes_only:
        for layer in reversed(range(n_layers)):
            if layer > 0:
                print(f'resid_{layer-1}')
                nodes[f'resid_{layer-1}'] = nodes_attr[resids[layer-1]]
            else:
                print('embed')
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

def get_circuit_marks(
        clean,
        patch,
        model,
        embed,
        attns,
        mlps,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum', # or 'none' for not aggregating across sequence position
        nodes_only=False,
        node_threshold=0.1,
        edge_threshold=0.01,
):
    all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    
    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method='ig' # get better approximations for early layers by using ig
    )

    def unflatten(tensor): # will break if dictionaries vary in size between layers
        b, s, f = effects[resids[0]].act.shape
        unflattened = rearrange(tensor, '(b s x) -> b s x', b=b, s=s)
        return SparseAct(act=unflattened[...,:f], res=unflattened[...,f:])
    
    features_by_submod = {
        submod : (effects[submod].to_tensor().flatten().abs() > node_threshold).nonzero().flatten().tolist() for submod in all_submods
    }

    n_layers = len(resids)

    nodes = {'y' : total_effect}
    nodes['embed'] = effects[embed]
    for i in range(n_layers):
        nodes[f'attn_{i}'] = effects[attns[i]]
        nodes[f'mlp_{i}'] = effects[mlps[i]]
        nodes[f'resid_{i}'] = effects[resids[i]]

    if nodes_only:
        if aggregation == 'sum':
            for k in nodes:
                if k != 'y':
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None

    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids)-1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten().to_sparse() }

    def N(upstream, downstream):
        return jvp(
            clean,
            model,
            dictionaries,
            downstream,
            features_by_submod[downstream],
            upstream,
            grads[downstream],
            deltas[upstream],
            return_without_right=True,
        )

    # now we work backward through the model to get the edges
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        MR_effect, MR_grad = N(mlp, resid)
        AR_effect, AR_grad = N(attn, resid)

        edges[f'mlp_{layer}'][f'resid_{layer}'] = MR_effect
        edges[f'attn_{layer}'][f'resid_{layer}'] = AR_effect

        if layer > 0:
            prev_resid = resids[layer-1]
        else:
            prev_resid = embed

        RM_effect, _ = N(prev_resid, mlp)
        RA_effect, _ = N(prev_resid, attn)

        MR_grad = MR_grad.coalesce()
        AR_grad = AR_grad.coalesce()

        RMR_effect = jvp(
            clean,
            model,
            dictionaries,
            mlp,
            features_by_submod[resid],
            prev_resid,
            {feat_idx : unflatten(MR_grad[feat_idx].to_dense()) for feat_idx in features_by_submod[resid]},
            deltas[prev_resid],
        )
        RAR_effect = jvp(
            clean,
            model,
            dictionaries,
            attn,
            features_by_submod[resid],
            prev_resid,
            {feat_idx : unflatten(AR_grad[feat_idx].to_dense()) for feat_idx in features_by_submod[resid]},
            deltas[prev_resid],
        )
        RR_effect, _ = N(prev_resid, resid)

        if layer > 0: 
            edges[f'resid_{layer-1}'][f'mlp_{layer}'] = RM_effect
            edges[f'resid_{layer-1}'][f'attn_{layer}'] = RA_effect
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect - RMR_effect - RAR_effect
        else:
            edges['embed'][f'mlp_{layer}'] = RM_effect
            edges['embed'][f'attn_{layer}'] = RA_effect
            edges['embed'][f'resid_0'] = RR_effect - RMR_effect - RAR_effect

    # rearrange weight matrices
    rearrange_weights(nodes, edges)
    aggregate_weights(nodes, edges, aggregation)

    return nodes, edges
