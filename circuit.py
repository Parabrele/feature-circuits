import argparse
import gc
import json
import math
import os
from collections import defaultdict

import torch as t
from einops import rearrange
from tqdm import tqdm

from utils import SparseAct
from utils import load_examples, load_examples_nopair
from utils import plot_circuit, plot_circuit_posaligned
from utils import rearrange_weights, aggregate_weights
from utils import get_hidden_states

from attribution import y_effect, get_effect
from attribution import patching_effect, jvp

from dictionary_learning import AutoEncoder
from nnsight import LanguageModel

tracer_kwargs = {'validate' : False, 'scan' : False}

def marche_pas():
    def decorator(f):
        def wrapper(*args, **kwargs):
            raise NotImplementedError("This function doesn't work yet")
        return wrapper
    return decorator

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
    original_marks=True, # whether to compute the circuit using the original marks method
    normalise_edges=False, # whether to divide the edges entering a node by their sum
    use_start_at_layer=False, # Whether to compute the layer-wise effects with the start at layer argument to save computation
    aggregation='max', # or 'none' for not aggregating across sequence position
    node_threshold=0.1,
    edge_threshold=0.01,
    steps=10,
    nodes_only=False,
):
    if original_marks :
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
            normalise_edges=normalise_edges,
            use_start_at_layer=use_start_at_layer,
            aggregation=aggregation,
            edge_threshold=edge_threshold,
            steps=steps,
        )

# TODO : separate batch agreg (default : max) and seq agreg (default : sum)
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
        edge_threshold=0.01,
        steps=10,
):
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
    edge_effect, node_effect = y_effect(
        model,
        clean, hidden_states_clean, hidden_states_patch,
        last_layer, dictionaries, is_tuple,
        steps,
        metric_fn, metric_kwargs,
        normalise_edges, edge_threshold,
        features_by_submod
    )

    nodes = {'y' : t.tensor([1.0]).to(node_effect.act.device)}
    nodes[f'resid_{n_layers-1}'] = node_effect

    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids)-1}'] = {'y' : edge_effect}
    
    # Now, backward through the model to get the effects of each layer on its successor.
    for layer in reversed(range(n_layers)):
        resid = resids[layer]
        if layer > 0:
            prev_resid = resids[layer-1]
        else:
            prev_resid = embed

        RR_effect, max_effect = get_effect(
            model,
            clean, hidden_states_clean, hidden_states_patch,
            dictionaries,
            layer, prev_resid, resid,
            features_by_submod,
            is_tuple, steps, normalise_edges, edge_threshold
        )
    
        if layer > 0:
            nodes[f'resid_{layer-1}'] = max_effect
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect
        else:
            nodes['embed'] = max_effect
            edges['embed'][f'resid_0'] = RR_effect

    rearrange_weights(nodes, edges)
    aggregate_weights(nodes, edges, aggregation)

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

@marche_pas()
def get_circuit_stop_at_layer(
        clean,
        patch,
        model,
        embed,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum', # or 'none' for not aggregating across sequence position
        edge_threshold=0.01,
        steps=10,
):
    """
    TODO :
    Integrate into get circuit.
    The loop with model.trace and then tracer.invoke will need reworking.
    """

    n_layers = len(resids)

    all_submods = [embed] + [submod for submod in resids]
    
    # dummy forward pass to get shapes of outputs
    is_tuple = {}
    with model.trace("_"), t.no_grad():
        for submodule in all_submods:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    _, _, _, attn_mask = model.input_to_embed(clean)
    # get encoding and reconstruction errors for clean and patch
    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.no_grad():
        for submodule in all_submods:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            upstream_act = dictionary.encode(x)
            x_hat = dictionary.decode(upstream_act)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=upstream_act.save(), res=residual.save())
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.no_grad():
            for submodule in all_submods:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                upstream_act = dictionary.encode(x)
                x_hat = dictionary.decode(upstream_act)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=upstream_act.save(), res=residual.save())
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    # start by the effect of the last layer to the metric
    last_layer = all_submods[-1]

    dictionary = dictionaries[last_layer]
    clean_state = hidden_states_clean[last_layer]
    patch_state = hidden_states_patch[last_layer]

    metrics = []
    fs = []

    # TODO : tracer.invoke batches all the calls automatically. Using start_at_layer is not compatible using .invoke : batch the calls manually and do only one model.trace
    for step in range(steps):
        alpha = step / steps

        upstream_act = (1 - alpha) * clean_state + alpha * patch_state
        upstream_act.act.retain_grad()
        upstream_act.res.retain_grad()

        fs.append(upstream_act)

        intervention = dictionary.decode(upstream_act.act) + upstream_act.res
        with model.trace(
            t.zeros_like(intervention),
            start_at_layer=n_layers-1,
            attention_mask=attn_mask,
            **tracer_kwargs
        ):
            if is_tuple[last_layer]:
                last_layer.output[0][:] = intervention
            else:
                last_layer.output = intervention

            metrics.append(metric_fn(model, **metric_kwargs).save())
    metric = sum([m for m in metrics])
    metric.sum().backward(retain_graph=True)

    mean_grad = sum([f.act.grad for f in fs]) / steps
    mean_residual_grad = sum([f.res.grad for f in fs]) / steps
    grad = SparseAct(act=mean_grad, res=mean_residual_grad)
    delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
    effect = grad @ delta

    tot_eff = effect.to_tensor().sum()
    effect = effect / tot_eff

    nodes = {'y' : t.tensor([1])}
    nodes[f'resid_{n_layers-1}'] = effect

    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids)-1}'] = {
        'y' : effect.to_tensor().flatten().to_sparse()
    }

    features_by_submod = {
        last_layer : (effect.to_tensor().flatten().abs() > edge_threshold).nonzero().flatten().tolist()
    }

    # Now, backward through the model to get the effects of each layer on its successor.

    # now we work backward through the model to get the edges
    for layer in reversed(range(n_layers)):
        
        resid = resids[layer]
        if layer > 0:
            prev_resid = resids[layer-1]
        else:
            prev_resid = embed
        
        downstream_submod = resid
        upstream_submod = prev_resid

        downstream_features = features_by_submod[downstream_submod]

        if not features_by_submod[resid]: # handle empty list
            features_by_submod[prev_resid] = []
            RR_effect = t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.cfg.device)
            break
        
        else:
            downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]

            effect_indices = {}
            effect_values = {}
            dictionary = dictionaries[prev_resid]
            clean_state = hidden_states_clean[prev_resid]
            patch_state = hidden_states_patch[prev_resid]

            fs_grads = [0 for _ in range(len(downstream_features))]
            
            for step in range(steps):
                alpha = step / steps
                upstream_act = (1 - alpha) * clean_state + alpha * patch_state
                upstream_act.act.retain_grad()
                upstream_act.res.retain_grad()

                intervention = dictionary.decode(upstream_act.act) + upstream_act.res

                start_layer = layer - 1 if layer > 0 else None 
                with model.trace(
                    t.zeros_like(intervention),
                    start_at_layer=start_layer,
                    stop_at_layer=layer+1,
                    attention_mask=attn_mask,
                    **tracer_kwargs
                ):
                    if is_tuple[upstream_submod]:
                        upstream_submod.output[0][:] = dictionary.decode(upstream_act.act) + upstream_act.res
                    else:
                        upstream_submod.output = dictionary.decode(upstream_act.act) + upstream_act.res


                    y = downstream_submod.output
                    if is_tuple[downstream_submod]:
                        y = y[0]
                    y_hat, g = downstream_dict(y, output_features=True)
                    y_res = y - y_hat
                    # the left @ down in the original code was at least useful to populate .resc instead of .res. I should just du .resc = norm of .res :
                    # all values represent the norm of their respective feature, so if we consider .res as a feature, then we should indeed
                    # consider its norm as the node.
                    # /!\ do the .to_tensor().flatten() outside of the with in order for the proxies to be populated and .to_tensor() to not crash
                    downstream_act = SparseAct(
                        act=g,
                        resc=t.norm(y_res, dim=-1)
                    ).save()
                
                downstream_act = downstream_act.to_tensor().flatten()
                
                for i, downstream_feat in enumerate(downstream_features):
                    upstream_act.act.grad = t.zeros_like(upstream_act.act)
                    upstream_act.res.grad = t.zeros_like(upstream_act.res)
                    downstream_act[downstream_feat].backward(retain_graph=True)
                    fs_grads[i] += SparseAct(
                        act=upstream_act.act.grad,
                        res=upstream_act.res.grad
                    )

            # get shapes
            d_downstream_contracted = t.tensor(hidden_states_clean[resid].act.size())
            d_downstream_contracted[-1] += 1
            d_downstream_contracted = d_downstream_contracted.prod()
            
            d_upstream_contracted = t.tensor(upstream_act.act.size())
            d_upstream_contracted[-1] += 1
            d_upstream_contracted = d_upstream_contracted.prod()

            max_effect = None
            for downstream_feat, fs_grad in zip(downstream_features, fs_grads):
                grad = fs_grad / steps
                delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
                flat_effect = (grad @ delta).to_tensor().flatten()

                effect_indices[downstream_feat] = flat_effect.nonzero().squeeze(-1)
                effect_values[downstream_feat] = flat_effect[effect_indices[downstream_feat]]
                tot_eff = effect_values[downstream_feat].sum()
                effect_values[downstream_feat] = effect_values[downstream_feat] / tot_eff

                if max_effect is None:
                    max_effect = effect / tot_eff
                else:
                    max_effect.act = t.where((effect.act / tot_eff).abs() > max_effect.act.abs(), effect.act / tot_eff, max_effect.act)
                    max_effect.resc = t.where((effect.resc / tot_eff).abs() > max_effect.resc.abs(), effect.resc / tot_eff, max_effect.resc)

            features_by_submod[prev_resid] = (max_effect.to_tensor().flatten().abs() > edge_threshold).nonzero().flatten().tolist()

            # converts the dictionary of indices to a tensor of indices
            effect_indices = t.tensor(
                [[downstream_feat for downstream_feat in downstream_features for _ in effect_indices[downstream_feat]],
                t.cat([effect_indices[downstream_feat] for downstream_feat in downstream_features], dim=0)]
            ).to(model.cfg.device)
            effect_values = t.cat([effect_values[downstream_feat] for downstream_feat in downstream_features], dim=0)

            RR_effect = t.sparse_coo_tensor(effect_indices, effect_values, (d_downstream_contracted, d_upstream_contracted))
    
        if layer > 0:
            nodes[f'resid_{layer-1}'] = max_effect
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect
        else:
            nodes['embed'] = max_effect
            edges['embed'][f'resid_0'] = RR_effect

    # rearrange weight matrices
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == 'y':
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
            else:
                bp, sp, fp = nodes[parent].act.shape
                assert bp == bc
                weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
            edges[child][parent] = weight_matrix
    
    if aggregation == 'sum':
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].mean(dim=0)
    
    elif aggregation == 'none':

        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return nodes, edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='simple_train',
                        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.")
    parser.add_argument('--num_examples', '-n', type=int, default=100,
                        help="The number of examples from the --dataset over which to average indirect effects.")
    parser.add_argument('--example_length', '-l', type=int, default=None,
                        help="The max length (if using sum aggregation) or exact length (if not aggregating) of examples.")
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m-deduped',
                        help="The Huggingface ID of the model you wish to test.")
    parser.add_argument("--dict_path", type=str, default="dictionaries/pythia-70m-deduped/",
                        help="Path to all dictionaries for your language model.")
    parser.add_argument('--d_model', type=int, default=512,
                        help="Hidden size of the language model.")
    parser.add_argument('--dict_id', type=str, default=10,
                        help="ID of the dictionaries. Use `id` to obtain circuits on neurons/heads directly.")
    parser.add_argument('--dict_size', type=int, default=32768,
                        help="The width of the dictionary encoder.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of examples to process at once when running circuit discovery.")
    parser.add_argument('--aggregation', type=str, default='sum',
                        help="Aggregation across token positions. Should be one of `sum` or `none`.")
    parser.add_argument('--node_threshold', type=float, default=0.2,
                        help="Indirect effect threshold for keeping circuit nodes.")
    parser.add_argument('--edge_threshold', type=float, default=0.02,
                        help="Indirect effect threshold for keeping edges.")
    parser.add_argument('--pen_thickness', type=float, default=1,
                        help="Scales the width of the edges in the circuit plot.")
    parser.add_argument('--nopair', default=False, action="store_true",
                        help="Use if your data does not contain contrastive (minimal) pairs.")
    parser.add_argument('--plot_circuit', default=False, action='store_true',
                        help="Plot the circuit after discovering it.")
    parser.add_argument('--nodes_only', default=False, action='store_true',
                        help="Only search for causally implicated features; do not draw edges.")
    parser.add_argument('--plot_only', action="store_true",
                        help="Do not run circuit discovery; just plot an existing circuit.")
    parser.add_argument("--circuit_dir", type=str, default="circuits/",
                        help="Directory to save/load circuits.")
    parser.add_argument("--plot_dir", type=str, default="circuits/figures/",
                        help="Directory to save figures.")
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()


    device = args.device

    model = LanguageModel(args.model, device_map=device, dispatch=True)

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]

    dictionaries = {}
    if args.dict_id == 'id':
        from dictionary_learning.dictionary import IdentityDict
        dictionaries[embed] = IdentityDict(args.d_model)
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = IdentityDict(args.d_model)
            dictionaries[mlps[i]] = IdentityDict(args.d_model)
            dictionaries[resids[i]] = IdentityDict(args.d_model)
    else:
        dictionaries[embed] = AutoEncoder.from_pretrained(
            f'{args.dict_path}/embed/{args.dict_id}_{args.dict_size}/ae.pt',
            device=device
        )
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/attn_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )
            dictionaries[mlps[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/mlp_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )
            dictionaries[resids[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/resid_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )
    
    if args.nopair:
        save_basename = os.path.splitext(os.path.basename(args.dataset))[0]
        examples = load_examples_nopair(args.dataset, args.num_examples, model, length=args.example_length)
    else:
        data_path = f"data/{args.dataset}.json"
        save_basename = args.dataset
        if args.aggregation == "sum":
            examples = load_examples(data_path, args.num_examples, model, pad_to_length=args.example_length)
        else:
            examples = load_examples(data_path, args.num_examples, model, length=args.example_length)
    
    batch_size = args.batch_size
    num_examples = min([args.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < args.num_examples: # warn the user
        print(f"Total number of examples is less than {args.num_examples}. Using {num_examples} examples instead.")

    if not args.plot_only:
        running_nodes = None
        running_edges = None

        for batch in tqdm(batches, desc="Batches"):
                
            clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
            clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

            if args.nopair:
                patch_inputs = None
                def metric_fn(model):
                    return (
                        -1 * t.gather(
                            t.nn.functional.log_softmax(model.embed_out.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)
                    )
            else:
                patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
                patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)
                def metric_fn(model):
                    return (
                        t.gather(model.embed_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                        t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                    )
            
            nodes, edges = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )

            if running_nodes is None:
                running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
                if not args.nodes_only: running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
            else:
                for k in nodes.keys():
                    if k != 'y':
                        running_nodes[k] += len(batch) * nodes[k].to('cpu')
                if not args.nodes_only:
                    for k in edges.keys():
                        for v in edges[k].keys():
                            running_edges[k][v] += len(batch) * edges[k][v].to('cpu')
            
            # memory cleanup
            del nodes, edges
            gc.collect()

        nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
        if not args.nodes_only: 
            edges = {k : {kk : 1/num_examples * v.to(device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}
        else: edges = None

        save_dict = {
            "examples" : examples,
            "nodes": nodes,
            "edges": edges
        }
        with open(f'{args.circuit_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt', 'wb') as outfile:
            t.save(save_dict, outfile)

    else:
        with open(f'{args.circuit_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt', 'rb') as infile:
            save_dict = t.load(infile)
        nodes = save_dict['nodes']
        edges = save_dict['edges']

    # feature annotations
    try:
        annotations = {}
        with open(f"annotations/{args.dict_id}_{args.dict_size}.jsonl", 'r') as annotations_data:
            for annotation_line in annotations_data:
                annotation = json.loads(annotation_line)
                annotations[annotation["Name"]] = annotation["Annotation"]
    except:
        annotations = None

    if args.aggregation == "none":
        example = model.tokenizer.batch_decode(examples[0]["clean_prefix"])[0]
        plot_circuit_posaligned(
            nodes, 
            edges,
            layers=len(model.gpt_neox.layers), 
            length=args.example_length,
            example_text=example,
            node_threshold=args.node_threshold, 
            edge_threshold=args.edge_threshold, 
            pen_thickness=args.pen_thickness, 
            annotations=annotations, 
            save_dir=f'{args.plot_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}'
        )
    else:
        plot_circuit(
            nodes, 
            edges, 
            layers=len(model.gpt_neox.layers), 
            node_threshold=args.node_threshold, 
            edge_threshold=args.edge_threshold, 
            pen_thickness=args.pen_thickness, 
            annotations=annotations, 
            save_dir=f'{args.plot_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}'
        )