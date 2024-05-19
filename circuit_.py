import argparse
import gc
import json
import math
import os
from collections import defaultdict

import torch as t
from einops import rearrange
from tqdm import tqdm

from activation_utils import SparseAct
from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from dictionary_learning import AutoEncoder
from loading_utils import load_examples, load_examples_nopair
from nnsight import LanguageModel

import time

tracer_kwargs = {'validate' : False, 'scan' : False}

###### utilities for dealing with sparse COO tensors ######
def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def prod(l):
    out = 1
    for x in l: out *= x
    return out

def sparse_flatten(x):
    x = x.coalesce()
    return t.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),)
    )

def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape)

def sparse_mean(x, dim):
    if isinstance(dim, int):
        return x.sum(dim=dim) / x.shape[dim]
    else:
        return x.sum(dim=dim) / prod(x.shape[d] for d in dim)

######## end sparse tensor utilities ########

def save_circuit(save_dir, nodes, edges, dataset_name, model_name, node_threshold, edge_threshold, num_examples):
    save_dict = {
        "nodes" : nodes,
        "edges" : edges
    }
    save_basename = f"{dataset_name}_{model_name}_node{node_threshold}_edge{edge_threshold}_n{num_examples}"
    with open(f'{save_dir}/{save_basename}.pt', 'wb') as outfile:
        t.save(save_dict, outfile)

def load_circuit(circuit_path):
    with open(circuit_path, 'rb') as infile:
        save_dict = t.load(infile)
    nodes = save_dict['nodes']
    edges = save_dict['edges']
    return nodes, edges

def get_circuit(
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
    # TODO : put back the nodes, and their "weight" is now the sum of their contributions, or the max
    #        lets do the max

    """
    TODO : for module - module, gather all act and patch, and when evaluating all - module, do
    1-gather all detached
    2-retain_grad and require_grad
    3-sum and give that to the module
    -> should be barely slower than current version, hopefully

    if QK dict, they are only linked to their respective attn head, not anything further.
    Get a dict for QKV to have the precise decomposition of the effect and then a full attn layer or attn head dict.
    """
    t_start = time.time()

    all_submods = [embed] + [submod for submod in resids]
    
    # dummy forward pass to get shapes of outputs
    is_tuple = {}
    with model.trace("_"), t.no_grad():
        for submodule in all_submods:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

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
    with model.trace(**tracer_kwargs) as tracer:
        metrics = []
        fs = []
        for step in range(steps):
            alpha = step / steps
            upstream_act = (1 - alpha) * clean_state + alpha * patch_state
            upstream_act.act.retain_grad()
            upstream_act.res.retain_grad()
            fs.append(upstream_act)
            with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                if is_tuple[last_layer]:
                    last_layer.output[0][:] = dictionary.decode(upstream_act.act) + upstream_act.res
                else:
                    last_layer.output = dictionary.decode(upstream_act.act) + upstream_act.res
                metrics.append(metric_fn(model, **metric_kwargs))
        metric = sum([m for m in metrics])
        metric.sum().backward(retain_graph=True)

    mean_grad = sum([f.act.grad for f in fs]) / steps
    mean_residual_grad = sum([f.res.grad for f in fs]) / steps
    grad = SparseAct(act=mean_grad, res=mean_residual_grad)
    delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
    effect = grad @ delta

    tot_eff = effect.to_tensor().sum()
    effect = effect / tot_eff

    n_layers = len(resids)

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

        t_end = time.time()
        print(f"Layer {layer} : {t_end - t_start} seconds")
        print(f"Now processing layer {layer} with {len(downstream_features)} features")
        print(downstream_features)
        t_start = time.time()

        if not features_by_submod[resid]: # handle empty list
            features_by_submod[prev_resid] = []
            RR_effect = t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device)
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
                
                with model.trace(clean, **tracer_kwargs):
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
                    downstream_act[downstream_feat].backward(retain_graph=True)
                    fs_grads[i] += SparseAct(
                        act=upstream_act.act.grad,
                        res=upstream_act.res.grad
                    )
                    upstream_act.act.grad = t.zeros_like(upstream_act.act)
                    upstream_act.res.grad = t.zeros_like(upstream_act.res)

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
            ).to(model.device)
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


def get_circuit_cluster(dataset,
                        model_name="EleutherAI/pythia-70m-deduped",
                        d_model=512,
                        dict_id=10,
                        dict_size=32768,
                        max_length=64,
                        max_examples=100,
                        batch_size=2,
                        node_threshold=0.1,
                        edge_threshold=0.01,
                        device="cuda:0",
                        dict_path="dictionaries/pythia-70m-deduped/",
                        dataset_name="cluster_circuit",
                        circuit_dir="circuits/",
                        plot_dir="circuits/figures/",
                        model=None,
                        dictionaries=None,):
    
    model = LanguageModel(model_name, device_map=device, dispatch=True)

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]
    dictionaries = {}
    dictionaries[embed] = AutoEncoder.from_pretrained(
        os.path.join(dict_path, f'embed/{dict_id}_{dict_size}/ae.pt'),
        device=device
    )
    for i in range(len(model.gpt_neox.layers)):
        dictionaries[attns[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'attn_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )
        dictionaries[mlps[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'mlp_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )
        dictionaries[resids[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'resid_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )

    examples = load_examples_nopair(dataset, max_examples, model, length=max_length)

    num_examples = min(len(examples), max_examples)
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < max_examples: # warn the user
        print(f"Total number of examples is less than {max_examples}. Using {num_examples} examples instead.")

    running_nodes = None
    running_edges = None

    for batch in tqdm(batches, desc="Batches"):
        clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
        clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

        patch_inputs = None
        def metric_fn(model):
            return (
                -1 * t.gather(
                    t.nn.functional.log_softmax(model.embed_out.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                ).squeeze(-1)
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
            aggregation="sum",
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        if running_nodes is None:
            running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
            running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
        else:
            for k in nodes.keys():
                if k != 'y':
                    running_nodes[k] += len(batch) * nodes[k].to('cpu')
            for k in edges.keys():
                for v in edges[k].keys():
                    running_edges[k][v] += len(batch) * edges[k][v].to('cpu')
        
        # memory cleanup
        del nodes, edges
        gc.collect()

    nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
    edges = {k : {kk : 1/num_examples * v.to(device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}

    save_dict = {
        "examples" : examples,
        "nodes": nodes,
        "edges": edges
    }
    save_basename = f"{dataset_name}_dict{dict_id}_node{node_threshold}_edge{edge_threshold}_n{num_examples}_aggsum"
    with open(f'{circuit_dir}/{save_basename}.pt', 'wb') as outfile:
        t.save(save_dict, outfile)

    nodes = save_dict['nodes']
    edges = save_dict['edges']

    # feature annotations
    try:
        annotations = {}
        with open(f'annotations/{dict_id}_{dict_size}.jsonl', 'r') as f:
            for line in f:
                line = json.loads(line)
                annotations[line['Name']] = line['Annotation']
    except:
        annotations = None

    plot_circuit(
        nodes, 
        edges, 
        layers=len(model.gpt_neox.layers), 
        node_threshold=node_threshold, 
        edge_threshold=edge_threshold, 
        pen_thickness=1, 
        annotations=annotations, 
        save_dir=os.path.join(plot_dir, save_basename))


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