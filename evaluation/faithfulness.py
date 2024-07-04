import torch

from ablation.edge_ablation import run_graph

from utils.activation import SparseAct
from utils.graph_utils import get_mask, prune, get_n_nodes, get_n_edges, get_density

# TODO : instead of clean and patch, give the buffer, to not re threshold for each batch.
# TODO : separate this function into one that computes metrics given a mask graph, and
#        one that gather all evaluations, not only faithfulness
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
        get_graph_info=True,
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
    print("Printing clean :")
    print(type(clean))
    print(clean)
    with model.trace(clean):
        clean_logits = model.output[0][torch.arange(metric_fn_kwargs['trg'][0].numel()), metric_fn_kwargs['trg'][0]].save()
        if isinstance(metric_fn, dict):
            metric = {}
            for name, fn in metric_fn.items():
                if name == "KL":
                    metric[name] = fn(model, clean_logits=clean_logits, **metric_fn_kwargs).save()
                else:
                    metric[name] = fn(model, **metric_fn_kwargs).save()
        else:
            metric = metric_fn(model, **metric_fn_kwargs).save()
    
    print("Metric is : ", type(metric))
    if isinstance(metric, dict):
        results['complete'] = {}
        print("Metric is a dict")
        for name, value in metric.items():
            print(f"Metric {name} : {value.value.mean().item()}")
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
    for i, threshold in enumerate(thresholds):
        print(f"Threshold {i+1}/{len(thresholds)} : {threshold}")
        results[threshold] = {}

        mask = get_mask(circuit, threshold)
        pruned = prune(mask)

        if get_graph_info:
            results[threshold]['n_nodes'] = get_n_nodes(pruned)
            results[threshold]['n_edges'] = get_n_edges(pruned)
            results[threshold]['avg_deg'] = results[threshold]['n_edges'] / (results[threshold]['n_nodes'] if results[threshold]['n_nodes'] > 0 else 1)
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
                if "MRR" in name or "acc" in name:
                    results[threshold]['faithfulness'][name] = threshold_result[name] / results['complete'][name]
                elif name == "KL":
                    results[threshold]['faithfulness'][name] = threshold_result[name]
                else:
                    results[threshold]['faithfulness'][name] = (threshold_result[name] - empty[name]) / (results['complete'][name] - empty[name])
        else:
            results[threshold]['faithfulness'] = (threshold_result - empty) / (metric - empty)
    
    return results
