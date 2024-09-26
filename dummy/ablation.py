from nnsight import LanguageModel
import torch as t
from argparse import ArgumentParser
from activation_utils import SparseAct

def run_with_ablations(
        clean, # clean inputs
        patch, # patch inputs for use in computing ablation values
        model, # a nnsight LanguageModel
        submodules, # list of submodules 
        dictionaries, # dictionaries[submodule] is an autoencoder for submodule's output
        nodes, # nodes[submodule] is a boolean SparseAct with True for the nodes to keep (or ablate if complement is True)
        metric_fn, # metric_fn(model, **metric_kwargs) -> t.Tensor
        metric_kwargs=dict(),
        complement=False, # if True, then use the complement of nodes
        ablation_fn=lambda x: x.mean(dim=0).expand_as(x), # what to do to the patch hidden states to produce values for ablation, default mean ablation
        handle_errors='default', # or 'remove' to zero ablate all; 'keep' to keep all
    ):

    if patch is None: patch = clean
    patch_states = {}
    with model.trace(patch), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if type(x.shape) == tuple:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    patch_states = {k : ablation_fn(v.value) for k, v in patch_states.items()}


    with model.trace(clean), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            submod_nodes = nodes[submodule].clone()
            x = submodule.output
            is_tuple = type(x.shape) == tuple
            if is_tuple:
                x = x[0]
            f = dictionary.encode(x)
            res = x - dictionary(x)

            # ablate features
            if complement: submod_nodes = ~submod_nodes
            submod_nodes.resc = submod_nodes.resc.expand(*submod_nodes.resc.shape[:-1], res.shape[-1])
            if handle_errors == 'remove':
                submod_nodes.resc = t.zeros_like(submod_nodes.resc).to(t.bool)
            if handle_errors == 'keep':
                submod_nodes.resc = t.ones_like(submod_nodes.resc).to(t.bool)

            f[...,~submod_nodes.act] = patch_states[submodule].act[...,~submod_nodes.act]
            res[...,~submod_nodes.resc] = patch_states[submodule].res[...,~submod_nodes.resc]
            
            if is_tuple:
                submodule.output[0][:] = dictionary.decode(f) + res
            else:
                submodule.output = dictionary.decode(f) + res

        metric = metric_fn(model, **metric_kwargs).save()
    return metric.value
