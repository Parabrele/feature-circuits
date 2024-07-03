import torch

from utils.activation import get_hidden_states, SparseAct

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
        dictionaries,
        mod2name,
        clean,
        patch,
        graph,
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
    dictionaries : dict
        dict [submodule] -> SAE
    name_dict : dict
        dict [submodule] -> str
    clean : str, list of str or tensor (batch, seq_len)
        the input to the model
    patch : None, str, list of str or tensor (batch, seq_len)
        the counterfactual input to the model to ablate edges
    graph : edges
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
    graph = reorder_mask(graph)

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
    patch_states = get_hidden_states(model, submodules, dictionaries, is_tuple, patch)
    patch_states = {k : ablation_fn(v).clone() for k, v in patch_states.items()}

    # forward through the model by computing each node as described by the graph and not as the original model does

    # For each downstream module, get it's potentially alive features (heuristic to not compute one forward pass per node
    # as there are too many of them) by reachability from previously alive ones
    # Then, for each of these features, get it's masked input, and compute a forward pass to get this particular feature.
    # This gives the new state for this downstream output.

    # with model.trace(clean):
    hidden_states = {}
    for downstream in submodules:
        # get downstream dict, output, ...
        downstream_dict = dictionaries[downstream]
        down_name = mod2name[downstream]
        print(f"Computing {down_name}")
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

        print("Got x_hat and f")

        # if downstream is embed, there is no upstream and the result stays unchanged
        if down_name == 'embed' or downstream == submodules[0]:
            print("Embed or first layer")
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

        for up_name in graph[down_name]:
            upstream = name2mod[up_name]
            mask = graph[down_name][up_name] # shape (f_down + 1, f_up + 1)

            upstream_hidden = hidden_states[upstream].act # shape (batch, seq_len, f_up)
            # reduce to (f_up) by maxing over batch and seq_len (should only have positive entries, but a .abs() can't hurt)
            upstream_hidden = upstream_hidden.abs().amax(dim=(0, 1)) # shape (f_up)
            up_nz = torch.cat([upstream_hidden > 0, torch.tensor([True], device=f.device)]) # shape (f_up + 1). Always keep the res feature alive
            
            print("Number of potentially alive features upstream : ", up_nz.sum().item())
            compiled_loop_pot_ali(mask.indices(), potentially_alive, up_nz)

        potentially_alive = potentially_alive.nonzero().squeeze(1)
        print("Number of potentially alive features downstream : ", potentially_alive.size(0))

        f[...] = patch_states[downstream].act # shape (batch, seq_len, f_down)
        for f_ in potentially_alive:
            edge_ablated_input = torch.zeros(tuple(input_shape)).to(f.device)
            for up_name in graph[down_name]:
                upstream = name2mod[up_name]
                upstream_dict = dictionaries[upstream]
                
                mask = graph[down_name][up_name][f_].to_dense() # shape (f_up + 1)

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
            last_layer.output[0][:] = dictionaries[last_layer].decode(hidden_states[last_layer].act) + hidden_states[last_layer].res
        else:
            last_layer.output = dictionaries[last_layer].decode(hidden_states[last_layer].act) + hidden_states[last_layer].res
        
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