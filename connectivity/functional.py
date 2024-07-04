import torch
from welford_torch import OnlineCovariance

import tqdm

from connectivity.attribution import patching_effect, head_attribution

from utils.dictionary import IdentityDict
from utils.activation import get_hidden_states
from utils.metric_fns import metric_fn_logit

# TODO : module only. Is it as simple as modifying the dictionaries ? Probably not, attribution has to do grad dot diff.

# TODO : zero ablation vs mean ablation vs patching ?
def ROI_activation(
        clean,
        model,
        metric_fn,
        attns,
        mlps,
        patch=None,
        dictionaries=None,
        metric_fn_kwargs={},
        activation=True,
        attribution=True,
        aggregation="sum",
        ablation_fn=lambda x : torch.zeros_like(x) if isinstance(x, torch.Tensor) else x.zeros_like(),
        steps=10,
):
    """
    Get the activation and/or attribution vector of all attention head and MLP outputs.

    clean : torch.Tensor
        The input to the model.
    model : LanguageModel
        An nnsight model.
    metric_fn : callable
        The metric function to evaluate the model.
    mlps : list of nn.Module
        The MLP modules of the model.
    attns : list of list of attention heads
        The attention heads of the model.
    patch : torch.Tensor
        Optional. The patch for attribution.
    dictionaries :
        Optional. if provided, treated as an AutoEncoder with a .encode() and .decode() method.
        In this context, intended to be used for changes of basis, but can also work for SAE, gated SAE or any non linear dictionary.
        They can be used to perform changes of basis.
    metric_fn_kwargs : dict
        Optional. Additional arguments to pass to the metric function.
    activation : bool
        Whether to compute the activation vector.
    attribution : bool
        Whether to compute the attribution vector.
    aggregation : str
        The aggregation method to contract across sequence length.
    ablation_fn : str
        The ablation method to use. Default to mean ablation. Also available : zero ablation.
    steps : int
        Number of steps to compute the attributions (precision of Integrated Gradients).
    """
    # TODO : do QKV intermediate, or just head output ?
    # TODO : SVD dict can be useful. V^T V = I, but mean !=0, so at least mean is useful. Whiten dict would change
    #        something though, but not sure they are a good idea.

    z_hooks = [attn.hook_z for attn in attns]

    if (not attribution) and (not activation):
        raise ValueError("You must set either attribution or activation to True")

    all_submods = [submod for layer_submods in zip(mlps, z_hooks) for submod in layer_submods]

    if dictionaries is None:
        dictionaries = {submod: IdentityDict() for submod in all_submods}

    
    is_tuple = {}
    with model.trace("_"), torch.no_grad():
        for submodule in all_submods:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states = get_hidden_states(model, all_submods, dictionaries, is_tuple, clean, reconstruction_error=False)

    if aggregation == 'sum':
        n_s_fct = lambda n: n.sum(dim=1)
    elif aggregation == 'max':
        n_s_fct = lambda n: n.amax(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    if activation:
        # first flatten attn head outputs from (b, s, h, d) to (b, s, h) by taking the norm over the last dimension
        for k, v in hidden_states.items():
            if k in z_hooks:
                b, s, h, d = v.act.shape
                hidden_states[k] = v.act.norm(dim=-1) # (b, s, h)
                print(hidden_states[k].shape)
                raise NotImplementedError("Check the shape here.")
            else:
                b, s, d = v.act.shape
                hidden_states[k] = v.act.norm(dim=-1).view(b, s, 1) # (b, s, 1)

        # concatenate all hidden states in one big tensor of shape (batch_size, n_tokens, n_features_tot)
        hidden_concat = torch.cat([v for v in hidden_states.values()], dim=-1)
        hidden_concat = n_s_fct(hidden_concat)

        if not attribution:
            return {'act' : hidden_concat}
        
    if attribution:
        effects = head_attribution(
            clean, patch, model,
            attns, mlps,
            dictionaries, metric_fn,
            steps=steps, metric_kwargs=metric_fn_kwargs,
            ablation_fn=ablation_fn,
        )

        # effects with key in other are (b, s), with key in attns are (b, s, h)

        attr_concat = torch.cat([v for v in effects.values()], dim=-1)
        attr_concat = n_s_fct(attr_concat)

        if activation:
            return {'act' : hidden_concat, 'attr' : attr_concat}
        else:
            return {'attr' : attr_concat}
        
def neuron_activation(
        clean,
        model,
        metric_fn,
        embed,
        resids=None,
        attns=None,
        mlps=None,
        patch=None,
        dictionaries=None,
        metric_fn_kwargs={},
        use_resid=False,
        activation=True,
        attribution=True,
        aggregation="sum",
        ablation_fn=lambda x : torch.zeros_like(x) if isinstance(x, torch.Tensor) else x.zeros_like(),
        steps=10,
        discard_reconstruction_error=False,
    ):
    """
    Get the activation and/or attribution vector of all neurons in the dictionaries of interest.
    Used for functional connectivity analysis to get the covariance matrix of the activations.

    Returns a dict with key in "attribution" and "activation" and the corresponding vectorized attr/act values of
    all dicts, with shape (batch_size, n_nodes).
    """

    if (not attribution) and (not activation):
        raise ValueError("You must set either attribution or activation to True")

    if use_resid:
        all_submods = [embed] + [submod for submod in resids]
    else:
        all_submods = [embed] + [submod for layer_submods in zip(mlps, attns) for submod in layer_submods]

    if dictionaries is None:
        dictionaries = {submod: IdentityDict() for submod in all_submods}
    
    # dummy forward pass to get shapes of outputs
    is_tuple = {}
    with model.trace("_"), torch.no_grad():
        for submodule in all_submods:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    # get encoding and reconstruction errors for clean and patch
    hidden_states = get_hidden_states(model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=clean)

    if aggregation == 'sum':
        n_s_fct = lambda n: n.sum(dim=1)
    elif aggregation == 'max':
        n_s_fct = lambda n: n.amax(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    if activation:
        # concatenate all hidden states in one big tensor of shape (batch_size, n_tokens, n_features_tot)
        if discard_reconstruction_error:
            hidden_concat = torch.cat([v.act for v in hidden_states.values()], dim=-1)
        else:
            hidden_concat = torch.cat([torch.cat([v.act, v.res.norm(dim=-1, keepdim=True)], dim=-1) for v in hidden_states.values()], dim=-1)

        # aggregate over sequence position
        hidden_concat = n_s_fct(hidden_concat)

        if not attribution:
            return {'act' : hidden_concat}
    
    if attribution:
        """
        Compute effect of all layers on the metric.
        """
        effects, _, _, _ = patching_effect(
            clean,
            patch,
            model,
            all_submods,
            dictionaries,
            metric_fn,
            method='ig',
            steps=steps,
            metric_kwargs=metric_fn_kwargs,
            ablation_fn=ablation_fn,
        )

        if discard_reconstruction_error:
            attr_concat = torch.cat([v.act for v in effects.values()], dim=-1)
        else:
            attr_concat = torch.cat([torch.cat([v.act, v.res.norm(dim=-1, keepdim=True)], dim=-1) for v in effects.values()], dim=-1)

        # aggregate over sequence position
        attr_concat = n_s_fct(attr_concat)

        if activation:
            return {'act' : hidden_concat, 'attr' : attr_concat}
        else:
            return {'attr' : attr_concat}

def generate_cov(
    data_buffer,
    model,
    embed,
    resids,
    attns,
    mlps,
    dictionaries=None,
    get_act=True,
    get_attr=True,
    neuron=True,
    ROI=False,
    batch_size=100,
    use_resid=False,
    n_batches=1000,
    aggregation='sum',
    steps=10,
    discard_reconstruction_error=True,
):
    """
    Get the covariance matrix of the activations of the model for some specified
    module (default : attn and MLP outputs) and dictionary (default : identity, or neurons)
    """
    if not sum([neuron, ROI]) == 1:
        raise ValueError("You must set exactly one of neuron or ROI to True")

    available_gpus = torch.cuda.device_count()
    device = torch.device(f'cuda:{0}') if available_gpus > 0 else torch.device('cpu')

    cov = {}
    if get_attr:
        cov['attr'] = OnlineCovariance()
    if get_act:
        cov['act'] = OnlineCovariance()
    
    i = 0
    act_kwargs = {
        'clean': None,
        'model': model,
        'metric_fn': metric_fn_logit,
        'attns': attns,
        'mlps': mlps,
        'patch': None,
        'dictionaries': dictionaries,
        'metric_fn_kwargs': {'trg': None},
        'activation': get_act,
        'attribution': get_attr,
        'aggregation': aggregation,
        'steps': steps,
    }
    if neuron:
        act_kwargs['embed'] = embed
        act_kwargs['resids'] = resids
        act_kwargs['use_resid'] = use_resid
        act_kwargs['discard_reconstruction_error'] = discard_reconstruction_error
    
    # there is an annoying bug with multiprocessing and nnsight. Don't use multiprocessing for now.
    DEBUG = True
    if DEBUG:
        for tokens, trg_idx, trg in tqdm(data_buffer, total=n_batches):
            act_kwargs['clean'] = tokens.to(device)
            act_kwargs['metric_fn_kwargs']['trg'] = (trg_idx.to(device), trg.to(device))
            if neuron:
                act = neuron_activation(**act_kwargs)
            elif ROI:
                act = ROI_activation(**act_kwargs)
            for key in act:
                cov[key].add_all(act[key].to(device))
            i += 1
            if i == n_batches:
                break
    else :
        raise NotImplementedError("Multiprocessing not implemented yet.")
        # with ProcessPoolExecutor(max_workers=available_gpus) as executor:
        #     futures = []
        #     for gpu in range(available_gpus):
        #         tokens, trg_idx, trg = next(buffer)
        #         args_dict_per_device[gpu]['clean'] = tokens.to(f'cuda:{gpu}')
        #         args_dict_per_device[gpu]['metric_fn_kwargs']['trg'] = (trg_idx.to(f'cuda:{gpu}'), trg.to(f'cuda:{gpu}'))
        #         futures.append(executor.submit(run_task, args_dict_per_device[gpu]))
        #     finished = False
        #     while not finished:
        #         for future in tqdm(as_completed(futures), total=None):
        #             # get the result
        #             act = future.result()
        #             print("Got act from future : ")
        #             for key in act:
        #                 print(key, act[key].shape)
        #                 current_gpu = act[key].device.index
        #                 cov[key].add_all(act[key].to(device))
        #             i += 1
        #             # remove the current future and add a new one
        #             futures.remove(future)
        #             if i == n_batches:
        #                 finished = True
        #                 break

        #             tokens, trg_idx, trg = next(buffer)
        #             args_dict_per_device[current_gpu]['clean'] = tokens.to(f'cuda:{current_gpu}')
        #             args_dict_per_device[current_gpu]['metric_fn_kwargs']['trg'] = (trg_idx.to(f'cuda:{current_gpu}'), trg.to(f'cuda:{current_gpu}'))
        #             futures.append(executor.submit(run_task, args_dict_per_device[current_gpu]))
            
        #     # kill all futures left
        #     for future in futures:
        #         future.cancel()

    return cov
