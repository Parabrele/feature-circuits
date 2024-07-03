import torch
from welford_torch import OnlineCovariance

import tqdm

from connectivity.attribution import patching_effect

from data.wikipedia import get_buffer

from utils.activation import get_hidden_states
from utils.metric_fns import metric_fn_logit
from utils.experiments_setup import load_model_and_modules, load_saes

# TODO : module only. Is it as simple as modifying the dictionaries ? Probably not, attribution has to do grad dot diff.

# TODO : zero ablation vs mean ablation vs patching ?
def ROI_activation(
        clean,
        model,
        metric_fn,
        embed,
        attns,
        mlps,
        patch=None,
        dictionaries=None,
        metric_fn_kwargs={},
        activation=True,
        attribution=True,
        aggregation="sum",
        ablation_fn="mean",
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
    embed, mlps : nn.Module and list of nn.Module
        The embedding and MLP modules of the model.
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
    if ablation_fn == 'mean':
        ablation_fn = lambda x: x.mean(dim=(0, 1)).expand_as(x)
    elif ablation_fn == 'zero':
        ablation_fn = lambda x: torch.zeros_like(x)
    elif ablation_fn == 'id':
        ablation_fn = lambda x: x
    elif callable(ablation_fn):
        pass
    else:
        raise ValueError(f"Unknown default ablation function : {ablation_fn}")
    
    pass # TODO

def neuron_activation(
        clean,
        model,
        dictionaries,
        metric_fn,
        embed,
        resids=None,
        attns=None,
        mlps=None,
        metric_fn_kwargs={},
        use_resid=False,
        activation=True,
        attribution=True,
        aggregation="sum",
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
            None,
            model,
            all_submods,
            dictionaries,
            metric_fn,
            method='ig',
            steps=steps,
            metric_kwargs=metric_fn_kwargs
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
    dicts="identity",
    get_act=True,
    get_attr=True,
    batch_size=100,
    use_resid=False,
    n_batches=1000,
    model="pythia70m-deduped",
    dataset='wikipedia',
    aggregation='sum',
    steps=10,
):
    """
    Get the covariance matrix of the activations of the model for some specified
    module (default : attn and MLP outputs) and dictionary (default : identity, or neurons)
    """
    available_dicts = ["identity", "svd", "whiten", "SAE"]
    assert dicts in available_dicts, f"dicts must be one of {available_dicts}"

    available_gpus = torch.cuda.device_count()
    device = torch.device(f'cuda:{0}') if available_gpus > 0 else torch.device('cpu')

    if model == "pythia70m-deduped":
        pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules(device)
        dictionaries = load_saes(
            pythia70m,
            pythia70m_embed,
            pythia70m_resids,
            pythia70m_attns,
            pythia70m_mlps,
            device=device,
            idd=dicts=="identity",
            svd=dicts=="svd",
            white=dicts=="whiten",
        )
    else:
        raise NotImplementedError(f"Model {model} not supported yet.")

    if dataset == 'wikipedia':
        buffer = get_buffer(pythia70m, batch_size, device)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented yet.")

    cov = {}
    if get_attr:
        cov['attr'] = OnlineCovariance()
    if get_act:
        cov['act'] = OnlineCovariance()
    
    i = 0
    act_kwargs = {
        'clean': None,
        'model': pythia70m,
        'dictionaries': dictionaries,
        'submod_names': submod_names,
        'metric_fn': metric_fn_logit,
        'embed': pythia70m_embed,
        'resids': pythia70m_resids,
        'attns': pythia70m_attns,
        'mlps': pythia70m_mlps,
        'metric_fn_kwargs': {'trg': None},
        'use_resid': use_resid,
        'activation': get_act,
        'attribution': get_attr,
        'aggregation': aggregation,
        'steps': steps,
        'discard_reconstruction_error': (not (dicts=="SAE"))
    }
    
    # there is an annoying bug with multiprocessing and nnsight.
    DEBUG = True
    if DEBUG:
        # don't use multiprocessing
        for tokens, trg_idx, trg in tqdm(buffer, total=n_batches):
            act_kwargs['clean'] = tokens.to(device)
            act_kwargs['metric_fn_kwargs']['trg'] = (trg_idx.to(device), trg.to(device))
            act = neuron_activation(**act_kwargs)
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
