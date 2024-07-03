import torch

from nnsight import LanguageModel
from utils.dictionary import IdentityDict, LinearDictionary, AutoEncoder

def load_model_and_modules(device): 
    pythia70m = LanguageModel(
        "EleutherAI/pythia-70m-deduped",
        device_map=device,
        dispatch=True,
    )

    pythia70m_embed = pythia70m.gpt_neox.embed_in

    pythia70m_resids = []
    pythia70m_attns = []
    pythia70m_mlps = []
    for layer in range(len(pythia70m.gpt_neox.layers)):
        pythia70m_resids.append(pythia70m.gpt_neox.layers[layer])
        pythia70m_attns.append(pythia70m.gpt_neox.layers[layer].attention)
        pythia70m_mlps.append(pythia70m.gpt_neox.layers[layer].mlp)

    submod_names = {
        pythia70m.gpt_neox.embed_in : 'embed'
    }
    for i in range(len(pythia70m.gpt_neox.layers)):
        submod_names[pythia70m.gpt_neox.layers[i].attention] = f'attn_{i}'
        submod_names[pythia70m.gpt_neox.layers[i].mlp] = f'mlp_{i}'
        submod_names[pythia70m.gpt_neox.layers[i]] = f'resid_{i}'

    return pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names

def load_saes(
    model,
    model_embed,
    model_resids,
    model_attns,
    model_mlps,
    idd=False,
    svd=False,
    white=False,
    device='cpu',
    path='/scratch/pyllm/dhimoila/'
):
    if white:
        raise NotImplementedError("Whitening is not implemented yet.")
    dictionaries = {}

    d_model = 512
    dict_size = 32768 if not svd else 512

    if idd:
        dictionaries[model_embed] = IdentityDict(d_model)
        for layer in range(len(model.gpt_neox.layers)):
            dictionaries[model_resids[layer]] = IdentityDict(d_model)
            dictionaries[model_attns[layer]] = IdentityDict(d_model)
            dictionaries[model_mlps[layer]] = IdentityDict(d_model)

        return dictionaries
    
    path = path + "dictionaires/pythia-70m-deduped/"# + ("SVDdicts/" if svd else "")

    if not svd:
        ae = AutoEncoder(d_model, dict_size).to(device)
        ae.load_state_dict(torch.load(path + f"embed/ae.pt", map_location=device))
        dictionaries[model_embed] = ae
    else:
        d = torch.load(path + f"embed/cov.pt", map_location=device)
        mean = d['mean']
        cov = d['cov']
        U, S, V = torch.svd(cov)
        dictionaries[model_embed] = LinearDictionary(d_model, dict_size)
        dictionaries[model_embed].E = V.T
        dictionaries[model_embed].D = V
        dictionaries[model_embed].bias = mean

    for layer in range(len(model.gpt_neox.layers)):
        
        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"resid_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[model_resids[layer]] = ae
        else:
            d = torch.load(path + f"resid_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov) # cov is symmetric so U = V
            dictionaries[model_resids[layer]] = LinearDictionary(d_model, dict_size)
            dictionaries[model_resids[layer]].E = V.T
            dictionaries[model_resids[layer]].D = V
            dictionaries[model_resids[layer]].bias = mean
        
        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"attn_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[model_attns[layer]] = ae
        else:
            d = torch.load(path + f"attn_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov)
            dictionaries[model_attns[layer]] = LinearDictionary(d_model, dict_size)
            dictionaries[model_attns[layer]].E = V.T # This will perform the operation x @ E.T = x @ V, but V is in it's transposed form
            dictionaries[model_attns[layer]].D = V
            dictionaries[model_attns[layer]].bias = mean

        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"mlp_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[model_mlps[layer]] = ae
        else:
            d = torch.load(path + f"mlp_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov)
            dictionaries[model_mlps[layer]] = LinearDictionary(d_model, dict_size)
            dictionaries[model_mlps[layer]].E = V.T
            dictionaries[model_mlps[layer]].D = V
            dictionaries[model_mlps[layer]].bias = mean
    
    return dictionaries