import torch

from nnsight import LanguageModel
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from utils.dictionary import IdentityDict, LinearDictionary, AutoEncoder

def load_model_and_modules(device, unified=True, model_name="EleutherAI/pythia-70m-deduped"):
    if unified:
        model = UnifiedTransformer(
            model_name,
            device=device,
            processing=False,
        )
        model.device = model.cfg.device
        model.tokenizer.padding_side = 'left'

        embed = model.embed

        resids = []
        attns = []
        mlps = []
        for layer in range(len(model.blocks)):
            resids.append(model.blocks[layer])
            attns.append(model.blocks[layer].attn)
            mlps.append(model.blocks[layer].mlp)

        submod_names = {
            model.embed : 'embed'
        }
        for i in range(len(model.blocks)):
            submod_names[model.blocks[i].attn] = f'attn_{i}'
            submod_names[model.blocks[i].mlp] = f'mlp_{i}'
            submod_names[model.blocks[i]] = f'resid_{i}'

        return model, embed, resids, attns, mlps, submod_names
    
    else:
        if model_name != "EleutherAI/pythia-70m-deduped":
            raise NotImplementedError("Only EleutherAI/pythia-70m-deduped is supported for non-unified models.")
        model = LanguageModel(
            model_name,
            device_map=device,
            dispatch=True,
        )

        embed = model.gpt_neox.embed_in

        resids = []
        attns = []
        mlps = []
        for layer in range(len(model.gpt_neox.layers)):
            resids.append(model.gpt_neox.layers[layer])
            attns.append(model.gpt_neox.layers[layer].attention)
            mlps.append(model.gpt_neox.layers[layer].mlp)

        submod_names = {
            model.gpt_neox.embed_in : 'embed'
        }
        for i in range(len(model.gpt_neox.layers)):
            submod_names[model.gpt_neox.layers[i].attention] = f'attn_{i}'
            submod_names[model.gpt_neox.layers[i].mlp] = f'mlp_{i}'
            submod_names[model.gpt_neox.layers[i]] = f'resid_{i}'

        return model, embed, resids, attns, mlps, submod_names

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
    path='/scratch/pyllm/dhimoila/',
    unified=True
):
    if white:
        raise NotImplementedError("Whitening is not implemented yet.")
    dictionaries = {}

    d_model = 512
    dict_size = 32768 if not svd else 512

    if idd:
        dictionaries[model_embed] = IdentityDict(d_model)
        for layer in range(len(model.gpt_neox.layers if not unified else model.blocks)):
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

    for layer in range(len(model.gpt_neox.layers if not unified else model.blocks)):
        
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
