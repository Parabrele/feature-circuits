"""
python SBM.py --generate_cov -id -act -attr -nb 1000 -bs 100 -path /scratch/pyllm/dhimoila/output/SBM/id/ &

python SBM.py --fit_cov -id -act -attr -path /scratch/pyllm/dhimoila/output/SBM/id/ &
"""

##########
# Parsing arguments
##########

print("Parsing arguments and importing.")

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--generate_cov", "-gc", action="store_true", help="Generate the covariance matrices of activations")
parser.add_argument("--fit_cov", "-fc", action="store_true", help="Fit a SBM to the covariance matrices of activations")

parser.add_argument("--identity_dict", "-id", action="store_true", help="Use identity dictionaries instead of SAEs")
parser.add_argument("--SVD_dict", "-svd", action="store_true", help="Use SVD dictionaries instead of SAEs")
parser.add_argument("--White_dict", "-white", action="store_true", help="Use whitening space as dictionaries instead of SAEs")

parser.add_argument("--activation", "-act", action="store_true", help="Compute activations")
parser.add_argument("--attribution", "-attr", action="store_true", help="Compute attributions")
parser.add_argument("--use_resid", "-resid", action="store_true", help="Use residual stream nodes instead of modules.")

parser.add_argument("--n_batches", "-nb", type=int, default=1000, help="Number of batches to process.")
parser.add_argument("--batch_size", "-bs", type=int, default=1, help="Number of examples to process in one go.")
parser.add_argument("--steps", type=int, default=10, help="Number of steps to compute the attributions (precision of Integrated Gradients).")

parser.add_argument("--aggregation", "-agg", type=str, default="sum", help="Aggregation method to contract across sequence length.")

parser.add_argument("--node_threshold", "-nt", type=float, default=0.)
parser.add_argument("--edge_threshold", "-et", type=float, default=0.1)

parser.add_argument("--ctx_len", "-cl", type=int, default=16, help="Maximum sequence lenght of example sequences")

parser.add_argument("--save_path", "-path", type=str, default='/scratch/pyllm/dhimoila/output/', help="Path to save and load the outputs.")

# There is a strange recursive call error with nnsight when using multiprocessing...
DEBUG = True

args = parser.parse_args()

idd = args.identity_dict
svd = args.SVD_dict
white = args.White_dict

if white:
    raise NotImplementedError("Whitening is not implemented yet.")

get_attr = args.attribution
get_act = args.activation

n_batches = args.n_batches
batch_size = args.batch_size
save_path = args.save_path

edge_threshold = args.edge_threshold
node_threshold = args.node_threshold

##########
# Imports
##########

import graph_tool.all as gt

import os

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

multiprocessing.set_start_method('spawn', force=True)

import torch
from nnsight import LanguageModel
from datasets import load_dataset

from transformers import logging
logging.set_verbosity_error()

from tqdm import tqdm

from welford_torch import OnlineCovariance

from dictionary_learning import AutoEncoder
from dictionary_learning.dictionary import IdentityDict, LinearDictionary
from buffer import TokenBuffer

from circuit import get_activation

print("Done importing.")

##########
# Functions
##########

def metric_fn_logit(model, trg=None):
    """
    default : return the logit
    """
    if trg is None:
        raise ValueError("trg must be provided")
    return model.embed_out.output[torch.arange(trg[0].numel()), trg[0], trg[1]]

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
    device,
):
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
    
    base = '/scratch/pyllm/dhimoila/'
    path = base + "dictionaires/pythia-70m-deduped/" + ("SVDdicts/" if svd else "")

    ae = AutoEncoder(d_model, dict_size).to(device)
    ae.load_state_dict(torch.load(path + f"embed/ae.pt", map_location=device))
    dictionaries[model_embed] = ae

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

def get_buffer(
    model,
    batch_size,
    device,
):
    dataset = load_dataset(
        "wikipedia",
        language="en",
        date="20240401",
        split="train",
        streaming=True,
        trust_remote_code=True
    ).shuffle()
    dataset = iter(dataset)

    buffer = TokenBuffer(
        dataset,
        model,
        n_ctxs=10,
        ctx_len=args.ctx_len,
        load_buffer_batch_size=10,
        return_batch_size=batch_size,
        device=device,
        max_number_of_yields=2**20,
        discard_bos=True
    )

    return buffer

def run_task(args_dict):
    print("Starting task...")
    return get_activation(**args_dict)

def generate_cov():
    available_gpus = torch.cuda.device_count()
    device = torch.device(f'cuda:{0}') if available_gpus > 0 else torch.device('cpu')

    pythia70ms = []
    pythia70m_embeds = []
    pythia70m_residss = []
    pythia70m_attnss = []
    pythia70m_mlpss = []
    submod_namess = []
    dictionariess = []

    for i in range(available_gpus):
        current_device = torch.device(f'cuda:{i}')
        pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules(current_device)

        dictionaries = load_saes(
            pythia70m,
            pythia70m_embed,
            pythia70m_resids,
            pythia70m_attns,
            pythia70m_mlps,
            current_device,
        )

        pythia70ms.append(pythia70m)
        pythia70m_embeds.append(pythia70m_embed)
        pythia70m_residss.append(pythia70m_resids)
        pythia70m_attnss.append(pythia70m_attns)
        pythia70m_mlpss.append(pythia70m_mlps)
        submod_namess.append(submod_names)
        dictionariess.append(dictionaries)

    buffer = get_buffer(pythia70m, batch_size, device)

    cov = {}
    if get_attr:
        cov['attr'] = OnlineCovariance()
    if get_act:
        cov['act'] = OnlineCovariance()
    
    i = 0
    args_dict_to_dupplicate = {
        'clean': [None] * available_gpus,
        'model': pythia70ms,
        'dictionaries': dictionariess,
        'submod_names': submod_namess,
        'metric_fn': [metric_fn_logit]*available_gpus,
        'embed': pythia70m_embeds,
        'resids': pythia70m_residss,
        'attns': pythia70m_attnss,
        'mlps': pythia70m_mlpss,
        'metric_fn_kwargs': [{'trg': None}] * available_gpus,
        'use_resid': [args.use_resid] * available_gpus,
        'activation': [get_act]*available_gpus,
        'attribution': [get_attr]*available_gpus,
        'aggregation': [args.aggregation]*available_gpus,
        'node_threshold': [args.node_threshold]*available_gpus,
        'steps': [args.steps]*available_gpus,
        'discard_reconstruction_error': [idd or svd or white] * available_gpus,
    }
    args_dict_per_device = [{k: v[i] for k, v in args_dict_to_dupplicate.items()} for i in range(available_gpus)]
    
    if DEBUG:
        # don't use multiprocessing
        gpu = 0
        for tokens, trg_idx, trg in tqdm(buffer, total=n_batches):
            args_dict_per_device[gpu]['clean'] = tokens.to(f'cuda:{gpu}')
            args_dict_per_device[gpu]['metric_fn_kwargs']['trg'] = (trg_idx.to(f'cuda:{gpu}'), trg.to(f'cuda:{gpu}'))
            act = get_activation(**args_dict_per_device[gpu])
            for key in act:
                cov[key].add_all(act[key].to(device))
            i += 1
            if i == n_batches:
                break
    else :
        with ProcessPoolExecutor(max_workers=available_gpus) as executor:
            futures = []
            for gpu in range(available_gpus):
                tokens, trg_idx, trg = next(buffer)
                args_dict_per_device[gpu]['clean'] = tokens.to(f'cuda:{gpu}')
                args_dict_per_device[gpu]['metric_fn_kwargs']['trg'] = (trg_idx.to(f'cuda:{gpu}'), trg.to(f'cuda:{gpu}'))
                futures.append(executor.submit(run_task, args_dict_per_device[gpu]))
            finished = False
            while not finished:
                for future in tqdm(as_completed(futures), total=None):
                    # get the result
                    act = future.result()
                    print("Got act from future : ")
                    for key in act:
                        print(key, act[key].shape)
                        current_gpu = act[key].device.index
                        cov[key].add_all(act[key].to(device))
                    i += 1
                    # remove the current future and add a new one
                    futures.remove(future)
                    if i == n_batches:
                        finished = True
                        break

                    tokens, trg_idx, trg = next(buffer)
                    args_dict_per_device[current_gpu]['clean'] = tokens.to(f'cuda:{current_gpu}')
                    args_dict_per_device[current_gpu]['metric_fn_kwargs']['trg'] = (trg_idx.to(f'cuda:{current_gpu}'), trg.to(f'cuda:{current_gpu}'))
                    futures.append(executor.submit(run_task, args_dict_per_device[current_gpu]))
            
            # kill all futures left
            for future in futures:
                future.cancel()

    return cov

def fit_SBM(cov):
    """
    cov : OnlineCovariance
    """
    mean = cov.mean.to('cpu').detach()
    cov = cov.cov.to('cpu').detach()
    #cov = cov / torch.sqrt(cov.diag().unsqueeze(0) * cov.diag().unsqueeze(1))

    print(cov.min(), cov.max(), cov.abs().mean())

    thresholds = torch.linspace(1, 10, 20)
    thresholds = [20]

    # # plt plot weights repartition
    # import matplotlib.pyplot as plt
    # w = cov.flatten()
    # w = w[w < 100]
    # w = w[w > -100]
    # plt.hist(w, bins=1000)
    # plt.yscale('log')
    # plt.savefig(save_path + "hist_cov.png")

    print(mean.shape, cov.shape)

    for threshold in thresholds:
        edges = torch.zeros_like(cov)
        edges[cov.abs() > threshold] = 1

        print("Threshold :", threshold)
        print("\tEdge density : ", edges.sum(), "/", edges.shape[0] * edges.shape[1], " = ", edges.sum() / (edges.shape[0] * edges.shape[1]))

        g = gt.Graph(directed=False)
        g.add_edge_list(torch.nonzero(edges).numpy())
        # print number of nodes and edges
        print("\tNumber of nodes :", g.num_vertices())
        print("\tNumber of edges :", g.num_edges())

        # # SBM :
        # print("\tSBM...", end="")
        # state = gt.minimize_blockmodel_dl(g)
        # print("Done.")
        # print("\tDrawing...", end="")
        # state.draw(output=save_path + f"SBM_{threshold}.svg")
        # print("Done.")

        # nested SBM :
        print("\tNested SBM...", end="")
        state = gt.minimize_nested_blockmodel_dl(g)
        print("Done.")

        # Get block assignments for each level
        levels = state.get_levels()
        for i, level in enumerate(levels):
            blocks = level.get_blocks()
            colapsed = level.g
            # Compute the modularity
            print(f"Level {i} number of nodes: {colapsed.num_vertices()}")
            print(f"Level {i} number of edges: {colapsed.num_edges()}")
            modularity = gt.modularity(colapsed, blocks)
            print(f"Level {i} block assignments: {blocks.a}")
            print(f"Level {i} modularity: {modularity}")

        # print("\tDrawing...", end="")
        # state.draw(output=save_path + f"nested_SBM_{threshold}.svg")
        # print("Done.")

        # Get the entropy of the partitions
        entropy = state.entropy()
        print(f"Entropy: {entropy}")
    
    # now do it on the complete graph with weights being the covariance :
    edges = torch.zeros_like(cov)
    edges[cov>0] = 1
    g = gt.Graph(directed=False)
    g.add_edge_list(torch.nonzero(edges).numpy())

    weights = cov[edges == 1].numpy()
    g.ep['weight'] = g.new_edge_property("double", vals=weights)

    state_args = dict(
        recs=[g.ep['weight']],
        rec_types=['real-normal'],
    )
    state = gt.minimize_nested_blockmodel_dl(g, state_args=state_args)
    levels = state.get_levels()
    for i, level in enumerate(levels):
        blocks = level.get_blocks()
        colapsed = level.g
        # Compute the modularity
        print(f"Level {i} number of nodes: {colapsed.num_vertices()}")
        print(f"Level {i} number of edges: {colapsed.num_edges()}")
        modularity = gt.modularity(colapsed, blocks)
        print(f"Level {i} block assignments: {blocks.a}")
        print(f"Level {i} modularity: {modularity}")

    # print("\tDrawing...", end="")
    # state.draw(output=save_path + f"nested_SBM_complete.svg")
    # print("Done.")



if __name__ == "__main__":
    JSON_args = {
        'generate_cov': args.generate_cov,
        'fit_cov': args.fit_cov,
        'identity_dict': idd,
        'SVD_dict': svd,
        'White_dict': args.White_dict,
        'activation': get_act,
        'attribution': get_attr,
        'use_resid': args.use_resid,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'steps': args.steps,
        'aggregation': args.aggregation,
        'node_threshold': args.node_threshold,
        'edge_threshold': args.edge_threshold,
        'ctx_len': args.ctx_len,
        'save_path': save_path,
    }

    print("Saving at : ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import json
    with open(save_path + "args.json", "w") as f:
        json.dump(JSON_args, f)

    if args.generate_cov:
        cov = generate_cov()
        if get_act:
            print("Saving activations...", end="")
            torch.save(cov['act'], save_path + "act_cov.pt")
            print("Done.")
        if get_attr:
            print("Saving attributions...", end="")
            torch.save(cov['attr'], save_path + "attr_cov.pt")
            print("Done.")
    
    if args.fit_cov:
        if get_act:
            act_cov = torch.load(save_path + "act_cov.pt", map_location='cpu')
            fit_SBM(act_cov)
        if get_attr:
            attr_cov = torch.load(save_path + "attr_cov.pt", map_location='cpu')
            fit_SBM(attr_cov)
        