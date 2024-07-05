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
from data.buffer import TokenBatches

from circuit import get_activation

print("Done importing.")

##########
# Functions
##########

# TODO : modularity is not a good thing to measure here, as some blocks might have very low internal edges.

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
        