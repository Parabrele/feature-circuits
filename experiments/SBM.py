"""
python -m experiments.SBM -fc --load_path /scratch/pyllm/dhimoila/output/functional/ROI/id/ioi/attr_cov.pt --save_path /scratch/pyllm/dhimoila/output/functional/ROI/id/ioi/
"""

##########
# Parsing arguments
##########

print("Parsing arguments and importing.")

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--fit_cov", "-fc", action="store_true", help="Fit a SBM to the covariance matrices of activations stored at `load_path`.")
parser.add_argument("--sparsity_threshold", type=float, default=-1, help="If -1, do not threshold according to some sparsity requirement. Otherwise, expected to be between 0 and 1, and gives the target density.")
parser.add_argument("--correlation_threshold", type=float, default=-1, help="If -1, do not threshold according to some sparsity requirement. Otherwise, expected to be between 0 and 1, and gives the target density.")

parser.add_argument("--fit_graph", "-fg", action="store_true", help="Fit a SBM to a graph given either as a networkx, networkit or graph_tool graph, or a dict of dict of sparse connections, stored at `load_path`.")

parser.add_argument("--load_path", "-load_path", type=str, default='/scratch/pyllm/dhimoila/output/', help="Path to load the inputs.")
parser.add_argument("--save_path", "-save_path", type=str, default='/scratch/pyllm/dhimoila/output/', help="Path to save and load the outputs.")

args = parser.parse_args()

fit_cov = args.fit_cov
sparsity_threshold = args.sparsity_threshold
correlation_threshold = args.correlation_threshold

fit_graph = args.fit_graph

load_path = args.load_path
save_path = args.save_path

##########
# Imports
##########

import os

import graph_tool.all as gt
import torch

from evaluation.SBM import cov2corr, fit_nested_SBM, plot_hierarchy, plot_3D_hierarchy, plot_block_corr, plot_block_prob, plot_distribution_fit

from utils.graph_utils import to_graph

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
        "fit_cov": fit_cov,
        "sparsity_threshold": sparsity_threshold,
        "correlation_threshold": correlation_threshold,
        "fit_graph": fit_graph,
        "load_path": load_path,
        "save_path": save_path,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Saving at : ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import json
    with open(save_path + "args.json", "w") as f:
        json.dump(JSON_args, f)

    if args.fit_cov:
        print("Getting correlation matrix")
        cov = torch.load(load_path, map_location=device)
        cov = cov.cov.detach()
        print("Covariance shape :", cov.shape)
        cov = cov2corr(cov)

        print("Fitting SBM")
        state = fit_nested_SBM(cov, already_correlation=True)

        print("Plotting")
        plot_hierarchy(state, save_path=save_path + "hierarchy.pdf")
        plot_block_corr(cov, state, already_correlation=True, save_path=save_path + "block_corr.png")
    
    if args.fit_graph:
        print("Getting the graph")
        graph = torch.load(load_path, map_location=device)
        if "nodes" in graph:
            nodes = graph["nodes"]
        edges = graph["edges"]
        graph = to_graph(edges)
        