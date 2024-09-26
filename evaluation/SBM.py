"""
https://journals.aps.org/pre/pdf/10.1103/PhysRevE.97.012306 and https://arxiv.org/pdf/1404.0431 :
weights are very important to keep. In fig 3 he fits on a covariance matrix and compare the infered distribution to that of shuffled data (with same empirical distribution)
TODO : how are fig 2 and 3 c fits generated ?
"""

import numpy as np
import torch
from tqdm import tqdm
import graph_tool.all as gt
import networkx as nx

from matplotlib import pyplot as plt
import matplotlib

from utils.graph_utils import to_graph

def cov2corr(cov):
    """
    Cov : covariance matrix

    Transforms it into a correlation matrix.
    """
    var = torch.diag(cov)
    std = torch.sqrt(var)
    corr = cov / torch.outer(std, std)
    # put 0 on the diagonal
    diag_idx = torch.arange(corr.shape[0])
    corr[diag_idx, diag_idx] = 0.
    return corr

def fit_nested_SBM(cov, already_correlation=False, threshold=None, sparsity_threshold=None):
    """
    Cov : covariance matrix

    Transforms it into a correlation matrix then apply fisher's formula to get each edge's weight.
    Fit a nested SBM on the complete graph with these weights.
    """
    if isinstance(cov, dict):
        cov = to_graph(cov)
        
    state_args = dict(
        deg_corr=False,
    )

    if isinstance(cov, torch.Tensor):
        # First transform the covariance matrix into a correlation matrix and apply fisher's formula to get the weights in a normal distribution
        corr = cov if already_correlation else cov2corr(cov)
        weights = 2 * torch.arctanh(corr)

        # Then create a full graph with these weights and fit a nested SBM on it
        n = cov.shape[0]
        g = gt.Graph(directed=False)
        g.add_vertex(n)
        
        edge_list = torch.nonzero(torch.ones_like(cov)).numpy()
        weight_assignment = weights.numpy()[edge_list[:, 0], edge_list[:, 1]]

        perm = np.argsort(np.abs(weight_assignment))[::-1]
        edge_list = edge_list[perm]
        weight_assignment = weight_assignment[perm]
        sparsity = 1
        edge_list = edge_list[:int(sparsity * len(edge_list))]
        weight_assignment = weight_assignment[:int(sparsity * len(weight_assignment))]

        print("Smallest weight:", corr.numpy()[edge_list[-1][0], edge_list[-1][1]])

        g.add_edge_list(edge_list)
        g.ep['weight'] = g.new_edge_property("float", vals=weight_assignment)

        state_args['recs']=[g.ep['weight']]
        state_args['rec_types']=['real-normal']

    elif isinstance(cov, nx.Graph):
        relabel_nodes = {node : i for i, node in enumerate(cov.nodes)}
        renamed = nx.relabel_nodes(cov, relabel_nodes)
        g = gt.Graph(directed=False)
        g.add_edge_list(renamed.edges)
    
    n_threads = gt.openmp_get_num_threads()
    with gt.openmp_context(n_threads):
        state = gt.minimize_nested_blockmodel_dl(g, state_args=state_args)

        # improve solution with merge-split
        for i in tqdm(range(100)):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    # TODO : find min entropy (with/without degree correction)
    # TODO : shuffle weights and edges to get a null model
    print("Model description length:", state.entropy())
    print("Summary :")
    state.print_summary()
    return state

def plot_hierarchy(state, save_path=None):
    """
    Use gt.draw_hierarchy.
    Plot the graph using the hierarchy of the blocks. Produces a circular graph with nodes grouped by blocks and edges routed according to the hierarchy.
    Produce a plot similar to : https://graph-tool.skewed.de/static/doc/_images/celegansneural_nested_mdl.png
    """
    gt.draw_hierarchy(state, output=save_path)
    g = state.get_levels()[0].g
    # modify g weights to lambda x : 0.01 + abs(x)
    g.ep['weight'] = g.new_edge_property("float", vals=np.abs(g.ep.weight.a) + 1e-2)
    state.draw(edge_color=gt.prop_to_size(g.ep.weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 1, 4, power=1, log=True),
           edge_gradient=[], output=save_path.replace(".pdf", "_weighted.pdf"))

def plot_3D_hierarchy(state, save_path=None):
    """
    Use gt.draw_hierarchy_3D.
    Plot the graph using the hierarchy of the blocks. Produces a 3D graph where each level of the hierarchy is represented above the previous one.
    Produce a plot similar to : https://graph-tool.skewed.de/static/doc/_images/nested-diagram.svg
    """
    raise NotImplementedError

def plot_block_corr(cov, state, already_correlation=False, save_path=None):
    """
    Plot the correlation matrix reordered by the block assignments. Plot lines between the blocks.
    """
    if not already_correlation:
        cov = cov2corr(cov)
    corr = cov.numpy()
    blocks = state.get_levels()[0].get_blocks().a
    perm = np.argsort(blocks)
    corr = corr[perm][:, perm]
    blocks = blocks[perm]

    # Pad with zeros to separate the blocks
    block_diff = np.diff(blocks)
    block_diff = np.concatenate([[1], block_diff])
    idx_to_insert = np.where(block_diff)[0]

    corr = np.insert(corr, idx_to_insert, 0, axis=0)
    corr = np.insert(corr, idx_to_insert, 0, axis=1)

    # Plot the correlation matrix
    fig, ax = plt.subplots()
    ax.matshow(corr, cmap='twilight')

    # Now the lines appear white, and we want them to be black. Plot them manually
    for i, idx in enumerate(idx_to_insert):
        ax.axhline(idx + i, color='black', linewidth=0.5)
        ax.axvline(idx + i, color='black', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Save the plot
    if save_path is not None:
        plt.savefig(save_path)
    else:
        raise ValueError("No save path provided.")

    # now plot the distribution of the correlation coefficients
    fig, ax = plt.subplots()
    ax.hist(corr.flatten(), bins=1000)
    ax.set_yscale('log')
    ax.set_title("Correlation coefficients distribution")
    # save the plot
    if save_path is not None:
        plt.savefig(save_path.replace(".png", "_hist.png"))
    else:
        raise ValueError("No save path provided.")

def plot_block_prob(cov, g, state, already_correlation=False, save_path=None):
    """
    Same plot as before, but replace the correlation by the value of the probability of an edge between the two blocks.
    """
    raise NotImplementedError

def plot_distribution_fit(cov, g, state, already_correlation=False, save_path=None):
    """
    Plot the distribution of covariances and the prediced values from the SBM.
    """
    raise NotImplementedError

"""
Script to test conversion from nx to gt graph :

import networkx as nx
import graph_tool.all as gt

g = nx.Graph()
g.add_edge("a", "b")
g.add_edge("a", "c")
g.add_edge("b", "c")
g.add_edge("gertrude44du69", "cbastienxx")
g.add_edge(45, 695)
g.add_edge(45, "gertrude44du69")

g_gt = gt.Graph(directed=False)
g_gt.add_edge_list(g.edges)
print(g_gt.get_vertices())
print(g_gt.get_edges())
print(g_gt)

This does not work, we need to convert the nodes to integers first.

rename_nodes = {node : i for i, node in enumerate(g.nodes)}
renamed_g = nx.relabel_nodes(g, rename_nodes)
g_gt = gt.Graph(directed=False)
g_gt.add_edge_list(renamed_g.edges)
print(g_gt.get_vertices())
print(g_gt.get_edges())
print(g_gt)
"""
