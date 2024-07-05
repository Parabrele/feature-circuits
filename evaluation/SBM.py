"""
https://arxiv.org/pdf/1705.10225

simple graphs without self loops :
    edges \in {0, 1} : (4)
multi graphs :
    edges \in N (5)

In the sparse limit (p/lambda = O(1/N), N the number of nodes), ln P(A|p/lambda, b) are the same

poisson distribution to generate edges :
Also has the same log probability, and might be nicer to use. All these models therefore generate the same networks in the sparse limit.

/!\ TODO : all of this seems to work if we are in the sparse case ! Make sure that it is the case !

https://journals.aps.org/pre/pdf/10.1103/PhysRevE.97.012306 and https://arxiv.org/pdf/1404.0431 :
weights are very important to keep. In fig 3 he fits on a covariance matrix and compare the infered distribution to that of shuffled data (with same empirical distribution)
TODO : how are fig 2 and 3 c fits generated ?

For correlation matrix : see sec 3.B of https://journals.aps.org/pre/pdf/10.1103/PhysRevE.97.012306.
    - [-1, 1] -> [-inf, inf] : 2 arctanh(x) = ln((1+x)/(1-x)) (fisher's formula)
    - When randomly shuffling the correlation matrix, only a single group is found. Even though there is still a bimodal distribution of the correlation coefficients, it is uncorrelated with any partition of the deputies, and the best fit is a normal distribution.
"""

import numpy as np
import torch
import graph_tool.all as gt

def cov2corr(cov):
    """
    Cov : covariance matrix

    Transforms it into a correlation matrix.
    """
    var = torch.diag(cov)
    std = torch.sqrt(var)
    corr = cov / torch.outer(std, std)
    return corr

def fit_nested_SBM(cov, already_correlation=False):
    """
    Cov : covariance matrix

    Transforms it into a correlation matrix then apply fisher's formula to get each edge's weight.
    Fit a nested SBM on the complete graph with these weights.
    """
    corr = cov if already_correlation else cov2corr(cov)
    
    weights = 2 * torch.arctanh(corr)
    n = cov.shape[0]
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    
    g.add_edge_list(torch.nonzero(torch.ones_like(cov)).numpy())
    g.ep['weight'] = g.new_edge_property("float", vals=weights.numpy()) # TODO : check if this is correct

    state_args = dict(
        recs=[g.ep['weight']],
        rec_types=['real-normal'],
    )
    state = gt.minimize_nested_blockmodel_dl(g, state_args=state_args)

def plot_hierarchy(state, save_path=None):
    """
    Use gt.draw_hierarchy.
    Plot the graph using the hierarchy of the blocks. Produces a circular graph with nodes grouped by blocks and edges routed according to the hierarchy.
    Produce a plot similar to : https://graph-tool.skewed.de/static/doc/_images/celegansneural_nested_mdl.png
    """
    pass

def plot_3D_hierarchy(state, save_path=None):
    """
    Use gt.draw_hierarchy_3D.
    Plot the graph using the hierarchy of the blocks. Produces a 3D graph where each level of the hierarchy is represented above the previous one.
    Produce a plot similar to : https://graph-tool.skewed.de/static/doc/_images/nested-diagram.svg
    """
    pass

def plot_block_prob(cov, g, state, already_correlation=False, save_path=None):
    """
    Same plot as before, but replace the correlation by the value of the probability of an edge between the two blocks.
    """
    pass

def plot_block_corr(cov, g, state, already_correlation=False, save_path=None):
    """
    Plot the correlation matrix reordered by the block assignments. Plot lines between the blocks.
    """
    pass

def plot_distribution_fit(cov, g, state, already_correlation=False, save_path=None):
    """
    Plot the distribution of covariances and the prediced values from the SBM.
    """
    pass
