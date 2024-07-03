import torch

import networkit as nk

from utils.graph_utils import to_Digraph, get_avg_degree, get_connected_components, get_density, get_degree_distribution

def shuffle_edges(edges):
    """
    edges : dict of dict of sparse_coo tensors
    returns a dict of dict of sparse_coo tensors, with edges uniformly randomly assigned to the active nodes
    """
    new_edges = {}
    for up in edges:
        new_edges[up] = {}
        for down in edges[up]:
            if down == 'y':
                new_edges[up][down] = edges[up][down]
                perm = torch.randperm(edges[up][down].values().size(0))
                new_edges[up][down].values()[...] = edges[up][down].values()[perm]
                continue
            up_alive = edges[up][down].indices()[1].unique()
            down_alive = edges[up][down].indices()[0].unique()
            n_edges = edges[up][down].values().size(0)
            new_edges[up][down] = torch.sparse_coo_tensor(
                torch.stack([
                    down_alive[torch.randint(down_alive.size(0), (n_edges,))],
                    up_alive[torch.randint(up_alive.size(0), (n_edges,))]
                ]),
                torch.ones(n_edges, device=edges[up][down].device, dtype=torch.bool),
                edges[up][down].size(),
                device=edges[up][down].device,
                dtype=torch.bool
            ).coalesce()
    return new_edges

def mean_std_modularity(edges, n_samples=50, method='Leiden'):
    """
    edges : dict of dict of sparse_coo tensors
    returns a tuple (mean, std) of the modularity of the graph
    """
    modularities = []
    from tqdm import tqdm
    for _ in tqdm(range(n_samples)):
        shuffled_edges = shuffle_edges(edges)
        modularities.append(modularity(shuffled_edges))#[1])
        print(modularities[-1])
    return torch.tensor(modularities).mean().item(), torch.tensor(modularities).std().item()

def Z_score(edges, n_samples=50):
    """
    edges : dict of dict of sparse_coo tensors
    returns a tuple (mean, std) of the modularity of the graph
    """
    mean, std = mean_std_modularity(edges, n_samples)
    return (modularity(edges)-mean)/std#[1] - mean) / std

# TODO : implement the communities function
def communities(graph, method='Leiden', method_kwargs={}):
    pass

# TODO : Louvain & spectral clustering
# TODO : change this function : one for extracting the communities (either Louvain or spectral clustering)
#        and one for computing the modularity of a given partition
@torch.no_grad()
def modularity(
    graph,
    method='Leiden',
    iterations=10,
    gamma=1.0, # 0 : single community, 1 : default, 2m : singletons
    weighted=True,
    log_weighted=False,
):
    """
    circuit : tuple (nodes, edges), dict or nx.DiGraph
    method : str
        only 'Leiden' is available for now
    iterations : int
        number of iterations for the algorithm
    gamma : float
        resolution parameter. Higher values lead to smaller communities
    weighted : bool
        if True, the graph is weighted
    log_weighted : bool
        if True, the weights are log( _ / eps )
    returns (dict : subset -> score), float, float
    """

    if isinstance(graph, tuple) or isinstance(graph, dict):
        graph = to_Digraph(graph)
    
    try :
        graph = graph.copy()
        graph.remove_node('y')
    except:
        pass

    G = nk.nxadapter.nx2nk(graph, weightAttr='weight' if weighted else None)
    G = nk.graphtools.toUndirected(G)

    # get communities
    if method == 'Leiden':
        partition = nk.community.ParallelLeiden(
            G,
            gamma=gamma,
            iterations=iterations,
        )
        partition.run()
        communities = partition.getPartition()
    else:
        raise NotImplementedError(f"Method {method} is not implemented")

    # get quality
    modularity = nk.community.Modularity().getQuality(communities, G)

    subset_ids = communities.getSubsetIds()
    print(f"Weighted : {weighted}, log weighted : {log_weighted}")
    dict_communities = {
        subset_id : {
            'n_nodes' : len(communities.getMembers(subset_id)),
            #'score' : single_community_modularity(G, communities.getMembers(subset_id), weighted=weighted, output_n_edges=True)
        }
        for subset_id in subset_ids
    }

    to_del = []
    for subset_id in dict_communities:
        if dict_communities[subset_id]['n_nodes'] == 1:
            to_del.append(subset_id)
            continue
        #dict_communities[subset_id]['n_edges'], dict_communities[subset_id]['score'] = dict_communities[subset_id]['score']
    
    print(f"N singletons : {len(to_del)}")
    for subset_id in to_del:
        del dict_communities[subset_id]

    # normalize by number of nodes ? larger communities are more likely to have better scores... choose whether to do it or not based on dict_communities
    #avg_single_community_modularity = sum([v['score'] * v['n_nodes'] for v in dict_communities.values()]) / sum([v['n_nodes'] for v in dict_communities.values()])

    print(f"Modularity : {modularity}")#, avg single community modularity : {avg_single_community_modularity}")
    print(f"Number of communities : {len(list(dict_communities.keys()))}")
    print(f"Communities : {dict_communities}")

    return modularity

@torch.no_grad()
def evaluate_graph(
    circuit,
    prune=False,
):
    """
    circuit : tuple (nodes, edges), dict (edges), nx.DiGraph or nk.Graph
    returns a dict
        'nedges' -> int
        'nnodes' -> int
        'avgdegree' -> float
        ...
        'pruned' -> dict
            'nedges' -> int
            'nnodes' -> int
            'avgdegree' -> float
            ...
    
    other interesting metrics :
        - number of connected components
        - Global clustering coefficient # should be close or equal to 0, as the graph is layered
        - density
        - degree distribution
        - transitivity
        - s metric
    """

    if isinstance(circuit, tuple) or isinstance(circuit, dict) or isinstance(circuit, nk.Graph):
        G = to_Digraph(circuit)
    else:
        G = circuit

    results = {
        'nedges' : G.number_of_edges(),
        'nnodes' : G.number_of_nodes(),
        'avgdegree' : get_avg_degree(G),
        'connected_components' : get_connected_components(G),
        'density' : get_density(circuit),
        'degree_distribution' : get_degree_distribution(G),
        'modularity' : modularity(G),
        #'z_score' : Z_score(circuit),
    }

    print("Everything after this is for random graphs")

    mu, sigma = mean_std_modularity(circuit, 1)
    results['mean_modularity'] = mu
    results['std_modularity'] = sigma
    results['z_score'] = (results['modularity'] - mu) / sigma

    # print("Computing modularity")
    # results['modularity'] = modularity(G)
    # print(f"Modularity : {results['modularity']}")

    # print("Computing modularity")
    # G_unweighted = G.copy()
    # for u, v in G_unweighted.edges:
    #     G_unweighted[u][v]['weight'] = 1
    # results['modularity'] = modularity(G_unweighted)
    # print(f"Modularity : {results['modularity']}")
    # print("Computing treewidth")
    # results['treewidth'] = nx.approximation.treewidth_min_degree(nx.Graph(G))
    # print(f"Treewidth : {results['treewidth']}")

    if prune:
        pruned_circuit = prune(G)
        pruned = evaluate_graph(pruned_circuit)
        results['pruned'] = pruned

    return results
