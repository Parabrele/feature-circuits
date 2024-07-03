import torch

import networkx as nx
import networkit as nk

@torch.no_grad()
def get_mask(graph, threshold):
    """
    graph :
    edges : dict of dict of sparse_coo tensors, [name_upstream][name_downstream] -> edge weights

    returns a dict similar to edges but binary sparse coo tensors : only weights above threshold are kept
    """
    if isinstance(graph, tuple):
        graph = graph[1]

    edges = graph
    mask = {}

    for upstream in edges:
        mask[upstream] = {}
        for downstream in edges[upstream]:
            weights = edges[upstream][downstream].coalesce()
            if threshold == -1:
                mask[upstream][downstream] = torch.sparse_coo_tensor(
                    [[]] if downstream == 'y' else [[], []],
                    [],
                    weights.size()
                )
            else:
                value_mask = weights.values() > threshold
                mask[upstream][downstream] = torch.sparse_coo_tensor(
                    weights.indices()[:, value_mask],
                    torch.ones(value_mask.sum(), device=weights.device, dtype=torch.bool),
                    weights.size(),
                    dtype=torch.bool
                )
    return mask


@torch.no_grad()
def to_Digraph(circuit, discard_res=False, discard_y=False):
    """
    circuit : tuple (nodes, edges), dict or nk.Graph
    returns a networkx DiGraph
    """
    if isinstance(circuit, nx.DiGraph):
        return circuit
    elif isinstance(circuit, nk.Graph):
        return nk.nxadapter.nk2nx(circuit)
    elif isinstance(circuit, tuple) or isinstance(circuit, dict):
        G = nx.DiGraph()

        if isinstance(circuit, tuple):
            nodes, edges = circuit
        else:
            edges = circuit

        for upstream in edges:
            for downstream in edges[upstream]:
                if downstream == 'y':
                    if discard_y:
                        continue
                    else:
                        for u in edges[upstream][downstream].coalesce().indices().t():
                            u = u.item()
                            if discard_res and u == edges[upstream][downstream].size(0) - 1:
                                continue
                            upstream_name = f"{upstream}_{u}"
                            G.add_edge(upstream_name, downstream, weight=edges[upstream][downstream][u].item())
                        continue
                for d, u in edges[upstream][downstream].coalesce().indices().t():
                    d = d.item()
                    u = u.item()
                    # this weight matrix has shape (f_down + 1, f_up + 1)
                    # reconstruction error nodes are the last ones
                    if discard_res and (
                        d == edges[upstream][downstream].size(0) - 1
                        or u == edges[upstream][downstream].size(1) - 1
                    ):
                        continue
                    
                    upstream_name = f"{upstream}_{u}"
                    downstream_name = f"{downstream}_{d}"
                    G.add_edge(upstream_name, downstream_name, weight=edges[upstream][downstream][d, u].item())

        return G

def to_tuple(G):
    """
    G : nx.DiGraph or nk.Graph
    returns a tuple (nodes, edges)
    """
    raise NotImplementedError

def prune(
    circuit
):
    """
    circuit : nx.DiGraph or dict of dict of sparse_coo tensors
    returns a new nx.DiGraph or dict of dict of sparse_coo tensors
    """
    if isinstance(circuit, nx.DiGraph):
        return prune_nx(circuit)
    else:
        return prune_sparse_coos(circuit)

def reorder_upstream(edges):
    """
    edges : dict of dict of sparse_coo tensors
    returns a dict of dict of sparse_coo tensors
    """
    new_edges = {}
    for up in reversed(list(edges.keys())):
        new_edges[up] = edges[up]
    return new_edges

def coalesce_edges(edges):
    """
    coalesce all edges sparse coo weights
    """
    for up in edges:
        for down in edges[up]:
            edges[up][down] = edges[up][down].coalesce()
    return edges

@torch.no_grad()
def prune_sparse_coos(
    circuit
):
    """
    circuit : edges
    returns a new edges

    Assumes edges is a dict of bool sparse coo tensors. If not, it will just forget the values.
    """
    circuit = coalesce_edges(reorder_upstream(circuit))

    # build a dict module_name -> sparse vector of bools, where True means that the feature is reachable from one embed node
    size = {}
    reachable = {}

    for upstream in circuit:
        for downstream in circuit[upstream]:
            if downstream == 'y':
                size[upstream] = circuit[upstream][downstream].size(0)
                size[downstream] = 1
            else:
                size[upstream] = circuit[upstream][downstream].size(1)
                size[downstream] = circuit[upstream][downstream].size(0)                    
    
    # build a sparse_coo_tensor of ones with size (size['embed']) :
    reachable['embed'] = torch.sparse_coo_tensor(
        torch.arange(size['embed']).unsqueeze(0),
        torch.ones(size['embed'], device=circuit[upstream][downstream].device, dtype=torch.bool),
        (size['embed'],),
        device=circuit[upstream][downstream].device,
        dtype=torch.bool
    ).coalesce()

    for upstream in circuit:
        for downstream in circuit[upstream]:
            if upstream not in reachable:
                raise ValueError(f"Upstream {upstream} reachability not available. Check the order of the keys.")
            if downstream not in reachable:
                reachable[downstream] = torch.sparse_coo_tensor(
                    [[]], [],
                    (size[downstream],),
                    device=circuit[upstream][downstream].device
                )

            idx1 = circuit[upstream][downstream].indices() # (2, n1)
            idx2 = reachable[upstream].indices() # (1, n2)

            # keep only rows of circuit[upstream][downstream] at idx in idx2
            new_edges = torch.sparse_coo_tensor(
                [[]] if downstream == 'y' else [[], []],
                [],
                circuit[upstream][downstream].size(),
                device=circuit[upstream][downstream].device,
                dtype=torch.bool
            ).coalesce()
            for u in idx2[0]:
                mask = (idx1[0] == u if downstream == 'y' else idx1[1] == u)
                new_edges = torch.sparse_coo_tensor(
                    torch.cat([new_edges.indices(), idx1[:, mask]], dim=1),
                    torch.cat([new_edges.values(), circuit[upstream][downstream].values()[mask]]),
                    new_edges.size(),
                    device=circuit[upstream][downstream].device,
                    dtype=torch.bool
                ).coalesce()

            circuit[upstream][downstream] = new_edges

            # now look at what downstream features are reachable, as just the indices in the new_edges and add them to reachable[downstream]
            idx = new_edges.indices()[0].unique()
            reachable[downstream] += torch.sparse_coo_tensor(
                idx.unsqueeze(0),
                torch.ones(idx.size(0), ),
                (size[downstream],),
                device=circuit[upstream][downstream].device
            )
            reachable[downstream] = reachable[downstream].coalesce()
    
    return circuit

@torch.no_grad()
def prune_nx(
    G
):
    """
    circuit : nx.DiGraph
    returns a new nx.DiGraph
    """

    G = G.copy()

    # save the 'embed' nodes and their edges to restore them later
    save = []
    to_relabel = {}
    for node in G.nodes:
        if 'embed' in node:
            save += G.edges(node)
            to_relabel[node] = 'embed'

    # merge nodes from embedding into a single 'embed' node, like 'y' is single.
    G = nx.relabel_nodes(G, to_relabel)

    # do reachability from v -> 'y' for all v, remove all nodes that are not reachable
    reachable = nx.ancestors(G, 'y')
    reachable.add('y')
    complement = set(G.nodes) - reachable

    G.remove_nodes_from(complement)

    # do reachability from 'embed' -> v for all v, remove all nodes that are not reachable
    reachable = nx.descendants(G, 'embed')
    complement = set(G.nodes) - reachable

    G.remove_nodes_from(complement)

    # untangle the 'embed' node into its original nodes and return the new graph
    G.add_edges_from(save)

    return G


def get_n_nodes(G):
    if isinstance(G, nx.DiGraph):
        return G.number_of_nodes()
    elif isinstance(G, dict):
        n_nodes = 0
        for up in G:
            up_nodes = None
            for down in G[up]:
                if down == 'y':
                    nodes = G[up][down].indices()[0].unique()
                else:
                    nodes = G[up][down].indices()[1].unique()
                if up_nodes is None:
                    up_nodes = nodes
                else:
                    up_nodes = torch.cat([up_nodes, nodes])
            n_nodes += up_nodes.unique().size(0)
        return n_nodes
    else :
        raise ValueError("Unknown graph type")

def get_n_edges(G):
    if isinstance(G, nx.DiGraph):
        return G.number_of_edges()
    elif isinstance(G, dict):
        n_edges = 0
        for up in G:
            for down in G[up]:
                n_edges += G[up][down].values().size(0)
        return n_edges
    else :
        raise ValueError("Unknown graph type")

def get_avg_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()

def get_connected_components(G):
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected()

    G = G.copy()
    G.remove_node('y')

    return nx.number_connected_components(G)

def get_density(edges):
    # edges is a dict of dict of sparse_coo tensors
    if isinstance(edges, nx.DiGraph):
        return nx.density(edges)
    n_edges = 0
    max_edges = 0
    for up in edges:
        for down in edges[up]:
            n_edges += edges[up][down].values().size(0)
            max_edges += edges[up][down].size(0) * (edges[up][down].size(1) if down != 'y' else 1)
    return n_edges / max_edges

def get_degree_distribution(G):
    return nx.degree_histogram(G)
