import torch

import networkx as nx
# import networkit as nk

from utils.activation import SparseAct
from utils.sparse_coo_helper import sparse_coo_maximum

def merge_circuits(
    tot_circuit,
    circuit,
    aggregation="max",
):
    if tot_circuit is None:
        tot_circuit = circuit
    else:
        for k, v in circuit[0].items():
            if v is not None:
                if aggregation == "sum":
                    tot_circuit[0][k] += v
                elif aggregation == "max":
                    if type(v) == SparseAct:
                        tot_circuit[0][k] = SparseAct.maximum(tot_circuit[0][k], v)
                    else:
                        tot_circuit[0][k] = torch.maximum(tot_circuit[0][k], v)
                else:
                    raise ValueError(f"Unknown aggregation method {aggregation}")
        for ku, vu in circuit[1].items():
            for kd, vd in vu.items():
                if vd is not None:
                    if aggregation == "sum":
                        tot_circuit[1][ku][kd] += vd
                    elif aggregation == "max":
                        tot_circuit[1][ku][kd] = sparse_coo_maximum(tot_circuit[1][ku][kd], vd)

    return tot_circuit

def mean_circuit(circuit, n):
    for k, v in circuit[0].items():
        if v is not None:
            circuit[0][k] /= n
    for ku, vu in circuit[1].items():
        for kd, vd in vu.items():
            if vd is not None:
                circuit[1][ku][kd] /= n

    return circuit

@torch.no_grad()
def get_mask(graph, threshold, threshold_on_nodes=False):
    """
    graph :
    edges : dict of dict of sparse_coo tensors, [name_upstream][name_downstream] -> edge weights

    returns a dict similar to edges but binary sparse coo tensors : only weights above threshold are kept
    """
    has_nodes = False
    if isinstance(graph, tuple):
        has_nodes = True and threshold_on_nodes
        nodes = graph[0]
        edges = graph[1]
    else:
        edges = graph
        # TODO remove this when all seems to be working as expected
        raise ValueError("expected a tuple of nodes and edges")
    
    # First threshold the nodes to know which to keep and discard
    node_mask = None
    if has_nodes:
        node_mask = {}
        for module in nodes:
            if module == 'y':
                continue
            node_mask[module] = nodes[module].abs() > threshold

    if edges is None:
        return node_mask, None
    edge_mask = {}

    for upstream in edges:
        edge_mask[upstream] = {}
        for downstream in edges[upstream]:
            weights = edges[upstream][downstream].coalesce()
            if threshold == -1:
                edge_mask[upstream][downstream] = torch.sparse_coo_tensor(
                    [[]] if downstream == 'y' else [[], []],
                    [],
                    weights.size()
                )
            else:
                if has_nodes and downstream != 'y':                
                    upstream_mask = node_mask[upstream].to_tensor()[weights.indices()[1]]
                    downstream_mask = node_mask[downstream].to_tensor()[weights.indices()[0]]
                    mask = upstream_mask & downstream_mask
                else:
                    mask = weights.values() > threshold
                edge_mask[upstream][downstream] = torch.sparse_coo_tensor(
                    weights.indices()[:, mask],
                    torch.ones(mask.sum(), device=weights.device, dtype=torch.bool),
                    weights.size(),
                    dtype=torch.bool
                )
    return node_mask, edge_mask

@torch.no_grad()
def to_Digraph(circuit, discard_res=False, discard_y=False):
    """
    circuit : tuple (nodes, edges), dict or nk.Graph
    returns a networkx DiGraph
    """
    if isinstance(circuit, nx.DiGraph):
        return circuit
    # elif isinstance(circuit, nk.Graph):
    #     return nk.nxadapter.nk2nx(circuit)
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

def to_graph(circuit, discard_res=False, discard_y=False):
    """
    return a networkx undirected graph
    """
    graph = to_Digraph(circuit, discard_res=discard_res, discard_y=discard_y)
    return graph.to_undirected()

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
    if circuit is None:
        return None
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
        # if G is given as a dict of SparseAct : these are only the nodes
        # if G is given as a dict of dict of sparse_coo tensors : these are the edges
        is_edges = False
        for up in G:
            if isinstance(G[up], dict):
                is_edges = True
                break
        if not is_edges:
            for up in G:
                n_nodes += G[up].to_tensor().sum().item()
            return n_nodes
        else:
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
    if G is None:
        return 0
    if isinstance(G, nx.DiGraph):
        return G.number_of_edges()
    elif isinstance(G, dict):
        # if G is a dict of dict of sparse coo tensors, they contain the edges :
        n_edges = 0
        for up in G:
            for down in G[up]:
                n_edges += G[up][down].values().size(0)
        return n_edges
    elif isinstance(G, tuple):
        # if G is a tuple of nodes and edges, we consider that we are in the node ablation setting and the edge dict is only here to give the dependencies between layers.
        n_edges = 0
        nodes, edges = G
        for up in edges:
            for down in edges[up]:
                if up == 'y':
                    n_edges += nodes[down].to_tensor().sum().item()
                    continue
                elif down == 'y':
                    n_edges += nodes[up].to_tensor().sum().item()
                    continue
                n_up = nodes[up].to_tensor().sum().item()
                n_down = nodes[down].to_tensor().sum().item()
                n_edges += n_up * n_down
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
    if edges is None:
        return 0
    if isinstance(edges, nx.DiGraph):
        return nx.density(edges)
    if isinstance(edges, tuple):
        n_edges = get_n_edges(edges)
        edges = edges[1]
    else:
        n_edges = 0
    max_edges = 0
    for up in edges:
        for down in edges[up]:
            if not isinstance(edges, tuple):
                n_edges += edges[up][down].values().size(0)
            max_edges += edges[up][down].size(0) * (edges[up][down].size(1) if down != 'y' else 1)
    max_edges = max(max_edges, 1)
    return n_edges / max_edges

def get_degree_distribution(G):
    return nx.degree_histogram(G)
