import os

import torch as t

from utils.sparse_coo_helper import sparse_coo_maximum
from utils.activation import SparseAct

def save_circuit(save_dir, nodes, edges, num_examples, dataset_name=None, model_name=None, node_threshold=None, edge_threshold=None):
    save_dict = {
        "nodes" : dict(nodes),
        "edges" : dict(edges)
    }
    node_threshold = str(node_threshold) if node_threshold is not None else 'None'
    node_threshold = node_threshold.replace('.', '_')
    edge_threshold = str(edge_threshold) if edge_threshold is not None else 'None'
    edge_threshold = edge_threshold.replace('.', '_')

    if dataset_name is not None:
        save_basename = f"{dataset_name}_{model_name}_node{node_threshold}_edge{edge_threshold}_n{num_examples}"
    else:
        save_basename = f"{num_examples}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}{save_basename}.pt', 'wb') as outfile:
        t.save(save_dict, outfile)

def load_latest(save_dir, merge_gpus=True, device=None):
    files = os.listdir(save_dir)

    none_to_merge = True
    for f in files:
        if os.path.isdir(f'{save_dir}{f}') and (f.startswith('gpu') or f.startswith('cuda')):
            none_to_merge = False
            break
    if none_to_merge:
        merge_gpus = False
    
    if merge_gpus:
        gpu_dirs = [f for f in files if os.path.isdir(f'{save_dir}{f}') and (f.startswith('gpu') or f.startswith('cuda'))]

        tot_circuit = None
        for gpu in gpu_dirs:
            print(f"Loading from {gpu}...", end='')
            circuit = load_latest(f'{save_dir}{gpu}/', merge_gpus=False)
            if tot_circuit is None:
                tot_circuit = circuit
            else:
                for k, v in circuit[0].items():
                    if v is not None:
                        d = device if device is not None else v.device
                        if type(v) == t.Tensor:
                            tot_circuit[0][k] = t.maximum(tot_circuit[0][k].to(d), v.to(d))
                        else:
                            tot_circuit[0][k] = SparseAct.maximum(tot_circuit[0][k].to(d), v.to(d))
                for ku, vu in circuit[1].items():
                    for kd, vd in vu.items():
                        if vd is not None:
                            d = device if device is not None else vd.device
                            tot_circuit[1][ku][kd] = sparse_coo_maximum(tot_circuit[1][ku][kd].to(d), vd.to(d))
            print(' done')

        save_circuit(save_dir + 'merged/', *tot_circuit, 0)
        return tot_circuit

    else:
        files = [save_dir + f for f in files if f.endswith('.pt')]
        files = sorted(files, key=os.path.getmtime)
        if len(files) == 0:
            raise ValueError(f"No files found in save directory {save_dir}")
        latest_file = files[-1]
        return load_from(f'{latest_file}', device=device)

def load_circuit(save_dir, dataset_name, model_name, node_threshold, edge_threshold, num_examples):
    path = f'{save_dir}{dataset_name}_{model_name}_node{node_threshold}_edge{edge_threshold}_n{num_examples}.pt'
    return load_from(path)

def load_from(circuit_path, device=None):
    with open(circuit_path, 'rb') as infile:
        save_dict = t.load(infile, map_location=t.device('cpu'))
    try:
        nodes = save_dict['nodes']
    except KeyError:
        nodes = None
    edges = save_dict['edges']

    if device is not None:
        for k, v in nodes.items():
            nodes[k] = v.to(device)
        for k, v in edges.items():
            for kk, vv in v.items():
                edges[k][kk] = vv.to(device)

    return nodes, edges