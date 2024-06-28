import os
import torch as t
import evaluation
import utils
from utils import SparseAct
import plotly.graph_objects as go
from plotly.subplots import make_subplots

device = None

base_path = "/scratch/pyllm/dhimoila/output/120624_01/circuit/"

interval = 50
start = 1
end = 1350 // 50
n_gpus = 8

n_examples = []
n_nodes = []
n_edges = []
avg_degree = []

saved_circuit = None

def fuse_idx(c1, c2):
    # c1, c2 : dict of dict of tensors. Return a dict of dict of tensors
    res = {}
    for k, v in c1.items():
        res[k] = {}
        for k2, v2 in v.items():
            res[k][k2] = t.cat([v2.coalesce().indices(), c2[k][k2].coalesce().indices()], dim=1) # n_dim x n_edges
            res[k][k2] = t.unique(res[k][k2], dim=-1)
            res[k][k2] = t.sparse_coo_tensor(res[k][k2], t.ones(res[k][k2].shape[1]), v2.shape).coalesce()
    return res

for gpu in range(n_gpus):
    base = base_path + f"cuda_{gpu}/"
    if gpu != 0:
        start = end-3
    for i in range(start, end + 1):
        try :
            n_examples.append(i * interval + gpu * end * interval)
            path = base + f"{i*interval}.pt"
            print(i * interval)
            circuit = utils.load_from(path)[1]
            if saved_circuit is not None:
                circuit = fuse_idx(saved_circuit, circuit)

            n_nodes.append(evaluation.get_n_nodes(circuit))
            n_edges.append(evaluation.get_n_edges(circuit))
            avg_degree.append(n_edges[-1] / n_nodes[-1])
        except:
            print(f"No file found for gpu {gpu} and example {i*interval}")

    saved_circuit = circuit

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Number of nodes", "Number of edges", "Average degree"))

fig.add_trace(go.Scatter(x=n_examples, y=n_nodes, mode='lines', name='Number of nodes'), row=1, col=1)
fig.add_trace(go.Scatter(x=n_examples, y=n_edges, mode='lines', name='Number of edges'), row=2, col=1)
fig.add_trace(go.Scatter(x=n_examples, y=avg_degree, mode='lines', name='Average degree'), row=3, col=1)

fig.update_layout(title_text="Circuit properties", xaxis_title="Number of examples", yaxis_title="Value")

fig.write_html(base_path + "properties.html")