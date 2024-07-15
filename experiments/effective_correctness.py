"""
python -m experiments.effective_correctness &
"""
NODE_THRESHOLD = 0.0001
EDGE_THRESHOLD = 0.01
NODE_ABLATION = True

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metric functions

from utils.metric_fns import *

metric_fn_dict = {
    "logit": metric_fn_logit,
    "KL": metric_fn_KL,
    "accuracy": metric_fn_acc,
    "MRR": metric_fn_MRR,
}

# Load the model and its dictionaries

from utils.experiments_setup import load_model_and_modules, load_saes

print("Loading model and modules...", end="")
pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules(device=device)
print("Done.")

print("Loading SAEs...", end="")
dictionaries = load_saes(
    pythia70m,
    pythia70m_embed,
    pythia70m_resids,
    pythia70m_attns,
    pythia70m_mlps,
    device=device,
)
print("Done.")

# define a toy input

clean = "When Mary and John went to the store, John gave a drink to"
tokens = pythia70m.tokenizer(clean, return_tensors="pt")
trg_idx = torch.tensor([-1]).to(device)
trg = tokens["input_ids"].to(device)[0][2].reshape(1)

print("clean:", clean)
print("trg:", trg)
print("trg_idx:", trg_idx)

# get the circuit

from connectivity.effective import get_circuit

print("Getting circuit...")
circuit = get_circuit(
    clean,
    None,
    model=pythia70m,
    embed=pythia70m_embed,
    resids=pythia70m_resids,
    dictionaries=dictionaries,
    metric_fn=metric_fn_logit,
    metric_kwargs={"trg": (trg_idx, trg)},
    edge_threshold=EDGE_THRESHOLD,
    node_threshold=NODE_THRESHOLD,
    nodes_only=False#NODE_ABLATION,
)
print("Done.")

# evaluate the circuit

# start at layer : starting from the embedding layer can be surprisingly bad in some cases, so starting a little bit after might help.

start_at_layer = 1

submodules = [pythia70m_embed] if start_at_layer == -1 else []

if hasattr(pythia70m, "gpt_neox"):
    n_layers = len(pythia70m.gpt_neox.layers)
elif hasattr(pythia70m, "cfg"):
    n_layers = pythia70m.cfg.n_layers

for i in range(max(start_at_layer, 0), n_layers):
    submodules.append(pythia70m_resids[i])

from evaluation.faithfulness import faithfulness

print("Evaluating faithfulness...")
thresholds = torch.logspace(-2, 0, 5, 10).tolist()
faith = faithfulness(
    pythia70m,
    submodules=submodules,
    sae_dict=dictionaries,
    name_dict=submod_names,
    clean=clean,
    circuit=circuit,
    thresholds=thresholds,
    metric_fn=metric_fn_dict,
    metric_fn_kwargs={"trg": (trg_idx, trg)},
    patch=None,
    default_ablation='zero',
    get_graph_info=True,
    node_ablation=NODE_ABLATION,
)
print("Done.")

# Plot the results :

from utils.plotting import plot_faithfulness

save_path = f"/scratch/pyllm/dhimoila/effective_correctness/esal{start_at_layer}/"
plot_faithfulness(faith, metric_fn_dict, save_path)