print("Importing...")

TODO :
mean on test, id on train
    python -m experiments.marks_tests --ablation_fn mean --use_corr --save_path /scratch/pyllm/dhimoila/outputs/marks_test/hehehe/ &
mean on test, zero on train
fix zero aplation
zero on train, mean on test for RC dataset

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--test_correctness", action="store_true")

parser.add_argument("--ablation_fn", type=str, default="mean")
parser.add_argument("--use_corr", action="store_true")

parser.add_argument("--n_elts", type=int, default=200)
parser.add_argument("--n_tests", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--eval_batch_size", type=int, default=100)

parser.add_argument("--nb_eval_thresholds", type=int, default=20)
parser.add_argument("--ctx_len", type=int, default=None)

parser.add_argument("--node_threshold", type=float, default=0.001)
parser.add_argument("--edge_threshold", type=float, default=0.001)
parser.add_argument("--start_at_layer", type=int, default=2)

parser.add_argument("--save_path", type=str, default="/scratch/pyllm/dhimoila/outputs/marks_test/")

args = parser.parse_args()

import os

import gc

import math

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

multiprocessing.set_start_method('spawn', force=True)

import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from transformers import logging
logging.set_verbosity_error()

from tqdm import tqdm

from data.buffer import wikipedia_buffer, gp_buffer, gt_buffer, ioi_buffer, bool_buffer, mixture_buffer, unpack_batch

from dummy.circuit import get_circuit as marks_circuit
from connectivity.effective import get_circuit as our_circuit

from evaluation.faithfulness import faithfulness as faithfulness_fn

from utils.activation import SparseAct
from utils.savior import save_circuit
from utils.plotting import plot_faithfulness
from utils.metric_fns import metric_fn_logit, metric_fn_KL, metric_fn_statistical_distance, metric_fn_acc, metric_fn_MRR
from utils.experiments_setup import load_model_and_modules, load_saes

print("Done.")

if args.ablation_fn == "zero":
    ablation_fn = lambda x: SparseAct(torch.zeros_like(x.act), torch.zeros_like(x.res))
elif args.ablation_fn == "mean":
    ablation_fn = lambda x: x.mean(dim=(0, 1)).expand_as(x)
elif args.ablation_fn == "id":
    ablation_fn = lambda x: x
else:
    raise ValueError(f"Unknown ablation function : {args.ablation_fn}")

class single_input_buffer:
    def __init__(self, model, batch_size, device, ctx_len=None, perm=None):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.ctx_len = ctx_len
        self.data = {
            "clean": ["When Mary and John went to the store, John gave a glass to"],
            "good": [[" Mary"]],
            "corr": ["When Mary and John went to the store, Paul gave a glass to"],
            "bad": [[" John"]],
        }
        self.done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        self.done = True
        tk = self.model.tokenizer
        clean_tokens = tk(self.data["clean"], return_tensors='pt', padding=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
        trg_idx = torch.zeros(clean_tokens.size(0), device=clean_tokens.device).long() - 1
        trg = []
        for i, good in enumerate(self.data["good"]):
            trg.append(tk(good, return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)[:, -1])
        corr_tokens = tk(self.data["corr"], return_tensors='pt', padding=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
        corr_trg = []
        for i, bad in enumerate(self.data["bad"]):
            corr_trg.append(tk(bad, return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)[:, -1])

        return {
            "clean": clean_tokens,
            "trg_idx": trg_idx,
            "trg": trg,
            "corr": corr_tokens,
            "corr_trg": corr_trg,
        }

# Set up parameters
metric_fn_dict = {
    'logit': metric_fn_logit,
    'KL': metric_fn_KL,
    'Statistical Distance': metric_fn_statistical_distance,
    # 'acc': metric_fn_acc,
    # 'MRR': metric_fn_MRR,
}
test_correctness = args.test_correctness
if not test_correctness:
    DATASET = ioi_buffer
else:
    DATASET = single_input_buffer
save_path = args.save_path \
    + ("ioi" if not test_correctness else "test_correctness") \
    + ("_" + args.ablation_fn) \
    + ("_without_corr/" if not args.use_corr else "/")
n_elts = args.n_elts
n_tests = args.n_tests
if DATASET == single_input_buffer:
    n_elts = 1
    n_tests = 1
elif DATASET == mixture_buffer:
    n_elts = 600
    n_tests = 600

batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
nb_eval_thresholds = args.nb_eval_thresholds
ctx_len = args.ctx_len
node_threshold = args.node_threshold
edge_threshold = args.edge_threshold
start_at_layer = args.start_at_layer

def circuit_to_device(nodes, edges, device):
    nodes = {k : v.to(device) if k != 'y' else None for k, v in nodes.items()}
    edges = {k : {kk : vv.to(device) for kk, vv in v.items()} for k, v in edges.items()}
    return nodes, edges

def add_circuit(tot_nodes, tot_edges, nodes, edges, factor=1):
    if tot_nodes is None:
        tot_nodes = {k : factor * v if k != 'y' else None for k, v in nodes.items()}
        tot_edges = {k : {kk : factor * vv for kk, vv in v.items() } for k, v in edges.items()}
    else:
        for k, effect in nodes.items():
            if k == 'y': continue
            tot_nodes[k] += factor * effect
        for k in edges.keys():
            for kk, effect in edges[k].items():
                tot_edges[k][kk] += factor * effect
    return tot_nodes, tot_edges

def normalize_circuit(tot_nodes, tot_edges, factor):
    tot_nodes = {k : v / factor if k != 'y' else None for k, v in tot_nodes.items()}
    tot_edges = {k : {kk : vv / factor for kk, vv in v.items()} for k, v in tot_edges.items()}
    return tot_nodes, tot_edges

def get_circuit(circuit_fn, device_id=0, perm=None, circuit_name=None):
    DEVICE = torch.device('cuda:{}'.format(device_id)) if torch.cuda.is_available() else torch.device('cpu')    

    # Set up the model and data
    print("Loading from gpu {}...".format(device_id))
    model, embed, resids, attns, mlps, name_dict = load_model_and_modules(DEVICE)
    dictionaries = load_saes(model, embed, resids, attns, mlps, device=DEVICE)
    print("Done.")

    tot_nodes = None
    tot_edges = None
    tot_inputs = 0

    buffer = DATASET(model, batch_size, DEVICE, ctx_len=None, perm=perm)

    # Compute the circuit
    
    for batch in tqdm(buffer):
        tokens, trg_idx, trg, corr, corr_trg = unpack_batch(batch)
        if not args.use_corr:
            corr = None
            corr_trg = None

        b = tokens.shape[0]
        tot_inputs += b

        nodes, edges = circuit_fn(
            clean=tokens,
            patch=corr,
            model=model,
            embed=embed,
            attns=attns,
            mlps=mlps,
            resids=resids,
            dictionaries=dictionaries,
            metric_fn=metric_fn_logit,
            metric_kwargs={"trg_idx": trg_idx, "trg_pos": trg, "trg_neg": corr_trg},
            ablation_fn=ablation_fn,
            nodes_only=False,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        tot_nodes, tot_edges = add_circuit(tot_nodes, tot_edges, nodes, edges, b)

    # normalize and return
    tot_nodes, tot_edges = normalize_circuit(tot_nodes, tot_edges, tot_inputs)

    save_circuit(
        save_path+f"circuit/{circuit_name}/{DEVICE.type}_{DEVICE.index}/",
        tot_nodes,
        tot_edges,
        0,   
    )

    return tot_nodes, tot_edges

def get_faithfulness(model, submodules, tot_nodes, tot_edges, node_ablation):
    buffer = DATASET(model, eval_batch_size, DEVICE)

    aggregated_outs = None
    n_batches = 0
    tot_inputs = 0

    for batch in tqdm(buffer):
        tokens, trg_idx, trg, corr, corr_trg = unpack_batch(batch)
        if not args.use_corr:
            corr = None
            corr_trg = None

        tot_inputs += tokens.shape[0]

        if tot_inputs > n_tests:
            break
        
        thresholds = torch.logspace(math.log10(node_threshold), 0, nb_eval_thresholds, 10).tolist()

        faithfulness = faithfulness_fn(
            model,
            submodules=submodules,
            sae_dict=dictionaries,
            name_dict=name_dict,
            clean=tokens,
            circuit=(tot_nodes, tot_edges),
            thresholds=thresholds,
            metric_fn=metric_fn_dict,
            metric_fn_kwargs={"trg_idx": trg_idx, "trg_pos": trg, "trg_neg": corr_trg},
            ablation_fn=ablation_fn,
            patch=corr,
            node_ablation=node_ablation,
        )

        if aggregated_outs is None:
            aggregated_outs = faithfulness
            continue

        for t, out in faithfulness.items():
            if t == 'complete' or t == 'empty':
                continue
            else:
                for fn_name in out['faithfulness']:                        
                    aggregated_outs[t]['faithfulness'][fn_name] += out['faithfulness'][fn_name]

        del faithfulness
        gc.collect()

    for t, out in aggregated_outs.items():
        if t == 'complete' or t == 'empty':
            continue
        for fn_name in out['faithfulness']:
            aggregated_outs[t]['faithfulness'][fn_name] /= n_batches

    return aggregated_outs

if __name__ == "__main__":
    ##########
    # Get Marks circuit
    ##########

    print("Getting Marks circuit...")

    if not os.path.exists(save_path+"circuit/marks/merged/0.pt"):
        perm = torch.randperm(n_elts)
        available_gpus = torch.cuda.device_count()
        elts_per_gpu = n_elts // available_gpus
        if elts_per_gpu == 0:
            elts_per_gpu = n_elts
            available_gpus = 1

        futures = []
        with ProcessPoolExecutor(max_workers=available_gpus) as executor:
            for i in range(available_gpus):
                futures.append(executor.submit(get_circuit, marks_circuit, i, perm[i*elts_per_gpu:(i+1)*elts_per_gpu], "marks"))
            
        tot_nodes = None
        tot_edges = None
        for future in futures:
            nodes, edges = future.result()
            nodes, edges = circuit_to_device(nodes, edges, DEVICE)
            tot_nodes, tot_edges = add_circuit(tot_nodes, tot_edges, nodes, edges)
        
        tot_nodes, tot_edges = normalize_circuit(tot_nodes, tot_edges, available_gpus)

        save_circuit(
            save_path+"circuit/marks/merged/",
            tot_nodes,
            tot_edges,
            0,
        )
    else:
        # Load the circuit :
        circuit_dict = torch.load(save_path+"circuit/marks/merged/0.pt")
        tot_nodes = circuit_dict["nodes"]
        tot_edges = circuit_dict["edges"]

    print("Done.")

    ##########
    # Evaluate with attn & mlps
    ##########

    model, embed, resids, attns, mlps, name_dict = load_model_and_modules(DEVICE)
    dictionaries = load_saes(model, embed, resids, attns, mlps, device=DEVICE)

    submodules = [embed] if start_at_layer == -1 else []
    i = 0
    for layer_modules in zip(mlps, attns, resids):
        if i >= start_at_layer:
            submodules.extend(layer_modules)
        i += 1

    print("Node evaluation with attn & mlps")
    aggregated_outs = get_faithfulness(model, submodules, tot_nodes, tot_edges, node_ablation=True)
    plot_faithfulness(aggregated_outs, save_path=save_path+'marks/node_ablation/all_submod/')
    print("Done.")

    print("Edge evaluation with attn & mlps")
    #aggregated_outs = get_faithfulness(model, submodules, tot_nodes, tot_edges, node_ablation=False)
    #plot_faithfulness(aggregated_outs, save_path=save_path+'marks/edge_ablation/all_submod/')
    print("Done.")

    ##########
    # Evaluate with resid only
    ##########

    submodules = [embed] if start_at_layer == -1 else []
    i = 0
    for resid in resids:
        if i >= start_at_layer:
            submodules.append(resid)
        i += 1

    print("Node evaluation with resid only")
    aggregated_outs = get_faithfulness(model, submodules, tot_nodes, tot_edges, node_ablation=True)
    plot_faithfulness(aggregated_outs, save_path=save_path+'marks/node_ablation/resid_only/')
    print("Done.")

    print("Edge evaluation with resid only")
    aggregated_outs = get_faithfulness(model, submodules, tot_nodes, tot_edges, node_ablation=False)
    plot_faithfulness(aggregated_outs, save_path=save_path+'marks/edge_ablation/resid_only/')
    print("Done.")

    ##########
    # Get Our circuit
    ##########

    print("Getting our circuit...")

    if not os.path.exists(save_path+"circuit/our/merged/0.pt"):
        perm = torch.randperm(n_elts)
        available_gpus = torch.cuda.device_count()
        elts_per_gpu = n_elts // available_gpus
        if elts_per_gpu == 0:
            elts_per_gpu = n_elts
            available_gpus = 1

        futures = []
        with ProcessPoolExecutor(max_workers=available_gpus) as executor:
            for i in range(available_gpus):
                futures.append(executor.submit(get_circuit, our_circuit, i, perm[i*elts_per_gpu:(i+1)*elts_per_gpu], "our"))

        tot_nodes = None
        tot_edges = None
        for future in futures:
            nodes, edges = future.result()
            nodes, edges = circuit_to_device(nodes, edges, DEVICE)
            tot_nodes, tot_edges = add_circuit(tot_nodes, tot_edges, nodes, edges)

        tot_nodes, tot_edges = normalize_circuit(tot_nodes, tot_edges, available_gpus)

        save_circuit(
            save_path+"circuit/our/merged/",
            tot_nodes,
            tot_edges,
            0,
        )
    else:
        # Load the circuit :
        circuit_dict = torch.load(save_path+"circuit/our/merged/0.pt")
        tot_nodes = circuit_dict["nodes"]
        tot_edges = circuit_dict["edges"]
    
    print("Done.")

    ##########
    # Evaluate with resid only
    ##########

    submodules = [embed] if start_at_layer == -1 else []
    i = 0
    for resid in resids:
        if i >= start_at_layer:
            submodules.append(resid)
        i += 1

    print("Node evaluation")
    aggregated_outs = get_faithfulness(model, submodules, tot_nodes, tot_edges, node_ablation=True)
    plot_faithfulness(aggregated_outs, save_path=save_path+'us/node_ablation/resid_only/')
    print("Done.")

    print("Edge evaluation")
    aggregated_outs = get_faithfulness(model, submodules, tot_nodes, tot_edges, node_ablation=False)
    plot_faithfulness(aggregated_outs, save_path=save_path+'us/edge_ablation/resid_only/')
    print("Done.")
