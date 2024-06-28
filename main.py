"""
For future usage with different kinds of SAEs (like LIB, gated, etc.), nothing has to change except load_saes function.
Also, currently the code is specific to pythia-70m-deduped model, but it can be trivially adapted to other models.

Example command to run this code :

python main.py --test_correctness -da -et 0.1 -sp /scratch/pyllm/dhimoila/output/test_correctness_jack/&

python main.py -gc -da -ec -cbs 2 -et 0.1 -cml 10 -eml 10 -sp /scratch/pyllm/dhimoila/output/test_correctness/&

python main.py -ec -eml 10 -sp /scratch/pyllm/dhimoila/output/120624_01/esal0/ -cp /scratch/pyllm/dhimoila/output/120624_01/circuit/merged/ -esal 0&
"""

##########
# Args
##########

if __name__ == "__main__":
    print("Starting...")
    print("Parsing arguments...", end="")

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--test_correctness", action="store_true", help="Test the correctness of the code. Run a minimal version on one sample.")
parser.add_argument("--profile", action="store_true", help="Run the code with cProfile (withouth multiprocessing)")
parser.add_argument("--get_circuit", "-gc", action="store_true", help="Run the first step - generate the circuit")
parser.add_argument("--dump_all", "-da", action="store_true", help="Dump all example circuits on disk")
parser.add_argument("--eval_circuit", "-ec", action="store_true", help="Run the second step - evaluate the circuit")
parser.add_argument("--marks_version", "-mv", action="store_true", help="Use marks original code")

parser.add_argument("--identity_dict", "-id", action="store_true", help="Use identity dictionaries instead of SAEs")
parser.add_argument("--SVD_dict", "-svd", action="store_true", help="Use SVD dictionaries instead of SAEs")

parser.add_argument("--node_threshold", "-nt", type=float, default=0.1)
parser.add_argument("--edge_threshold", "-et", type=float, default=0.1)

parser.add_argument("--ctx_len", "-cl", type=int, default=16, help="Maximum sequence lenght of example sequences")

parser.add_argument("--circuit_method", "-cm", type=str, default="resid", help="Method to build the circuit. Available : resid, marks, resid_topk (use to your own risk)")
parser.add_argument("--aggregation", "-agg", type=str, default="sum", help="Method to aggregate graphs across examples. Available : sum, max")
parser.add_argument("--circuit_batch_size", "-cbs", type=int, default=1, help="Batch size for circuit building. You can increase this until you run out of memory.")
parser.add_argument("--circuit_max_loop", "-cml", type=int, default=1000, help="Maximum number of example batches to process when building the graph.")

parser.add_argument("--eval_method", "-em", type=str, default="edge_patching", help="Method to evaluate the circuit. Available : node_patching, edge_patching. Node patching is much faster but does not evaluate the same graph as the one built in the first step.")
parser.add_argument("--eval_metric", "-emt", type=str, nargs='+', default=["logit", "KL", "acc", "MRR"], help="Metric to evaluate the circuit. Available : logit, KL")
parser.add_argument("--eval_batch_size", "-ebs", type=int, default=100, help="Batch size for circuit evaluation.")
parser.add_argument("--eval_max_loop", "-eml", type=int, default=100, help="Maximum number of example batches to process when evaluating the circuit.")
parser.add_argument("--eval_start_at_layer", "-esal", type=int, default=1, help="At what layer to start evaluating the circuit. -1 means the whole circuit will be evaluated. Will run the original model up to the specified layer and then evaluate the circuit.")

parser.add_argument("--save_path", "-sp", type=str, default='/scratch/pyllm/dhimoila/output/', help="Path to save the outputs.")
parser.add_argument("--circuit_path", "-cp", type=str, default=None, help="Path to load the circuit from. Not necessary if running the first step.")

args = parser.parse_args()

available_cm = ["resid", "marks", "resid_topk"]
available_em = ["node_patching", "edge_patching"]

run_step_1 = args.get_circuit
dump_all = args.dump_all
run_step_2 = args.eval_circuit
marks_version = args.marks_version

idd = args.identity_dict
svd = args.SVD_dict
cm = args.circuit_method
cbs = args.circuit_batch_size
em = args.eval_method
ebs = args.eval_batch_size
esal = args.eval_start_at_layer
save_path = args.save_path
circuit_path = args.circuit_path

if marks_version:
    cm = "marks"
    em = "node_patching"

if cm not in available_cm:
    raise ValueError(f"--circuit_method must be in {available_cm}. Got {cm}")
if em not in available_em:
    raise ValueError(f"--eval_method must be in {available_em}. Got {em}")

if (not run_step_1) and run_step_2 and circuit_path is None:
    raise ValueError("--circuit_path must be provided if running only evaluation")

if run_step_1 and circuit_path is None:
    circuit_path = save_path + "circuit/"

if __name__ == "__main__":
    print("Done.")

##########
# Imports
##########

if __name__ == "__main__":
    print("Importing...", end="")

import os
import gc
import pickle
import math
import importlib

import cProfile
import pstats

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

multiprocessing.set_start_method('spawn', force=True)

import torch
from nnsight import LanguageModel
from datasets import load_dataset

from transformers import logging
logging.set_verbosity_error()

from tqdm import tqdm, trange

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate

import networkx as nx

from dictionary_learning import AutoEncoder
from dictionary_learning.dictionary import IdentityDict, SVDDictionary
from utils import SparseAct
from buffer import TokenBuffer
import evaluation
from circuit import get_circuit
from utils import save_circuit, load_circuit, load_latest, sparse_coo_maximum
from ablation import run_with_ablations

from matplotlib import pyplot as plt

if __name__ == "__main__":
    print("Done.")

##########
# Helper functions
##########

def run_main(device_id, step_1, step_2):
    DEVICE = torch.device('cuda:{}'.format(device_id)) if torch.cuda.is_available() else torch.device('cpu')
    def metric_fn_logit(model, trg=None):
        """
        default : return the logit
        """
        if trg is None:
            raise ValueError("trg must be provided")
        return model.embed_out.output[torch.arange(trg[0].numel()), trg[0], trg[1]]

    def metric_fn_KL(model, trg=None, clean_logits=None):
        if clean_logits is None:
            raise ValueError("clean_logits must be provided")
        logits = model.embed_out.output[torch.arange(trg[0].numel()), trg[0]] # (b, s, d_model) -> (b, d_model)
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=-1),
            torch.nn.functional.log_softmax(clean_logits, dim=-1),
            reduction='none',
            log_target=True
        ).sum(dim=-1)
    
    def metric_fn_acc(model, trg=None):
        """
        default : return 1 if the model's prediction is correct, 0 otherwise
        """
        if trg is None:
            raise ValueError("trg must be provided")
        logits = model.embed_out.output[torch.arange(trg[0].numel()), trg[0]]
        return (logits.argmax(dim=-1) == trg[1]).float()
    
    def metric_fn_MRR(model, trg=None):
        """
        default : return 1/rank of the correct answer
        """
        if trg is None:
            raise ValueError("trg must be provided")
        logits = model.embed_out.output[torch.arange(trg[0].numel()), trg[0]]
        return 1 / (1 + (logits.argsort(dim=-1, descending=True) == trg[1].unsqueeze(-1)).argmax(dim=-1).float())

    metric_fn_dict = {}
    if "KL" in args.eval_metric:
        metric_fn_dict["KL"] = metric_fn_KL
    if "logit" in args.eval_metric:
        metric_fn_dict["logit"] = metric_fn_logit
    # if "acc" in args.eval_metric:
    #     metric_fn_dict["acc"] = metric_fn_acc
    # if "MRR" in args.eval_metric:
    #     metric_fn_dict["MRR"] = metric_fn_MRR

    def load_model_and_modules(): 
        pythia70m = LanguageModel(
            "EleutherAI/pythia-70m-deduped",
            device_map=DEVICE,
            dispatch=True,
        )

        pythia70m_embed = pythia70m.gpt_neox.embed_in

        pythia70m_resids = []
        pythia70m_attns = []
        pythia70m_mlps = []
        for layer in range(len(pythia70m.gpt_neox.layers)):
            pythia70m_resids.append(pythia70m.gpt_neox.layers[layer])
            pythia70m_attns.append(pythia70m.gpt_neox.layers[layer].attention)
            pythia70m_mlps.append(pythia70m.gpt_neox.layers[layer].mlp)

        submod_names = {
            pythia70m.gpt_neox.embed_in : 'embed'
        }
        for i in range(len(pythia70m.gpt_neox.layers)):
            submod_names[pythia70m.gpt_neox.layers[i].attention] = f'attn_{i}'
            submod_names[pythia70m.gpt_neox.layers[i].mlp] = f'mlp_{i}'
            submod_names[pythia70m.gpt_neox.layers[i]] = f'resid_{i}'

        return pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names
    
    def load_saes(
        model,
        model_embed,
        model_resids,
        model_attns,
        model_mlps,
    ):
        dictionaries = {}

        d_model = 512
        dict_size = 32768 if not svd else 512

        if idd:
            dictionaries[model_embed] = IdentityDict(d_model)
            for layer in range(len(model.gpt_neox.layers)):
                dictionaries[model_resids[layer]] = IdentityDict(d_model)
                dictionaries[model_attns[layer]] = IdentityDict(d_model)
                dictionaries[model_mlps[layer]] = IdentityDict(d_model)

            return dictionaries
        
        base = '/scratch/pyllm/dhimoila/'
        path = base + "dictionaires/pythia-70m-deduped/" + ("SVDdicts/" if svd else "")

        ae = AutoEncoder(d_model, dict_size).to(DEVICE)
        ae.load_state_dict(torch.load(path + f"embed/ae.pt", map_location=DEVICE))
        dictionaries[model_embed] = ae


        for layer in range(len(model.gpt_neox.layers)):
            ae = AutoEncoder(d_model, dict_size).to(DEVICE)
            ae.load_state_dict(torch.load(path + f"resid_out_layer{layer}/ae.pt", map_location=DEVICE))
            dictionaries[model_resids[layer]] = ae

            ae = AutoEncoder(d_model, dict_size).to(DEVICE)
            ae.load_state_dict(torch.load(path + f"attn_out_layer{layer}/ae.pt", map_location=DEVICE))
            dictionaries[model_attns[layer]] = ae

            ae = AutoEncoder(d_model, dict_size).to(DEVICE)
            ae.load_state_dict(torch.load(path + f"mlp_out_layer{layer}/ae.pt", map_location=DEVICE))
            dictionaries[model_mlps[layer]] = ae
        
        return dictionaries
    
    # TODO : plot distrib of trg tokens to see if stupid or not
    def get_buffer(
        model,
        batch_size,
    ):
        dataset = load_dataset(
            "wikipedia",
            language="en",
            date="20240401",
            split="train",
            streaming=True,
            trust_remote_code=True
        ).shuffle()
        dataset = iter(dataset)

        buffer = TokenBuffer(
            dataset,
            model,
            n_ctxs=10,
            ctx_len=args.ctx_len,
            load_buffer_batch_size=10,
            return_batch_size=batch_size,
            device=DEVICE,
            max_number_of_yields=2**20,
            discard_bos=True
        )

        return buffer

    def marks_get_fcs(
        model,
        circuit,
        clean,
        trg_idx,
        trg,
        submodules,
        dictionaries,
        ablation_fn,
        thresholds,
        metric_fn,
        dict_size,
        submod_names,
        handle_errors = 'default', # also 'remove' or 'resid_only'
    ):
        # get m(C) for the circuit obtained by thresholding nodes with the given threshold
        clean_inputs = clean

        # def metric_fn(model):
        #     return (
        #         - t.gather(model.embed_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) + \
        #         t.gather(model.embed_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
        #     )
        def metric_fn_(model):
            return metric_fn(model, trg=(trg_idx, trg))
        
        circuit = circuit[0]

        with torch.no_grad():
            out = {}

            # get F(M)
            with model.trace(clean_inputs):
                metric = metric_fn_(model).save()
                
            fm = metric.value.mean().item()

            out['fm'] = fm

            # get m(∅)
            fempty = run_with_ablations(
                clean_inputs,
                None,
                model,
                submodules,
                dictionaries,
                nodes = {
                    submod : SparseAct(
                        act=torch.zeros(dict_size, dtype=torch.bool), 
                        resc=torch.zeros(1, dtype=torch.bool)).to(DEVICE)
                    for submod in submodules
                },
                metric_fn=metric_fn_,
                ablation_fn=ablation_fn,
            ).mean().item()
            out['fempty'] = fempty

            for threshold in thresholds:
                out[threshold] = {}
                nodes = {
                    submod : circuit[submod_names[submod]].abs() > threshold for submod in submodules
                }

                if handle_errors == 'remove':
                    for k in nodes: nodes[k].resc = torch.zeros_like(nodes[k].resc, dtype=torch.bool)
                elif handle_errors == 'resid_only':
                    for k in nodes:
                        if k not in model.gpt_neox.layers: nodes[k].resc = torch.zeros_like(nodes[k].resc, dtype=torch.bool)

                n_nodes = sum([n.act.sum() + n.resc.sum() for n in nodes.values()]).item()
                out[threshold]['n_nodes'] = n_nodes
                
                out[threshold]['fc'] = run_with_ablations(
                    clean_inputs,
                    None,
                    model,
                    submodules,
                    dictionaries,
                    nodes=nodes,
                    metric_fn=metric_fn_,
                    ablation_fn=ablation_fn,
                ).mean().item()
                out[threshold]['fccomp'] = run_with_ablations(
                    clean_inputs,
                    None,
                    model,
                    submodules,
                    dictionaries,
                    nodes=nodes,
                    metric_fn=metric_fn_,
                    ablation_fn=ablation_fn,
                    complement=True
                ).mean().item()
                out[threshold]['faithfulness'] = (out[threshold]['fc'] - fempty) / (fm - fempty)
                out[threshold]['completeness'] = (out[threshold]['fccomp'] - fempty) / (fm - fempty)
                
        return out

    def marks_plot_faithfulness(outs, thresholds):
        # plot faithfulness results
        fig = go.Figure()

        colors = {
            'features' : 'blue',
            'features_wo_errs' : 'red',
            'features_wo_some_errs' : 'green',
            'neurons' : 'purple',
            # 'random_features' : 'black'
        }

        y_min = 0
        y_max = 1
        for setting, subouts in outs.items():
            x_min = max([min(subouts[t]['n_nodes'] for t in thresholds)]) + 1
            x_max = min([max(subouts[t]['n_nodes'] for t in thresholds)]) - 1
            fs = {
                "ioi" : interpolate.interp1d([subouts[t]['n_nodes'] for t in thresholds], [subouts[t]['faithfulness'] for t in thresholds])
            }
            xs = torch.logspace(math.log10(x_min), math.log10(x_max), 100, 10).tolist()

            fig.add_trace(go.Scatter(
                x = [subouts[t]['n_nodes'] for t in thresholds],
                y = [subouts[t]['faithfulness'] for t in thresholds],
                mode='lines', line=dict(color=colors[setting]), opacity=0.17, showlegend=False
            ))

            y_min = min(y_min, min([subouts[t]['faithfulness'] for t in thresholds]))
            y_max = max(y_max, max([subouts[t]['faithfulness'] for t in thresholds]))

            fig.add_trace(go.Scatter(
                x=xs,
                y=[ sum([f(x) for f in fs.values()]) / len(fs) for x in xs ],
                mode='lines', line=dict(color=colors[setting]), name=setting
            ))

        fig.update_xaxes(type="log", range=[math.log10(x_min), math.log10(x_max)])
        fig.update_yaxes(range=[y_min, min(y_max, 2)])

        fig.update_layout(
            xaxis_title='Nodes',
            yaxis_title='Faithfulness',
            width=800,
            height=375,
            # set white background color
            plot_bgcolor='rgba(0,0,0,0)',
            # add grey gridlines
            yaxis=dict(gridcolor='rgb(200,200,200)',mirror=True,ticks='outside',showline=True),
            xaxis=dict(gridcolor='rgb(200,200,200)', mirror=True, ticks='outside', showline=True),

        )

        # fig.show()
        fig.write_image('faithfulness.pdf')

    def plot_faithfulness(outs):
        """
        plot faithfulness results

        Plot all w.r.t. thresholds, and plot line for complete and empty graphs

        TODO : Plot edges w.r.t. nodes, and other plots if needed
        """

        thresholds = []
        n_nodes = []
        n_edges = []
        avg_deg = []
        density = []
        # modularity = []
        # z_score = []
        faithfulness = [[] for _ in metric_fn_dict]
        complete = []
        empty = []

        for t in outs:
            if t == 'complete':
                for i, fn_name in enumerate(metric_fn_dict):
                    complete.append(outs[t][fn_name])
                continue
            if t == 'empty':
                for i, fn_name in enumerate(metric_fn_dict):
                    empty.append(outs[t][fn_name])
                continue
            thresholds.append(t)
            n_nodes.append(outs[t]['n_nodes'])
            n_edges.append(outs[t]['n_edges'])
            avg_deg.append(outs[t]['avg_deg'])
            density.append(outs[t]['density'])
            # modularity.append(outs[t]['modularity'])
            # z_score.append(outs[t]['z_score'])
            for i, fn_name in enumerate(metric_fn_dict):
                faithfulness[i].append(outs[t]['faithfulness'][fn_name])

        fig = make_subplots(
            rows=4 + len(list(metric_fn_dict.keys())),
            cols=1,
        )

        for i, fn_name in enumerate(metric_fn_dict):
            print(faithfulness[i])
            fig.add_trace(go.Scatter(
                    x=thresholds,
                    y=faithfulness[i],
                    mode='lines+markers',
                    #title_text=fn_name+" faithfulness vs threshold",
                    name=fn_name,
                ),
                row=i+1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=n_nodes,
                mode='lines+markers',
                #title_text="n_nodes vs threshold",
                name='n_nodes',
            ),
            row=len(list(metric_fn_dict.keys()))+1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=n_edges,
                mode='lines+markers',
                #title_text="n_edges vs threshold",
                name='n_edges',
            ),
            row=len(list(metric_fn_dict.keys()))+2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=avg_deg,
                mode='lines+markers',
                #title_text="avg_deg vs threshold",
                name='avg_deg',
            ),
            row=len(list(metric_fn_dict.keys()))+3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=density,
                mode='lines+markers',
                #title_text="density vs threshold",
                name='density',
            ),
            row=len(list(metric_fn_dict.keys()))+4, col=1
        )

        # Update x-axes to log scale
        fig.update_xaxes(type="log")

        # default layout is : height=600, width=800. We want to make it a bit bigger so that each plot has the original size
        fig.update_layout(
            height=600 + 400 * (4 + len(list(metric_fn_dict.keys()))),
            width=800,
            title_text="Faithfulness and graph properties w.r.t. threshold",
            showlegend=True,
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.write_html(save_path + "faithfulness.html")

    ##########
    # Main functions
    ##########

    def run_test_correctness():
        pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules()

        dictionaries = load_saes(
            pythia70m,
            pythia70m_embed,
            pythia70m_resids,
            pythia70m_attns,
            pythia70m_mlps,
        )

        buffer = get_buffer(pythia70m, 1)
        tokens, trg_idx, trg = next(buffer)

        circuit = get_circuit(
            tokens,
            None,
            model=pythia70m,
            embed=pythia70m_embed,
            attns=pythia70m_attns,
            mlps=pythia70m_mlps,
            resids=pythia70m_resids,
            dictionaries=dictionaries,
            metric_fn=metric_fn_logit,
            metric_kwargs={"trg": (trg_idx, trg)},
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            method=cm,
            aggregation=args.aggregation,
            dump_all=dump_all,
            save_path=save_path,
        )

        faithfulness = evaluation.faithfulness(
            pythia70m,
            submodules=[pythia70m_embed],
            sae_dict=dictionaries,
            name_dict=submod_names,
            clean=tokens,
            circuit=circuit,
            thresholds=[args.node_threshold],
            metric_fn=metric_fn_dict,
            metric_fn_kwargs={"trg": (trg_idx, trg)},
            patch=None,
            default_ablation='mean',
            get_graph_info=True,
        )

        plot_faithfulness(faithfulness)

    ##########
    # This function builds the computational graph.
    ##########
    def run_first_step():
        pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules()

        dictionaries = load_saes(
            pythia70m,
            pythia70m_embed,
            pythia70m_resids,
            pythia70m_attns,
            pythia70m_mlps,
        )

        buffer = get_buffer(pythia70m, cbs)
        
        tot_circuit = None
        
        i = 0
        max_loop = args.circuit_max_loop
        edge_threshold = args.edge_threshold
        node_threshold = args.node_threshold

        for tokens, trg_idx, trg in tqdm(buffer):
            if i >= max_loop:
                break
            i += 1
            circuit = get_circuit(
                tokens,
                None,
                model=pythia70m,
                embed=pythia70m_embed,
                attns=pythia70m_attns,
                mlps=pythia70m_mlps,
                resids=pythia70m_resids,
                dictionaries=dictionaries,
                metric_fn=metric_fn_logit,
                metric_kwargs={"trg": (trg_idx, trg)},
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                method=cm,
                aggregation=args.aggregation,
                dump_all=dump_all,
                save_path=save_path,
            )
            if tot_circuit is None:
                tot_circuit = circuit
            else:
                for k, v in circuit[0].items():
                    if v is not None:
                        if type(v) == SparseAct:
                            tot_circuit[0][k] = SparseAct.maximum(tot_circuit[0][k], v)
                        else:
                            tot_circuit[0][k] = torch.maximum(tot_circuit[0][k], v)
                for ku, vu in circuit[1].items():
                    for kd, vd in vu.items():
                        if vd is not None:
                            tot_circuit[1][ku][kd] = sparse_coo_maximum(tot_circuit[1][ku][kd], vd)
            
            if i % 10 == 0:
                save_circuit(
                    save_path + "circuit/" + f"{DEVICE.type}_{DEVICE.index}/",
                    tot_circuit[0],
                    tot_circuit[1],
                    i * cbs,
                    dataset_name="",
                    model_name="",
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                )
            
            del circuit
            gc.collect()

    ##########
    # This function is only slightly modified from the original code of marks et al.
    # It plots the faithfulness of the model w.r.t. the number of nodes in the circuit.
    # It uses node patching, so it evaluates a different graph than the one built in the first step...
    ##########
    def run_second_step_marks_version():

        def metric_fn(model, trg=None):
            """
            default : return the logit
            """
            if trg is None:
                raise ValueError("trg must be provided")
            return model.embed_out.output[torch.arange(trg[0].numel()), trg[0], trg[1]]

        ablation_fn = lambda x: x.mean(dim=0).expand_as(x)
        
        pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules()
        
        start_at_layer = esal
        submodules = [pythia70m_embed] if start_at_layer == -1 else []
        for i in range(max(start_at_layer, 0), len(pythia70m.gpt_neox.layers)):
            submodules.append(pythia70m_attns[i])
            submodules.append(pythia70m_mlps[i])
            submodules.append(pythia70m_resids[i])
        
        feat_dicts = load_saes(
            pythia70m,
            pythia70m_embed,
            pythia70m_resids,
            pythia70m_attns,
            pythia70m_mlps,
        )

        circuit = load_latest(circuit_path, device=DEVICE)
        
        i = 0
        max_loop = args.eval_max_loop
        dict_size = 32768

        buffer = get_buffer(pythia70m)
        
        aggregated_outs = {
            'features' : {
                'fm' : 0,
                'fempty' : 0,
            },
            'features_wo_errs' : {
                'fm' : 0,
                'fempty' : 0,
            },
            'features_wo_some_errs' : {
                'fm' : 0,
                'fempty' : 0,
            }
        }
        for tokens, trg_idx, trg in tqdm(buffer):
            if i >= max_loop:
                break
            i += 1
            #thresholds = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
            thresholds = torch.logspace(-6, 2, 20, 10).tolist()
            outs = {
                'features' :
                    marks_get_fcs(
                        pythia70m,
                        circuit,
                        tokens,
                        trg_idx,
                        trg,
                        submodules,
                        feat_dicts,
                        ablation_fn=ablation_fn,
                        thresholds = thresholds,
                        metric_fn=metric_fn,
                        dict_size=dict_size,
                        submod_names=submod_names,
                    ),
                'features_wo_errs' :
                    marks_get_fcs(
                        pythia70m,
                        circuit,
                        tokens,
                        trg_idx,
                        trg,
                        submodules,
                        feat_dicts,
                        ablation_fn=ablation_fn,
                        thresholds = thresholds,
                        handle_errors='remove'
                    ),
                'features_wo_some_errs' :
                    marks_get_fcs(
                        pythia70m,
                        circuit,
                        tokens,
                        trg_idx,
                        trg,
                        submodules,
                        feat_dicts,
                        ablation_fn=ablation_fn,
                        thresholds = thresholds,
                        handle_errors='resid_only'
                    )
            }

            for setting, subouts in outs.items():
                for t, out in subouts.items():
                    if t not in aggregated_outs[setting]:
                        aggregated_outs[setting][t] = {}
                    if t == 'fm' or t == 'fempty':
                        aggregated_outs[setting][t] += out
                    else:
                        for k, v in out.items():
                            if k not in aggregated_outs[setting][t]:
                                aggregated_outs[setting][t][k] = 0
                            aggregated_outs[setting][t][k] += v
            
            del outs
            gc.collect()

        for setting, subouts in aggregated_outs.items():
            for t, out in subouts.items():
                if t == 'fm' or t == 'fempty':
                    aggregated_outs[setting][t] /= i
                else:
                    for k, v in out.items():
                        aggregated_outs[setting][t][k] /= i

        marks_plot_faithfulness(aggregated_outs, thresholds)

    ##########
    # This function evaluates the circuit built in the first step.
    ##########
    def run_second_step():
        pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules()
        
        start_at_layer = esal
        submodules = [pythia70m_embed] if start_at_layer == -1 else []
        for i in range(max(start_at_layer, 0), len(pythia70m.gpt_neox.layers)):
            submodules.append(pythia70m_resids[i])
        
        dictionaries = load_saes(
            pythia70m,
            pythia70m_embed,
            pythia70m_resids,
            pythia70m_attns,
            pythia70m_mlps,
        )

        circuit = load_latest(circuit_path, device=DEVICE)
        
        i = 0
        max_loop = args.eval_max_loop

        buffer = get_buffer(pythia70m, ebs)
        
        aggregated_outs = None
        for tokens, trg_idx, trg in tqdm(buffer):
            if i >= max_loop:
                break
            i += 1
            #thresholds = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
            thresholds = torch.logspace(-2, 2, 15, 10).tolist()
            faithfulness = evaluation.faithfulness(
                pythia70m,
                submodules=submodules,
                sae_dict=dictionaries,
                name_dict=submod_names,
                clean=tokens,
                circuit=circuit,
                thresholds=thresholds,
                metric_fn=metric_fn_dict,
                metric_fn_kwargs={"trg": (trg_idx, trg)},
                patch=None,
                default_ablation='mean',
                get_graph_info=(i <= 1),
            )
            if aggregated_outs is None:
                aggregated_outs = faithfulness
                continue

            for t, out in faithfulness.items():
                if t == 'complete' or t == 'empty':
                    for fn_name in metric_fn_dict:
                        if fn_name not in aggregated_outs[t]:
                            aggregated_outs[t][fn_name] = 0
                        aggregated_outs[t][fn_name] += out[fn_name]
                else:
                    for fn_name in metric_fn_dict:                        
                        aggregated_outs[t]['faithfulness'][fn_name] += out['faithfulness'][fn_name]
            
            del faithfulness
            gc.collect()
            
            plot_faithfulness(aggregated_outs)

        for t, out in aggregated_outs.items():
            if t == 'complete' or t == 'empty':
                for fn_name in metric_fn_dict:
                    aggregated_outs[t][fn_name] /= i
            else:
                for fn_name in metric_fn_dict:
                    aggregated_outs[t]['faithfulness'][fn_name] /= i

        plot_faithfulness(aggregated_outs)

    if args.test_correctness:
        print("\nRunning test correctness. This will ignore all other arguments.\n")
        run_test_correctness()
        print("\nDone running test correctness.\n")
        return

    if step_1:
        print("\nRunning step 1 on device {}...\n".format(device_id))
        run_first_step()
        print("\nDone running step 1 on device {}.\n".format(device_id))
    if step_2:
        if marks_version:
            print("\nRunning step 2 (Marks version)...\n")
            run_second_step_marks_version()
            print("\nDone running step 2 (Marks version).\n")
        else:
            print("\nRunning step 2...\n")
            run_second_step()
            print("\nDone running step 2.\n")

if __name__ == "__main__":
    import time
    available_gpus = torch.cuda.device_count()

    if args.test_correctness:
        run_main(0, False, False)
        exit()

    if run_step_1:
        print("Running step 1 on {} devices...".format(available_gpus))
        def run():
            if args.profile:
                run_main(0, True, False)
                return
            else:
                futures = []
                # use process pool executor to run one instance of run_main(d, True, False) for each device in parallel
                with ProcessPoolExecutor(max_workers=available_gpus) as executor:
                    for i in range(available_gpus):
                        futures.append(executor.submit(run_main, i, True, False))
                        # this is just to avoid stupid crashes when accessing wikipedia...
                        time.sleep(10)
                for future in futures:
                    future.result()
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if args.profile:
            cProfile.run('run()', save_path + 'profile_step1.txt')
        else:
            run()

    if run_step_2:
        print("Running step 2...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if args.profile:
            cProfile.run('run_main(0, False, True)', save_path + 'profile_step2.txt')
        else:
            run_main(0, False, True)
        print("Done")