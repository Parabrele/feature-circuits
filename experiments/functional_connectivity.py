##########
# Args
##########

if __name__ == "__main__":
    print("Starting...")
    print("Parsing arguments...", end="")

from argparse import ArgumentParser

import evaluation.faithfulness

parser = ArgumentParser()

parser.add_argument("--identity_dict", "-id", action="store_true", help="Use identity dictionaries")
parser.add_argument("--SVD_dict", "-svd", action="store_true", help="Use SVD dictionaries")
parser.add_argument("--White_dict", "-white", action="store_true", help="Use Whitening dictionaries")
parser.add_argument("--SAE", action="store_true", help="Use SAE dictionaries")

parser.add_argument("--ROI", "-roi", action="store_true", help="Use Regions of Interest -attn heads and MLP layers- instead of neurons - canonical dimensions in dictionary space for the given dictionaries.")

parser.add_argument("--activation", "-act", action="store_true", help="Compute activations")
parser.add_argument("--attribution", "-attr", action="store_true", help="Compute attributions")
parser.add_argument("--use_resid", "-resid", action="store_true", help="Use residual stream nodes instead of modules.")

parser.add_argument("--n_batches", "-nb", type=int, default=1000, help="Number of batches to process.")
parser.add_argument("--batch_size", "-bs", type=int, default=1, help="Number of examples to process in one go.")
parser.add_argument("--steps", type=int, default=10, help="Number of steps to compute the attributions (precision of Integrated Gradients).")

parser.add_argument("--node_threshold", "-nt", type=float, default=0.1)
parser.add_argument("--edge_threshold", "-et", type=float, default=0.1)

parser.add_argument("--aggregation", "-agg", type=str, default="sum", help="Method to aggregate graphs across examples. Available : sum, max")

parser.add_argument("--ctx_len", "-cl", type=int, default=16, help="Maximum sequence lenght of example sequences")

parser.add_argument("--save_path", "-sp", type=str, default='/scratch/pyllm/dhimoila/output/', help="Path to save the outputs.")

args = parser.parse_args()

if not sum([args.identity_dict, args.SVD_dict, args.White_dict, args.SAE]) == 1:
    raise ValueError("Exactly one of --identity_dict, --SVD_dict, --White_dict, --SAE must be provided.")

idd = args.identity_dict
svd = args.SVD_dict
white = args.White_dict
sae = args.SAE

roi = args.ROI

act = args.activation
attr = args.attribution
use_resid = args.use_resid

n_batches = args.n_batches
batch_size = args.batch_size
steps = args.steps

node_threshold = args.node_threshold
edge_threshold = args.edge_threshold

aggregation = args.aggregation

ctx_len = args.ctx_len

save_path = args.save_path

if __name__ == "__main__":
    print("Done.")

##########
# Imports
##########

if __name__ == "__main__":
    print("Importing...")

import os
import gc
import math

import cProfile

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

multiprocessing.set_start_method('spawn', force=True)

import torch
from nnsight import LanguageModel
from datasets import load_dataset

from transformers import logging
logging.set_verbosity_error()

from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate

from data.wikipedia import get_buffer
import evaluation
from connectivity.effective import get_circuit
from ablation.node_ablation import run_with_ablations

from utils.dictionary import AutoEncoder, IdentityDict
from utils.activation import SparseAct
from utils.savior import save_circuit, load_latest
from utils.graph_utils import merge_circuits
from utils.metric_fns import metric_fn_KL, metric_fn_logit, metric_fn_acc, metric_fn_MRR
from utils.experiments_setup import load_model_and_modules, load_saes

if __name__ == "__main__":
    print("Done.")

##########
# Helper functions
##########

def run_main(device_id, step_1, step_2):
    DEVICE = torch.device('cuda:{}'.format(device_id)) if torch.cuda.is_available() else torch.device('cpu')

    metric_fn_dict = {}
    if "KL" in args.eval_metric:
        metric_fn_dict["KL"] = metric_fn_KL
    if "logit" in args.eval_metric:
        metric_fn_dict["logit"] = metric_fn_logit
    if "acc" in args.eval_metric:
        metric_fn_dict["acc"] = metric_fn_acc
    if "MRR" in args.eval_metric:
        metric_fn_dict["MRR"] = metric_fn_MRR
    
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

    ##########
    # This function builds the computational graph.
    ##########
    def run_first_step():
        pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules(DEVICE)

        dictionaries = load_saes(
            pythia70m,
            pythia70m_embed,
            pythia70m_resids,
            pythia70m_attns,
            pythia70m_mlps,
            idd=idd,
            svd=svd,
            device=DEVICE,
        )

        buffer = get_buffer(pythia70m, cbs, DEVICE, args.ctx_len)
        
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
            merge_circuits(tot_circuit, circuit)
            
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
        pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules(DEVICE)
        
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
            idd=idd,
            svd=svd,
            device=DEVICE,
        )

        circuit = load_latest(circuit_path, device=DEVICE)
        
        i = 0
        max_loop = args.eval_max_loop

        buffer = get_buffer(pythia70m, ebs, DEVICE, args.ctx_len)
        
        aggregated_outs = None
        for tokens, trg_idx, trg in tqdm(buffer):
            if i >= max_loop:
                break
            i += 1
            thresholds = torch.logspace(-2, 2, 15, 10).tolist()
            faithfulness = evaluation.faithfulness.faithfulness(
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