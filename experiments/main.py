"""
For future usage with different kinds of SAEs (like LIB, gated, etc.), nothing has to change except load_saes function.
Also, currently the some part of the code is specific to pythia-70m-deduped model, but it can be trivially adapted to other models.

Example command to run this code :

python -m experiments.main -gc -ec -et 0.1 -cbs 2 -cml 8 -ebs 2 -eml 10 --dataset ioi -sp /scratch/pyllm/dhimoila/output/ioi/&

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

parser.add_argument("--profile", action="store_true", help="Run the code with cProfile (withouth multiprocessing)")
parser.add_argument("--get_circuit", "-gc", action="store_true", help="Run the first step - generate the circuit")
parser.add_argument("--dump_all", "-da", action="store_true", help="Dump all example circuits on disk")
parser.add_argument("--eval_circuit", "-ec", action="store_true", help="Run the second step - evaluate the circuit")
parser.add_argument("--marks_version", "-mv", action="store_true", help="Use marks original code")

parser.add_argument("--identity_dict", "-id", action="store_true", help="Use identity dictionaries instead of SAEs")
parser.add_argument("--SVD_dict", "-svd", action="store_true", help="Use SVD dictionaries instead of SAEs")

parser.add_argument("--node_threshold", "-nt", type=float, default=0.1)
parser.add_argument("--edge_threshold", "-et", type=float, default=0.1)

parser.add_argument("--dataset", type=str, default="wikipedia", help="Dataset to use. Available : wikipedia, gp (gender pronoun), gt (greater than), bool (evaluation of boolean expressions), ioi (indirect object identification) or mixture (of ioi, gt and gp).")
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
    print("Importing...")

import os
import gc

import cProfile

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

multiprocessing.set_start_method('spawn', force=True)

import torch

from transformers import logging
logging.set_verbosity_error()

from tqdm import tqdm

from ablation.node_ablation import get_fcs

from data.buffer import wikipedia_buffer, gp_buffer, gt_buffer, ioi_buffer, bool_buffer, mixture_buffer

from connectivity.effective import get_circuit

from evaluation.faithfulness import faithfulness as faithfulness_fn

from utils.plotting import plot_faithfulness, marks_plot_faithfulness
from utils.savior import save_circuit, load_latest
from utils.graph_utils import merge_circuits, mean_circuit
from utils.metric_fns import metric_fn_KL, metric_fn_logit, metric_fn_acc, metric_fn_MRR
from utils.experiments_setup import load_model_and_modules, load_saes

if args.dataset == "wikipedia":
    buffer_fn = wikipedia_buffer
elif args.dataset == "gp":
    buffer_fn = gp_buffer
elif args.dataset == "gt":
    buffer_fn = gt_buffer
elif args.dataset == "ioi":
    buffer_fn = ioi_buffer
elif args.dataset == "bool":
    buffer_fn = bool_buffer
elif args.dataset == "mixture":
    buffer_fn = mixture_buffer
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

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

        buffer = buffer_fn(pythia70m, cbs, DEVICE, args.ctx_len)
        
        tot_circuit = None
        
        i = 0
        if torch.cuda.is_available() and __name__ != "__main__":
            # We are dividing number of batches across devices
            n_devices = torch.cuda.device_count()
        else:
            n_devices = 1
        max_loop = args.circuit_max_loop // n_devices
        edge_threshold = args.edge_threshold
        node_threshold = args.node_threshold

        for batch in tqdm(buffer):
            if i >= max_loop:
                break
            i += 1
            
            tokens = batch["clean"]
            trg_idx = batch["trg_idx"]
            trg = batch["trg"]
            corr = None
            if "corr" in batch:
                corr = batch["corr"]
            corr_trg = None
            if "corr_trg" in batch:
                corr_trg = batch["corr_trg"]
                
            circuit = get_circuit(
                tokens,
                corr,
                model=pythia70m,
                embed=pythia70m_embed,
                attns=pythia70m_attns,
                mlps=pythia70m_mlps,
                resids=pythia70m_resids,
                dictionaries=dictionaries,
                metric_fn=metric_fn_logit,
                metric_kwargs={"trg": (trg_idx, trg, corr_trg)},
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                method=cm,
                aggregation=args.aggregation,
                dump_all=dump_all,
                save_path=save_path,
            )
            tot_circuit = merge_circuits(tot_circuit, circuit, aggregation=args.aggregation)
            
            if i % 10 == 0 or i == max_loop:
                to_save = tot_circuit
                if args.aggregation == "sum":
                    to_save = mean_circuit(tot_circuit, i)
                save_circuit(
                    save_path + "circuit/" + f"{DEVICE.type}_{DEVICE.index}/",
                    to_save[0],
                    to_save[1],
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

        buffer = buffer_fn(pythia70m)
        
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
        for batch in tqdm(buffer):
            if i >= max_loop:
                break
            i += 1
            #thresholds = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
            thresholds = torch.logspace(-6, 2, 20, 10).tolist()
            tokens = batch["clean"]
            trg_idx = batch["trg_idx"]
            trg = batch["trg"]
            outs = {
                'features' :
                    get_fcs(
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
                    get_fcs(
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
                    get_fcs(
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

        if hasattr(pythia70m, "gpt_neox"):
            n_layers = len(pythia70m.gpt_neox.layers)
        elif hasattr(pythia70m, "cfg"):
            n_layers = pythia70m.cfg.n_layers

        for i in range(max(start_at_layer, 0), n_layers):
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

        buffer = buffer_fn(pythia70m, ebs, DEVICE, args.ctx_len)
        
        aggregated_outs = None
        for batch in tqdm(buffer):
            if i >= max_loop:
                break
            i += 1

            tokens = batch["clean"]
            trg_idx = batch["trg_idx"]
            trg = batch["trg"]
            corr = None
            if "corr" in batch:
                corr = batch["corr"]
            corr_trg = None
            if "corr_trg" in batch:
                corr_trg = batch["corr_trg"]

            thresholds = torch.logspace(-2, 2, 15, 10).tolist()
            faithfulness = faithfulness_fn(
                pythia70m,
                submodules=submodules,
                sae_dict=dictionaries,
                name_dict=submod_names,
                clean=tokens,
                circuit=circuit,
                thresholds=thresholds,
                metric_fn=metric_fn_dict,
                metric_fn_kwargs={"trg": (trg_idx, trg, corr_trg)},
                patch=corr,
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
            
            plot_faithfulness(aggregated_outs, metric_fn_dict=metric_fn_dict, save_path=save_path)

        for t, out in aggregated_outs.items():
            if t == 'complete' or t == 'empty':
                for fn_name in metric_fn_dict:
                    aggregated_outs[t][fn_name] /= i
            else:
                for fn_name in metric_fn_dict:
                    aggregated_outs[t]['faithfulness'][fn_name] /= i

        plot_faithfulness(aggregated_outs, metric_fn_dict=metric_fn_dict, save_path=save_path)

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
    JSON_args = {
        "profile": args.profile,
        "get_circuit": args.get_circuit,
        "dump_all": args.dump_all,
        "eval_circuit": args.eval_circuit,
        "marks_version": args.marks_version,
        "identity_dict": args.identity_dict,
        "SVD_dict": args.SVD_dict,
        "node_threshold": args.node_threshold,
        "edge_threshold": args.edge_threshold,
        "ctx_len": args.ctx_len,
        "circuit_method": args.circuit_method,
        "aggregation": args.aggregation,
        "circuit_batch_size": args.circuit_batch_size,
        "circuit_max_loop": args.circuit_max_loop,
        "eval_method": args.eval_method,
        "eval_metric": args.eval_metric,
        "eval_batch_size": args.eval_batch_size,
        "eval_max_loop": args.eval_max_loop,
        "eval_start_at_layer": args.eval_start_at_layer,
        "save_path": args.save_path,
        "circuit_path": args.circuit_path,
    }

    print("Saving at : ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import json
    with open(save_path + "args.json", "w") as f:
        json.dump(JSON_args, f)
    
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