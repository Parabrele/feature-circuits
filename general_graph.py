"""
Get the general graph of a given model

Algorithm :
Get model & dicts
Get circuit fct
aggregate graphs over dataset
test this graph on dataset
"""

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--device", "-d", type=int, default=0)

    parser.add_argument("--node_threshold", "-nt", type=float, default=0.1)
    parser.add_argument("--edge_threshold", "-et", type=float, default=0.01)

    parser.add_argument("--max_loop", "-ml", type=int, default=1000)

    parser.add_argument("--circuit_method", "-cm", type=str, default="resid")
    parser.add_argument("--circuit_batch_size", "-cbs", type=int, default=1)
    parser.add_argument("--ctx_len", "-cl", type=int, default=16)

    parser.add_argument("--eval_method", "-em", type=str, default="edge_ablation")
    parser.add_argument("--eval_batch_size", "-ebs", type=int, default=1)

    parser.add_argument("--save_path", "-sp", type=str, default='/scratch/pyllm/dhimoila/output/')
    parser.add_argument("--circuit_path", "-cp", type=str, default=None)

    args = parser.parse_args()

    from tqdm import tqdm, trange

    import os
    import gc
    import pickle
    import math
    import importlib


    from transformers import logging
    logging.set_verbosity_error()

    import torch
    from nnsight import LanguageModel
    from datasets import load_dataset

    import plotly.graph_objects as go
    from scipy import interpolate

    import networkx as nx

    from dictionary_learning import AutoEncoder
    from utils import SparseAct
    from buffer import TokenBuffer
    import evaluation
    from circuit import get_circuit
    from utils import save_circuit, load_circuit
    from ablation import run_with_ablations

    from matplotlib import pyplot as plt

    available_gcm = ["resid", "marks", "resid_topk"]
    available_em = ["node_ablation", "edge_ablation"]

    DEVICE = torch.device('cuda:{}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    print("DEVICE :", DEVICE)

    cm = args.circuit_method
    cbs = args.circuit_batch_size
    em = args.eval_method
    ebs = args.eval_batch_size
    save_path = args.save_path
    circuit_path = args.circuit_path

    def load_model_and_modules(): 
        pythia70m = LanguageModel(
        "EleutherAI/pythia-70m-deduped",
        device_map=DEVICE,
        dispatch=True,
        )

        pythia70m_embed = pythia70m.gpt_neox.embed_in

        pythia70m_resids= []
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
        dict_size = 32768
        
        base = '/scratch/pyllm/dhimoila/'
        path = base + "dictionaires/pythia-70m-deduped/"

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
    
    def run_first_step():
            
        def metric_fn(model, trg=None):
            """
            default : return the logit
            """
            if trg is None:
                raise ValueError("trg must be provided")
            return model.embed_out.output[torch.arange(trg[0].numel()), trg[0], trg[1]]
        
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
        max_loop = args.max_loop
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
                metric_fn=metric_fn,
                metric_kwargs={"trg": (trg_idx, trg)},
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                method=cm,
            )
            if tot_circuit is None:
                tot_circuit = circuit
            else:
                for k, v in circuit[0].items():
                    if v is not None:
                        tot_circuit[0][k] += v
                for ku, vu in circuit[1].items():
                    for kd, vd in vu.items():
                        if vd is not None:
                            tot_circuit[1][ku][kd] += vd
            
            if i % 10 == 0:
                save_circuit(
                    save_path + f"circuit/{DEVICE.type}_{DEVICE.index}/",
                    tot_circuit[0],
                    tot_circuit[1],
                    "wikipedia",
                    "pythia_70m_deduped",
                    node_threshold,
                    edge_threshold,
                    i * cbs,
                )
            
            del circuit
            gc.collect()
        
    run_first_step()