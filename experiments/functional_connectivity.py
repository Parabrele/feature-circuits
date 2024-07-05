"""
Example commands to run the script:

python -m experiments.functional_connectivity -id -roi -act -attr -nb 100 -bs 100 --dataset wikipedia --save_path /scratch/pyllm/dhimoila/output/functional/ROI/id/wikipedia/ -d 0 &
python -m experiments.functional_connectivity -id -roi -act -attr -bs 100 --dataset ioi --save_path /scratch/pyllm/dhimoila/output/functional/ROI/id/ioi/ -d 1 &
python -m experiments.functional_connectivity -id -roi -act -attr -bs 100 --dataset gp --save_path /scratch/pyllm/dhimoila/output/functional/ROI/id/gp/ -d 2 &
python -m experiments.functional_connectivity -id -roi -act -attr -bs 100 --dataset gt --save_path /scratch/pyllm/dhimoila/output/functional/ROI/id/gt/ -d 3 &
python -m experiments.functional_connectivity -id -roi -act -attr -bs 100 --dataset bool --save_path /scratch/pyllm/dhimoila/output/functional/ROI/id/bool/ -d 4 &
python -m experiments.functional_connectivity -id -roi -act -attr -bs 100 --dataset mixture --save_path /scratch/pyllm/dhimoila/output/functional/ROI/id/mixture/ -d 5 &

"""

##########
# Args
##########

if __name__ == "__main__":
    print("Starting...")
    print("Parsing arguments...", end="")

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--device", "-d", type=int, default=0, help="Device to use. Default : 0.")

parser.add_argument("--identity_dict", "-id", action="store_true", help="Use identity dictionaries")
parser.add_argument("--SVD_dict", "-svd", action="store_true", help="Use SVD dictionaries")
parser.add_argument("--White_dict", "-white", action="store_true", help="Use Whitening dictionaries")
parser.add_argument("--SAE", action="store_true", help="Use SAE dictionaries")

parser.add_argument("--ROI", "-roi", action="store_true", help="Use Regions of Interest -attn heads and MLP layers- instead of neurons - canonical dimensions in dictionary space for the given dictionaries.")

parser.add_argument("--activation", "-act", action="store_true", help="Compute activations")
parser.add_argument("--attribution", "-attr", action="store_true", help="Compute attributions")
parser.add_argument("--use_resid", "-resid", action="store_true", help="Use residual stream nodes instead of modules.")

parser.add_argument("--dataset", type=str, default="wikipedia", help="Dataset to use. Available : wikipedia, gp, gt, bool, ioi or mixture.")
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

device_id = args.device

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

from connectivity.functional import generate_cov

if __name__ == "__main__":
    print("Importing...")

import os

import torch

from transformers import logging
logging.set_verbosity_error()

from data.buffer import wikipedia_buffer, gp_buffer, gt_buffer, bool_buffer, ioi_buffer, mixture_buffer
from utils.experiments_setup import load_model_and_modules, load_saes

if args.dataset == "wikipedia":
    buffer_fn = wikipedia_buffer
elif args.dataset == "gp":
    buffer_fn = gp_buffer
elif args.dataset == "gt":
    buffer_fn = gt_buffer
elif args.dataset == "bool":
    buffer_fn = bool_buffer
elif args.dataset == "ioi":
    buffer_fn = ioi_buffer
elif args.dataset == "mixture":
    buffer_fn = mixture_buffer
else:
    raise ValueError(f"Unknown dataset : {args.dataset}")

if __name__ == "__main__":
    print("Done.")

##########
# Helper functions
##########

if __name__ == "__main__":
    JSON_args = {
        'identity_dict': idd,
        'SVD_dict': svd,
        'White_dict': white,
        'SAE': sae,
        'ROI': roi,
        'activation': act,
        'attribution': attr,
        'use_resid': use_resid,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'steps': steps,
        'node_threshold': node_threshold,
        'edge_threshold': edge_threshold,
        'aggregation': aggregation,
        'ctx_len': ctx_len,
        'save_path': save_path,
    }

    print("Saving at : ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import json
    with open(save_path + "args.json", "w") as f:
        json.dump(JSON_args, f)
    
    DEVICE = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    pythia70m, pythia70m_embed, pythia70m_resids, pythia70m_attns, pythia70m_mlps, submod_names = load_model_and_modules(DEVICE)

    dictionaries = load_saes(
        pythia70m,
        pythia70m_embed,
        pythia70m_resids,
        pythia70m_attns,
        pythia70m_mlps,
        idd=idd,
        svd=svd,
        white=white,
        device=DEVICE,
    )

    buffer = buffer_fn(pythia70m, batch_size, DEVICE, args.ctx_len)
        
    cov = generate_cov(
        data_buffer=buffer,
        model=pythia70m,
        embed=pythia70m_embed,
        resids=pythia70m_resids,
        attns=pythia70m_attns,
        mlps=pythia70m_mlps,
        dictionaries=dictionaries,
        get_act=act,
        get_attr=attr,
        neuron=(not roi),
        ROI=roi,
        batch_size=batch_size,
        use_resid=use_resid,
        n_batches=n_batches,
        aggregation=aggregation,
        steps=steps,
        discard_reconstruction_error=(not sae),
    )
    
    name = "identity" if idd else "SVD" if svd else "White" if white else "SAE" if sae else "Unknown"
    name += "_ROI" if roi else "_Neurons"
    name += "_cov.pt"
    if act:
        print("Saving activations...", end="")
        torch.save(cov['act'], save_path + "act_" + name)
        print("Done.")
    if act:
        print("Saving attributions...", end="")
        torch.save(cov['attr'], save_path + "attr_" + name)
        print("Done.")