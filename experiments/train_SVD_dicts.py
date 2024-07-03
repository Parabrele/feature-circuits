"""
python train_SVD_dicts.py --device 0&
python train_SVD_dicts.py --device 1&
python train_SVD_dicts.py --device 2&
python train_SVD_dicts.py --device 3&
python train_SVD_dicts.py --device 4&
python train_SVD_dicts.py --device 5&
python train_SVD_dicts.py --device 6&
python train_SVD_dicts.py --device 7&
"""

import torch
import time

from dictionary_learning.training import trainSVDdict
from dictionary_learning.buffer import ActivationBuffer
from argparse import ArgumentParser

from nnsight import LanguageModel
from datasets import load_dataset

parser = ArgumentParser()
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

print('device:', device)

base_path = '/scratch/pyllm/dhimoila/dictionaires/pythia-70m-deduped/SVDdicts/'

print('starting to load...')

pythia70m = LanguageModel(
    "EleutherAI/pythia-70m-deduped",
    device_map=device,
    dispatch=True,
)
dataset = load_dataset("wikipedia", language="en", date="20240401", split="train", streaming=True, trust_remote_code=True)
dataset = iter(dataset.shuffle())

d_model = pythia70m._model.config.hidden_size
submodules = {
    'embed': (('embed',), pythia70m.gpt_neox.embed_in),
}

for i, layer in enumerate(pythia70m.gpt_neox.layers):
    submodules[f'attn_out_layer{i}'] = (('attn', i), layer.attention)
    submodules[f'mlp_out_layer{i}'] = (('mlp', i), layer.mlp)
    submodules[f'resid_out_layer{i}'] = (('resid', i), layer)


for i, (name, module) in enumerate(submodules.items()):
    if not(((i % 8) == args.device) and torch.cuda.is_available()):
        continue

    print(f'Now training {name}...')

    start = time.time()
    buffer = ActivationBuffer(
        dataset,
        pythia70m,
        module,
        d_model,
        n_ctxs=1000,
        ctx_len=128,
        load_buffer_batch_size=512,
        return_act_batch_size=8192,
        device=device,
    )

    trainSVDdict(
        buffer,
        d_model,
        args.n_steps,
        save_steps=10,
        save_dir=base_path + name + '/',
        device=device,
    )
    end = time.time()
    print(f'{name} done! Time taken: {end - start} seconds.')
