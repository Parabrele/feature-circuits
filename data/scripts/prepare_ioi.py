import os
import json
import random
from tqdm import tqdm
import sys
import argparse

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

places = [
    "bar",
    "store",
    "restaurant",
    "park",
    "theater",
    "zoo",
    "museum",
    "school",
    "library",
    "hospital",
    "bank",
    "street",
    "beach",
    "lake",
    "river",
    "mountain",
    "forest",
    "gym",
    "cave",
    "cafe",
    "shop",
    "hotel",
    "airport",
    "market",
]

objects = [
    "book",
    "plum",
    "apple",
    "orange",
    "mango",
    "cherry",
    "bottle",
    "glass",
    "coin",
    "watch",
    "ring",
    "necklace",
    "ball",
    "bat",
    "pant",
    "pear",
    "shell",
    "rock",
    "pearl",
    "pen",
    "pencil",
    "eraser",
]

def select_single_token_names(names, tokenizer):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    names_ = [name for name in names if len(tokenizer.encode(name)) == 1]
    return [name for name in names_ if len(tokenizer.encode(" " + name)) == 1]

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--out_dir", "-o", default="/home/pyllm/dhimoila/feature-circuits-1/data/datasets/ioi")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--train", "-t", type=int, default=200)
    parser.add_argument("--validation", "-v", type=int, default=200)
    parser.add_argument("--test", "-e", type=int, default=10000)
    parser.add_argument("--tokenizer", "-tk", default="gpt2")
    parser.add_argument("--names", "-nm", default="/home/pyllm/dhimoila/feature-circuits-1/data/helper_files/names.json")
    
    args = parser.parse_args()
    
    args.names = json.load(open(args.names))
    if "girls" in args.names:
        args.names = args.names["boys"] + args.names["girls"]

    args.names = select_single_token_names(args.names, args.tokenizer)
    
    return args

templates = [
    "Then, {A} and {B} went to the {PLACE}. {C} gave a {OBJECT} to",
    "Then, {A} and {B} had a lot of fun at the {PLACE}. {C} gave a {OBJECT} to",
    "Then, {A} and {B} were working at the {PLACE}. {C} decided to give a {OBJECT} to",
    "Then, {A} and {B} were thinking about going to the {PLACE}. {C} wanted to give a {OBJECT} to",
    "Then, {A} and {B} had a long argument, and afterwards {C} said to",
    "After {A} and {B} went to the {PLACE}, {C} gave a {OBJECT} to",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {C} decided to give it to",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {C} decided to give the {OBJECT} to",
    "While {A} and {B} were working at the {PLACE}, {C} gave a {OBJECT} to",
    "While {A} and {B} were commuting to the {PLACE}, {C} gave a {OBJECT} to",
    "After the lunch, {A} and {B} went to the {PLACE}. {C} gave a {OBJECT} to",
    "Afterwards, {A} and {B} went to the {PLACE}. {C} gave a {OBJECT} to",
    "Then, {A} and {B} had a long argument. Afterwards {C} said to",
    "The {PLACE} {A} and {B} went to had a {OBJECT}. {C} gave it to",
    "Friends {A} and {B} found a {OBJECT} at the {PLACE}. {C} gave it to",
]

def main():
    # Don't load the data to do I don't know what with it. Generate the data using the templates.
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    ioi_dataset = {}
    for split in ['train', 'validation', 'test']:
        ioi_dataset[split] = []
        n_samples = getattr(args, split)
        for _ in tqdm(range(n_samples)):
            baba_or_abba = random.randint(0, 1)
            template = random.choice(templates)
            a, b, c = random.sample(args.names, 3)
            place = random.choice(places)
            obj = random.choice(objects)
            if baba_or_abba:
                sentence = template.format(A=a, B=b, C=b, PLACE=place, OBJECT=obj)
                corrupted = template.format(A=a, B=b, C=c, PLACE=place, OBJECT=obj)
            else:
                sentence = template.format(A=b, B=a, C=b, PLACE=place, OBJECT=obj)
                corrupted = template.format(A=b, B=a, C=c, PLACE=place, OBJECT=obj)
            ioi_dataset[split].append({
                'ioi_sentences': sentence,
                'corr_ioi_sentences': corrupted,
                'a': a,
                'b': b,
                'c': c,
                'place': place,
                'object': obj,
                'template': template,
                'order': 'abba' if baba_or_abba else 'baba'
            })

    ioi_dataset = DatasetDict({
        k: Dataset.from_list(v) for k, v in ioi_dataset.items()
    })

    ioi_dataset.save_to_disk(args.out_dir)

if __name__ == '__main__':
    main()