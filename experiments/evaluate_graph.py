import pickle
import os
import evaluation
from utils import load_latest

path = '/scratch/pyllm/dhimoila/output/120624_01/circuit/merged/'

circuit = load_latest(path)[1]
eval = evaluation.evaluate_graph(circuit)

#print(eval)