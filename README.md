# Computational Graphs

This project contains implementation for building and evaluating representations of neural networks as computational graphs.

You can find your favorite dataset and graph building algorithm in the `data` and `connectivity` folders.

To run forward passes on the graph, see the `ablation` folder.

To evaluate the graph, see the `evaluation` folder.

## Data

Contains functions to load various datasets, general or task specific.

## Connectivity

Contains functions to build computational graphs based on several paradigms, which we will call by their corresponding name in neurosciences as a hope of bringing MI and neuroscience closer together:
- **Structural connectivity** : computational graph built purely based on an analysis of the weights of the neural network.
    - In neurosciences, this would be the anatomical connections between neurons, e.g. axons between individual neurons or regions of interest.
- **Functional connectivity** : Builds the computational graph of the neural network based on the activations (or attribution) of its neurons.
    - Uses the covariance of the activations of the neurons to infer connections between them.
    - This is pretty much the exact same thing as what is done in neurosciences.
- **Effective connectivity** : Builds the computational graph of the neural network based on the causal relationships between its neurons.
    - Uses more expensive attribution methods to infer the causal relationships between neurons.
    - In neurosciences, an estimation of this would be given by time series data.

## Ablation

Contains edge and node ablation.
- edge ablation : run a given computational graph supposed to represent a neural network.
- node ablation : Ablates some nodes during a normal forward pass of a neural network. The effective computational graph is the complete graph between the remaining nodes.

## Evaluation

Contains functions for
- evaluation of the faithfulness of the computational graph to the original model on some input data.
- detection of communities and block structures in the graph using classical methods such as Louvain or spectral clustering as well as less known methods especially in machine learning such as Stochastic Blockmodels (SBM).