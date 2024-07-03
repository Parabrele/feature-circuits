# Evaluation

This file contains functions to evaluate graphs (see `connectivity` folder for more details on how they can be built).

- `SBM.py` contains functions to fit SBM models to graphs for later study of the block structure.
- `community.py` contains functions to compute communities using Louvain or spectral clustering methods. These methods are less general than SBMs and assume that the graph is only made of communities (i.e. sets of nodes with more edges between them than with the rest of the graph). SBMs are more general and are a block version of Erdos-Renyi graphs.
- `faithfulness.py` contains functions to evaluate the faithfulness of a graph to a given model, given some input data. This is useful to check if the induced model is a good fit for the model on the given data, or to run ablation experiments, where some subset of nodes or edges are removed.