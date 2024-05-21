"""
This file contains evaluations for compuational graphs, including :

faithfulness :
    - override forward fct to keep only edges defined by some graph
    - measure recovered metric for some graph

sparsity :
    - get edges / nodes as a function of nodes (n^2, nln(n), O(n), etc.) (1)
    - get minimum edges (and edges / nodes) (per layer / total / etc.) requiered to recover 1 +- eps of some metric (e.g. accuracy, CE, etc.)
    - plot recovered metric as a function of (1) (mostly O(n), but also include n^2, nsqrt(n), nln(n), etc.) in the plot as vertical lines as an indication
    - do as marks (for completeness) and patch nodes instead of edges, and plot how much nodes are required to recover some metric with the complete graph between those nodes
    
modularity :
    - cluster the nodes st degree(intra) / degree(inter) is maximized, or degree(inter) is minimized, or etc.
    - measure the separation and the quality of the clusters
    - do clusters of clusters and iteratively to estimate the degree of complexity of the model
        (this would require clusters to be quite small though, and assume that cluster graph is still sparse)

"""