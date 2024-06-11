# abstract :
-----
we explore ways to abstract transformer models by building a graph of computation, and ways to evaluate the quality of these graphs.
Also, apparently it is common practice to fight for who gets the best subgraph discovery algorithm but then everyone evaluates faithfulness & co on a different subgraph, using node ablation rather than edge ablation. We use edge ablation.

-----
# Introduction :
-----

TODO : models are a really big function and we try to express it as a composition of simpler functions, and we try to build the graph of dependencies between these simpler functions.

TODO : do all of it on weighted head graph to see if some things appear (in the brain, the number of nodes mapped is really small and maybe comparable to the number of heads)

The goal of MI is to reverse engineer the inner workings of a model.
A shit ton of work has been done in the last few years in trying to find subgraphs of the graph of computation responsible for some selected task or behavior, everyone is fighting over who will get the best graph discovery algorithm.

We want to go to the next step, so no more task specific subgraph discovery, but describing the whole graph of computation and what can be learned from it. What we hope to do, or future work to do, is to find functionally relevant subgraphs in an unsuperised way as communities of these graphs of computation using either leidin or activation patterns, as well as being able to label nodes with simple functions, still in an unsupervised way. We further want to compare graphs of models of the same and increasing size to look for reocurring patterns which would correspond to matching communities corresponding to the same functionality.

Some work already exist on this and try to find communities of nodes in MLP and CNNs (clusterability in NN & functional modularity), and to some extend in transformers (LIB) with mitigated success in toy models and none on language models.

We want to disantangle the model, from a coarse grain dense graph of computation extremely efficient in terms of computation performances, to more fine grained graph of computation, slower but more useful representations.

We will largely build upon the dictionary learning literature (Marks anthropic logan etc SAEs, Kaarel LIB, gated SAE, sparse attribution SAE (with sparsity penalty on gradient)) and our method is independent of these dictionary learning method, any of them can be used to build and use the graph of computation using our method. TODO : do not specifically describe MLP dictionaries, just speak of encoder and decoder as arbitrary functions depending on your prefered dictionary learning method, and say that for our experiments we use MLP dictionaries.

Hope is that
- since fine grained & (hopefully) sparse, nodes correspond to easier functions (so provided that dict is good, not only does it correspond to a feature, it also describes it as an easy function of previous features) whereas more coarse grain graphs have no hope of describing the function associated to some node, except on very simple specific tasks. E.g. general graph based on heads are just complete graphs since everything is so interconected. The reason why models are black boxes is because their immediate graph of computation based purely on their architecture is really not that informative and doesn't allow for simple description of the functions implemented by the model. By having a more fine grained graph of computation hopefully we can automatically find simple functions corresponding to each node from which the global, very complicated function emerges.
- communities of nodes correspond to behaviors & conversely
- subgraphs / communities are shared between models of the same size / of increasing size
- graphs of computation are fractionally isomorphic
- even if both above points hold, two isomorphic subgraph based only on dependencies might implement very different functions, so labeling nodes might be important, and checking that matching communities actually correspond to the same behavior is important
- when doing the activation matrix thing, do it jointly on two different models and see how close these communities are.

Since we need a fine grained graph we will chose to build it from sparse dictionaries in a similar way to Marks although with some differences. Proceed to explain why he lies and does shit. Proceed to explain my superiority.

-----
# Background :
-----

- Notations : introduce
    - $\textit{module}$ as either attention head, MLP or whole transformer block. Introduce transformer architecture ?
    - $d_{model}$
    - activation as output of a module on some input sequence
    - ...
    - [TODO : This comes a little early as we did not introduce these concepts yet. How to phrase it ?] In section \ref{TODO} we introduce $d_{dict}$ the size of a dictionary, as well as $b_i$ and $\alpha_i$ - respectively the features of a dictionary and their activation, or magnitude, for some input. In the remainder of this article, we will often identify these two concepts under the notation $f_i$ or simply $f$, for feature.
    
- Attribution / explanation methods (parler de IG en 1er, vu que c'est la méthode exacte, mais les autres sont des approximations)
    - IG, GIG
    - Backprop, guided backprop
    - GradCam, guided gradcam
    - LRP

- Sparse AutoEncoders
    - linear representation hypothesis : The linear representation hypothesis assumes that a model's latent space behaves mostly linearly, with high level concepts represented linearly as directions in this space often called features. Evidence of that is provided by [TODO : citations].
    - This motivates the use of dictionary learning and sparse coding methods to find overcomplete bases able to sparsely represent the data at any point in the model. Formally, we want a familly $\{b_i \in \R^{d_{model}}\}_{i=0}^{d_{dict}}$ such that, for all activations $x$ of some module, there is a set $\{\alpha_i \in \R\}_{i=0}^{d_dict}$ mostly 0 such that $x = \sum_{i=0}^{d_dict} \alpha_i b_i$. There are two dual problems here, one is to find such a familly, the other is, given a familly and a data point, find the $\alpha_i$.
    - $\textbf{Notation}$ In the rest of this article, these $b_i$ will be called features, and both $b_i$ and it's magnitude $\alpha_i$ will be identified under the notation $f_i$.
    - People usually train sparse autoencoders to solve both of these problems, the decoder serving as the dictionary, and the encoder as the sparse coding function.
        - Current research sometimes use linear autoencoders, but most of the time one layer MLPs, trained to learn the identity function with an added sparsity loss. [TODO : citet] also introduced a change of basis called LIB [TODO : full name] that can be viewed as an autoencoder capturing important functionally important features, though the decomposition is not exactly sparse.
        - future research have been proposed with sparse attribution (TODO : what is their name ?), gated SAE and Topk SAE for better quality of the dictioanry and of the sparse coding function.
    - We will choose to use these features as nodes for our graph, and their causal dependencies as edges. This allows for an extremely fine grained study of the computational graph, where each node corresponds to a basic feature, and it provides useful information on the dependencies between these features.

- Circuit discovery
    - granularity : heads or lower, deemed to be task specific, can't be used for whole model description, as they would simply end up being the complete graph given by the architecture
        - [TODO : keep this ? It's not really circuit discovery...] Look before you leap : split the model based on the layer after which some behavior is observed.
        - information flow : build the computational graph for example sequences, where nodes are heads, MLPs and residual streams. Can only get the frequency of activation of each module, so in the end they don't really build a graph. They define contribution of each module $m_1$'s output $x_1$ to any of its successor $m_2$'s output $x_2$ as follows. They first linearise the function of $m_2$ into $f$, and then look at how close $f(x_1)$ is to $x_2$. [TODO : speak about my experiments on making the same thing with different inner products and by linking directly modules, without residual stream checkpoints, and how it didn't work ? If using different attribution methods, it becomes AtP.]
        - ACDC, AtP/AtP* : build a graph of computation responsible for some specific behavior. Nodes are individual attention heads and MLP layers. Edges are the dependencies between them using various attribution methods. In ACDC, they patch the edges to find the ones who change the most the downstream module's output. In AtP, they use the norm of the gradient of the downstream module's output with respect to the upstream module's output. These graphs of computations can't be applied to describe the whole model, as they turn out to be almost the complete graph corresponding to the model architecture.
    - granularity : features, allow for a much higher expressivity with individual nodes capturing only a small, local function.
        - Marks : use integrated gradients to get the contribution of all features to some metric on the model's output. They restric the model to nodes with a contribution above some threshold - so the computational graph is the complete graph between these nodes. They also define edges between these nodes as follows. They draw an edge between some upstream feature $f_1$ to some downstream feature $f_2$ if the product [TODO : math formula instead of words : of the contribution of $f_2$ to the model's output and the gradient of $f_2$ with respect to $f_1$ is above some threshold / IG(m, f_2) cdot partial_d f_2 / d f_1]. We fail to understand this choice as it is not mentioned in their paper but it is what they do in their code, and they never seem to use these graphs except to draw figures to illustrate their method.
            - expose his trichery
        - https://arxiv.org/pdf/2402.12201.pdf
        - LIB


-----
# Graph discovery - building the graph of computation :
-----

-----
### Method

Ne pas oublier de dire que ce n'est pas spécifique aux MLP SAE qu'on a utilisé (voir, utiliser les gated SAE sur gpt2small si possible). Ca marche aussi pour LIB ou n'importe quel couple (sparse coding, dictionary) qui peut être "vu" comme un encoder-decoder, et dérivable.

-----
### Experiments

-----
# Exploring communities
-----

-----
# Method

-----
# Experiments


-----
# Discussions, future work & conclustion
-----

