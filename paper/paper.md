`TEMP` flags indicate that these sections are temporary and should be rewritten once more results are available and we have a clearer idea of what we are going to say.

`FUTURE` flags indicate that these points should be left for future work.

<!-- TOSAY flags indicate something that probably should be said somewhere. -->

# abstract :
-----
`TEMP`
we explore ways to abstract transformer models as computational graphs, and propose ways to evaluate the quality of these graphs. We further propose new ways for automating interpetability, relying heavily on these graphs. We show that 
Also, apparently it is common practice to fight for who gets the best subgraph discovery algorithm but then everyone evaluates faithfulness & co on a different subgraph, using node ablation rather than edge ablation. We use edge ablation.

-----
# Introduction :
-----
`TEMP`

<!--
(for internship report : ) SAEs are just one direction in trying to find better geometric representations of models. Most obvious : SVD/whitened space, more complex : LIB space, feature space, for any dictionary one can think of. Finding specific directions / features as well as studying interventions are very related fields when studying the geometry of latent space. 

TOSAY : sleeper agent & ensuring only "training" behaviors are present even in "evaluation" : sleeper agents might be different nodes / communities / links in the graph that we didn't get in training, so by enforcing the graph we have, these sleeper agents are now kept asleep. Also, if some nodes/communities are labeled as harmful, we can prune them

TOSAY : models are a really big function and we try to express it as a composition of simpler functions by building the graph of dependencies between these simpler functions to enhance interpretability.
-->

<!-- (probably forget that) TOSAY : say somewhere that head graph is interesting as in neuroscience, networks are generally build between big regions of the brain, never between individual neurons, so there might be a parallel as heads and MLPs are "big regions". Whether this parallel fails or not is an interesting question (by "fail" we mean that a head based computational graph might be too entangled to be able to extract any useful information from it). -->

The goal of MI is to reverse engineer the inner workings of a model.
A ton of work has been done in the last few years in trying to find subgraphs of the computational graph responsible for some selected task or behavior, everyone is fighting over who will get the best graph discovery algorithm.

We want to go to the next step, so no more task specific subgraph discovery, but describing the whole computational graph and what can be learned from it. What TODO FUTURE WORK ? : we hope to do, or future work to do, is to find functionally relevant subgraphs in an unsuperised way as communities of these computational graphs.<!-- `FUTURE` TODO (labeling : probably foget, matching and training : only if we have time and do the experiment. Otherwise, keep for further paper, but mention it in the internship report.), as well as being able to label nodes with simple functions, still in an unsupervised way. We further want to compare graphs of models of the same and increasing size to look for reocurring patterns which would correspond to matching communities. We also want to study the emergence of these structures across training, and how they evolve across time.-->

Some work already exist on this and try to find communities of nodes in MLP and CNNs \citep{lu2019checking, filan2021clusterability}, and to some extend in transformers (LIB) with mitigated success in toy models and none on language models.<!--TODO ref riggs has some primary results on labeling nodes with simple functions, where he labels nodes as linear functions of their predecessors.-->

We want to disantangle the model, from a coarse grain dense computational graph extremely efficient in terms of computation performances, to more fine grained computational graph, slower but more useful representations. <!--We compare different representations with various granularity in the experiment section.-->

We will largely build upon the dictionary learning literature (Marks anthropic logan etc SAEs, gated SAE, sparse attribution SAE (with sparsity penalty on gradient)) and our methods are largely independent of the chosen dictionary learning method so long as it has a derivable encoding and decoding functions, any can be used to build and study the computational graph using our method. In particular, our method also works for changes of basis (ID, SVD, LIB, or any further proposed basis).<!--TODO : do not specifically describe MLP dictionaries, just speak of encoder and decoder as arbitrary functions depending on your prefered dictionary learning method, and say that for our experiments we use MLP dictionaries.-->

The rest of the introduction will be completed later.

<!--
Hope is that
- since fine grained & (hopefully) sparse, nodes correspond to easier functions (so provided that sparse dict is good, not only does it correspond to an atomic feature, it also describes it as an easy/atomic function of previous features) whereas more coarse grain graphs have no hope of describing the function associated to some node, except on very specific tasks with human labeling.
    - e.g. computational graph based on heads are just complete graphs - basically the original architecture - since everything is so interconected. The reason why models are black boxes is because their immediate graph of computation based purely on their architecture is really not that informative and doesn't allow for simple description of atomic functions implemented by the model or their dependencies. By having a more fine grained graph of computation hopefully we can automatically find simple functions corresponding to each node from which the global, very complicated function emerges.
- communities of nodes correspond to behaviors & conversely
- subgraphs / communities are shared between models of the same size / of increasing size.
    - it would give some indication that some learned behavior is optimal and having a bigger model only allows for new behaviors, but the one already present in smaller models are robust enough.
- graphs of computation are fractionally isomorphic
- even if both above points hold, two isomorphic subgraph based only on dependencies might implement very different functions, so labeling nodes might be important, and checking that matching communities actually correspond to the same behavior is important (so their nodes activate on the same examples, they are correlated). Conversely, check that subgraphs of correlated communities in two models are close.
    - match communities based on graph structure vs based on behavior / activation patterns
- when doing the activation matrix thing, do it jointly on two different models and see how close these communities are.

Since we need a fine grained graph we will chose to build it from sparse dictionaries in a similar way to Marks although with some differences. We propose very general algorithm to build computational graphs.
-->

-----
# Background :
-----

- Notations :
    - A neural network model will be denoted as $\mathcal{M}$.
    - $\textit{module}$ or $m$ as either attention head, MLP or whole transformer block - or residual stream. Introduce transformer architecture ?
    - $d_{model}$
    - $x^m$, or $\textit{activation}$ as output of a module $m$ on some input $x$
    - a scalar function depending on the model and some input will be denoted as $\lambda(\mathcal{M}, x)$. $\lambda$ can be a metric function on the output of the model, or any scalar function depending on the internal activations of the model.
    - ...
    - [TODO : This comes a little early as we did not introduce these concepts yet. How to phrase it ?] In section \ref{TODO} we introduce $d_{dict}$ the size of a dictionary, as well as $b_i$ and $\alpha_i$ - respectively the features of a dictionary and their activation, or magnitude, for some input. In the remainder of this article, we will often identify these two concepts under the notation $f_i$ or simply $f$, for feature.
    
- Attribution / explanation methods
    - We discuss here classical attribution methods in neural networks. Attribution is the process of explaining a model's prediction by attributing it to some input or intermediate feature. This is a very active field of research, and many methods have been proposed, all with their own strengths and weaknesses.

    - We begin with the most basic and popular methods. \citet{springenberg2015strivingGuidedBackprop} used gradient in vision models to highligh which pixels of an image best explain some prediction. Let $\lambda : x \in \R^{d_{model}} \mapsto \lambda(x) \in \R$ be a scalar function and $x \in \R^{d_{model}}$ an input image of size $d_{model}$.
    $$ Attr(x_i) = \frac{\partial \lambda(x)}{\partial x_i} $$

    They introduce guided backprop, a variant that removes negative gradients in $ReLU$ layers in the backward gradient computation, which seems to improve the quality of the explanation. The attribution when passing through a $ReLU$ layer is not
    $$ Attr^l = (x^l > 0) \cdot Attr^{l+1} $$
    which would be the gradient, but
    $$ Attr^l = (x^l > 0) \cdot (Attr^{l+1} > 0) \cdot Attr^{l+1} $$
    which is the masked gradient, kept only when positive. $x^l$ is the activation of the layer $l$, $Attr^l$ is the attribution of the layer $l$, and $Attr^{l+1}$ is the attribution of the layer $l+1$. For non $ReLU$ layers, the attribution is simply the gradient.

    Later on, \citet{zhou2016CAM} and \citet{selvaraju2017GradCAM} introduced CAM, GradCAM and guided GradCAM, all method that build upon these results to refine the attribution of the model's prediction to the input specifically for convolutional layers. Another noteworthy method for attribution is Layer-Wise Relevance Propagation \citep{montavon2019layerLRP}. Methods purely based on gradients most notably suffer from the fact that the model is not a linear function and that we don't know which directions pointed by the gradient are relevant - unless there is some sense of counterfactual input, as in \citep{syed2023attributionAtP}.
    
    - Building upon those, \citet{sundararajan2017axiomaticIG} introduced Integrated Gradients along with a set of axioms desirable for an attribution method. This method can be formalised in a general manner as follows. Let $\lambda : x \in \R^{d_{model}} \mapsto \lambda(x) \in \R$ be a scalar function, $x_{clean} \in \R^{d_{model}}$ an input, $x_{baseline} \in \R^{d_{model}}$ a baseline input. Let also $\gamma : [0, 1] \rightarrow \R^{d_{model}}$ be a path function between $x_{baseline}$ and $x_{clean}$. They define the attribution function as follows :

    $$ Attr_{\gamma}(x) = \int_{0}^{1} \frac{\partial \lambda(\gamma(t))}{\partial \gamma_i(t)} \cdot \frac{\partial \gamma_i(t)}{\partial t} dt $$

    The main idea is that since the model is not linear, the gradient alone is not enough to explain the model's prediction - adressing the first issue of the methods mentioned earlier. E.g. if some very important feature happens to be on a plateau, it will not be picked up by the gradient. <!--TODO Evidence of this sort of thing happening is found in \citep{TODO}.-->
    
    \citet{smilkov2017smoothgradIG, miglani2020investigatingIG, kapishnikov2021guidedGIG} improved this method of integrated gradient mainly through the choice of path and baseline which constitute the main limitations of integrated gradients, and can lead to some noise in the attribution. <!--TOSAY : we do not use these improvements and stick to the simplest version of IG. Further work can investigate the impact of model specific vs model agnostic paths and baselines on the quality of the explanation and computational graph.-->
    
    - In the circuit discovery literature \citep{olah2020zoomCircuits}, people got interested in the contribution of edges or nodes in the computational graph, and used method developped in computer vision \citep{syed2023attributionCircuits, marks2024sparseCircuits} as well as novel methods \citep{wang2022interpretabilityCircuits, conmy2023automatedACDCCircuits, ferrando2024informationCircuits, he2024dictionaryCircuits} mostly involving patching individual nodes or edges to get their contribution, or linearizing the internal computations.

- Sparse AutoEncoders
    - linear representation hypothesis : The linear representation hypothesis assumes that a model's latent space behaves mostly linearly, with high level concepts represented as disantangled directions in this space often called features. It also states that linear directions can be found using linear probes and that they can be used as steering vectors to predictably influence the model's behavior. Evidence of that is provided by \citet{park2023linear, NIPS2013_9aa42b31, pennington2014glove, nanda2023emergent, gurnee2024language, wang2024concept, turner2024activationSteering}
    - This motivates the use of dictionary learning and sparse coding methods to find overcomplete bases able to sparsely represent the data at any point in the model. Formally, we want a familly $\{b_i \in \R^{d_{model}}\}_{i=0}^{d_{dict}}$ such that, for all activations $x$ of some module, there is a set $\{\alpha_i \in \R\}_{i=0}^{d_dict}$ mostly 0 such that $x = \sum_{i=0}^{d_dict} \alpha_i b_i$. There are two dual problems here, one is to find such a familly, the other is, given a familly and a data point, find the $\alpha_i$.
    - $\textbf{Notation}$ In the rest of this article, these $b_i$ will be unit vectors called features, and both $b_i$ and it's magnitude $\alpha_i$ will be identified under the notation $f_i$.
    - People usually train sparse autoencoders - SAE- \citep{ng2011sparse} to solve both of these problems. The decoder serves as the dictionary, while the encoder is the sparse coding function.
        - The majority of research in transformers uses one layer MLPs \citep{cunningham2023sparse, gao2024scaling} trained to learn the identity function with an added sparsity loss. \citet{rajamanoharan2024improving} proposed an improvement to this method by adding a gated mechanism to the MLPs.
        - \citet{bushnaq2024using} introduced a change of basis called LIB that can be viewed as a linear autoencoder capturing functionally important features, though the decomposition is not exactly sparse. Any change of basis given by future research can be similarly used as a linear autoencoder. `FUTURE`
        - All of this work finds higly interpretable features.
        <!--- `FUTURE` future work have been proposed with sparse attribution (TODO : what is their name ?), -->
    <!-- - We will choose to use these features as nodes for our graph, and their causal dependencies as edges. This allows for an extremely fine grained study of the computational graph, where each node corresponds to a basic feature, and it provides useful information on the dependencies between these features. -->

- Circuit discovery
    - early work, vision models, find related features across layers seeminly responsible for some behavior \citep{olah2020zoomCircuits}. This sparked interest in the community, and recent work has been done on both vision and language transformers.
    - \citet{conmy2023automatedACDCCircuits} patches each edge with a baseline value. Edges are those of the complete computational graph, as defined by the original transformer architecture - between modules. They then compare the induced model to the unmodified one. Edges with an effect smaller than some threshold are removed. This is done for a specific task with it's associated dataset. However, this approach is very computationally expensive and doesn't scale well with model size.
    - \citet{syed2023attributionCircuits} reused some gradient based attribution method by comparing the gradient to some patching direction in order to have a much more efficient method.
    - Such methods based on whole modules are very coarse and aim at finding the role played by specific modules under a specific context. When applied to a general dataset to get the general computational graph - not only a task specific subgraph - they give an almost complete graph of dependencies as computations in the model are very entangled and each module implements a very complicated function involved in a lot of different behaviors. This motivates the use of more fine grained methods to disantangle the model's computation.
        <!-- (I don't think I will put this : ) - information flow : build the computational graph for example sequences, where nodes are heads, MLPs and residual streams. Can only get the frequency of activation of each module, so in the end they don't really build a graph. They define contribution of each module $m_1$'s output $x_1$ to any of its successor $m_2$'s output $x_2$ as follows. They first linearise the function of $m_2$ into $f$, and then look at how close $f(x_1)$ is to $x_2$. [TODO : speak about my experiments on making the same thing with different inner products and by linking directly modules, without residual stream checkpoints, and how it didn't work ? If using different attribution methods, it becomes AtP.]-->
    - \citet{oneill2024sparse} used SAE's trained specifically on the task under study to disantangle the computation and build computational graphs, still based on heads but using information flow through these disantangled features. \citet{he2024dictionaryCircuits} also used SAE's, building dependencies between features by linearising the model's computations, removing the need for patches<!-- or hiding it well enough indirectly in some computation-->.
    - \citet{marks2024sparseCircuits} again used sparse autoencoder to disantangle the model's computation, and used integrated gradients to get the contribution of each feature to the model's output. Their computational graph is however the complete graph between these selected features, which, if applied to a general dataset, would simply yield the complete graph and the original model would be unchanged. They mention trying to attribute edges contributions too, but they never used them in their experiments.
    - Some work also tried to use clever changes of basis, or linear autoencoders, to disantangle the model's computation. \citet{merullo2024talking} worked purely based on the weights of the model to link singular vecotrs of weight matrices as features. \citet{bushnaq2024using} introduced a change of basis called LIB that supposed to capture functionally important features and dependencies.
    - All these methods seem to be able to give interesting cherry picked explanations on toy models or tasks.

-----
# Graph discovery - building the computational graph :
-----

In this section, we describe the representation we chose to represent a transformer model as a computational graph.

-----
## Method

### AE-based computational graph

For all model's module $m$ - attention head, attention layer, MLP layer or residual stream (whole transformer block) - let $E_m : \R^{d_{model}} \rightarrow \R^{d_{dict}}$ and $D_m : \R^{d_{dict}} \rightarrow \R^{d_{model}}$ be an encoder and decoder pair. These can be an arbitrary autoencoder, from a linear change of basis to a gated SAE. Each of the canonical $d_{dict}$ dimensions of the dictionary will be called a feature, and the magnitude of the activation of this feature will be called the feature's activation. We chose these features as nodes for our computational graph, they are atomic functions in the grand scheme of compositions in the model. <!-- litterally, as defined by our computational graph -->We add a sink node $y$ for the model's final output. To get the dependencies between these nodes, we use the integrated gradients attribution method as follows.

Let $x \in \mathcal{D}$ be some input from a dataset $\mathcal{D}$ to the model $\mathcal{M}$. We begin by building the expanded computational graph for this input, where we dupplicate each node for each token position in the input. Let $\lambda : \mathcal{M}, x \mapsto \lambda(\mathcal{M}, x)$ be a metric function on the model. For some module $m$ with output $x^m \in \R^{d_{model}}$, let $f^m := E_m(x^m) \in \R^{d_{dict}}$ be the vector of activations of this module's features. We use the integrated gradients attribution method to get the contribution of $f^m_i$ to $m(\mathcal{M}, x)$ :
$$ Attr(f^m_i) = \int_{0}^{1} \frac{\partial \lambda(\gamma(t))}{\partial \gamma_i(t)} \cdot \frac{\partial \gamma_i(t)}{\partial t} dt $$
where $\gamma : t \in [0, 1] \rightarrow t \cdot f^m \in \R^{d_{dict}}$ is the linear interpolation path between the baseline $0$ and the input $f^m$.

- To get the dependencies between $m$'s features and $y$, one can choose $\lambda$ to be a metric on the model's output, typically the value of the target logit.
- To get the dependencies between $m$'s features and some downstream module $m'$'s $i$-th feature, we typically choose $\lambda(\mathcal{M}, x) := f^{m'}_i$.

Then, let $\alpha$, $\beta$ be aggregation functions, typically among the $max$, $mean$ or $sum$ functions. We contract this expanded graph using $\alpha$, and aggregate graphs over many examples using $\beta$. For convenience of notation, we can include some thresholding or any variant thereof to get a sparser graph in these aggregation functions. Note that in practice, we typically threshold on the fly, during the construction of the expanded graph, for computational reasons.

Please also note that the choice for $\beta$ is not trivial, both $mean$ and $max$ have their drawbacks. $mean$ presents the risk to put edges coresponding to rare but important behaviors under the threshold, while keeping unimportant but consistent edges across examples. $max$ on the other hand keeps any rare behaviors, but as integrated gradient might be noisy - especially with such baseline and path - the amount of noise might be too high and the graph of dependencies might be too overcomplete. Future work can investigate the choice of better attribution functions or further pruning on the final graph. `FUTURE`
<!--
TODO in experiments :
    - choice of $\mathcal{D}$ (task specific - IOI, GT, ... - vs general - wikipedia, the pile -)
    - what modules to use ? `FUTURE` ? Ideally, heads and MLPs, without attention layers of transformer blocks, for maximum granularity. Is it worth going all the way to QKV separation ?
    - choice of $\alpha$ and $\beta$ - $max$ vs $mean$ (for beta), for \alpha we probably don't care ? I don't have a strong opinion or intuition on that.
    - choice of $E_m$ and $D_m$ - SAE vs gated SAE vs SVD vs LIB (`FUTURE` ?)
    - choice of baseline and $\gamma$ `FUTURE`
    - choice of thresholding
    - further pruning
    - other architecture than transformers `FUTURE`
-->


<!--To test the quality of our graph, we will use several metrics :
- faithfulness for various metrics : not perfect as the complete graph has ideal faithfulness and it is clearly not a usefull representation.
- minimality : if we prune more the graph, it's faithfulness is significantly reduced.
- sparsity : the graph should be sparse, meaning that each node depends on only a small number of ancestors. Otherwise, the representation is too complex to be useful and interpretable [TODO : sparse attr dict should help with that].
- modularity : the graph should be modular, meaning that nodes get grouped in communities. Intuitively, these communities correspond to more advanced functions. Indeed, a community is a set of nodes highly interconnected and poorly connected to the rest of the graph [TODO : formulas, sets & co]. We explore communities in deeper detail in section \ref{TODO}. Recursive communities - communities of communities - might also be interesting to explore, as based on the same intuition, they should correspond to a hierarchy of more and more complex functions and behaviors [TODO : do that too].-->


### Weight-based computational graph

`FUTURE` ?

-----
### Experiments

- SVD : empirical vs pure module weights : is it even close ? Which is more relevant ? Which's directions are more interpretable ?
    - This could be a paper of its own for some greedy people, but I think workshop paper is realistic for that... ? `FUTURE`

- run experiments specific to IOI, GT, ... tasks just to be able to compare both to other method and to ours on general tasks by benchmarking these subgraphs with all the same metrics used for general tasks.
    - validate that our methods performs at least as good as others
    - validate that these subgraphs are shit on general task and completely different from the original model, so need for a more general and still equally as interpretable representation of the model

- Graph building method variations :
    - example based graph :
        - choice of baseline and path for IG `FUTURE`
        - Choice of AE : SAEs, LIB, gated SAE, sparse attribution SAE, topk SAE, SVD, Id
        - choice of modules : resid - resid vs module - module (across layers)
        - aggregation : sum vs max (both have their drawbacks)
        - thresholding vs Topk vs Top% edges ?
        - further pruning
            - remove nodes with degree (or total weight) below some threshold
            - remove nodes with degree (or total weight) above some threshold
            - starting the graph only after some layer : starting from the embedding layer might lead to too bad results : can we start from earlier layers ?
    - weight based graph :
        `FUTURE` ?
        - module vs SVD
            - "SVD" representation is very general and can be applied to any change of basis at the output of a model that one finds relevant.

- For all graph variation :
    - Faithfullness (logit, KL, others ?) vs avg_deg vs modularity w.r.t. threshold
        - modularity with Louvain/Leiden vs modularity with correlation (either communities of nodes or of edges)
        - faithfullness/denoising/sufficiency vs completeness/noising/necessity
    - benchmark on LLMs : is it comparable to the original model ? If it is worse, is it still good enough ?
    - are neurons/nodes with more/less deggre/weight more/less interpretable ? What about neurons inside strong communities ?
    - Interp communities & their individual nodes
        - what sequences tend to activate them ?
        - tune their sizes
        - communities of communities ? Can we see an interpretable hierarchy of complexity in the functions ?
    - Let C be a community, D a dataset consisting of sequences on which this community activates.
        - kill C, measure the loss on D
        - kill V - C, measure the loss on D
        - This tells us if we can "remove" some harmful or unwanted behavior.

- Do it all for vision or language transformer as well as sleeper agent transformers. `FUTURE`

-> What is the "best" abstraction of a transformer model ?

- evolution of the graph across time (on model checkpoints) `FUTURE`
    - for SAEs based graphs, train them such that there is a 1-1 correspondence between the features of the two checkpoints. (same init and data, or init one as the result of the other and retrain (potentially for a small number of steps), or init one as the result of the other and freeze it, then learn a low rank transformation to add to that (Id + low rank) (either to modify the parameters or to modify the activations (second is probably more interesting but not sure)))

- evolution across size `FUTURE`

-----
# Exploring communities
-----

This section is largely independent of the previous one, it applies to any computational graph

-----
# Method

-----
# Experiments


-----
# Discussions, future work & conclustion
-----

