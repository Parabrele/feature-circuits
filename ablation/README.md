# Ablation
-----
This folder contains two main functions :
- node ablation
- edge ablation

These functions are used to run a forward pass on a computational graph representing a model.

-----
They both take as input
- a `clean` and a `patch`ed model input, along with an `ablation_fn` that's applied to the hidden states of the patch.
    - If `patch` is `None`, clean is then used as the patch, and will be passed through the ablation function.
- a `model` and (callable) `submodules`, with corresponding `dictionaries`
    - `dictionaries` can be a dict of SAEs, changes of basis, ... The only requirement is an encode and a decode function.
- a metric function and its kwargs : this is what you want to be returned by this forward pass. It can be the final logits or any metric on the model's output or internal states.

-----

Node ablation needs a `nodes` dict, with same keys as `dictionaries` that contains `SparseAct`s with boolean entries, specifying which dimensions of the hidden states given by each dictionary to keep.

-----

Edge ablation needs a `graph` dict1 of dicts2, with dict1 keys being upstream submodules, dict2 keys being downstream submodules, and dict2 values being `SparseAct`s with boolean entries, specifying which edges to keep.