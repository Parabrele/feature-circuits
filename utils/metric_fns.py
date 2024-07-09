import torch

from nnsight.models.UnifiedTransformer import UnifiedTransformer

def metric_fn_logit(model, trg=None):
    """
    return the target logit
    """
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    if trg is None:
        raise ValueError("trg must be provided")
    if len(trg) == 2:
        return module.output[torch.arange(trg[0].numel()), trg[0], trg[1]]
    elif len(trg) == 3:
        if trg[2] is not None:
            return module.output[torch.arange(trg[0].numel()), trg[0], trg[1]] - module.output[torch.arange(trg[0].numel()), trg[0], trg[2]]
        else:
            return module.output[torch.arange(trg[0].numel()), trg[0], trg[1]]
    else:
        raise ValueError("This should not happen")

def metric_fn_KL(model, trg=None, clean_logits=None):
    """
    return the KL divergence between the current logits and a target clean logits
    """
    if clean_logits is None:
        raise ValueError("clean_logits must be provided")
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    logits = module.output[torch.arange(trg[0].numel()), trg[0]] # (b, s, d_model) -> (b, d_model)
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits, dim=-1),
        torch.nn.functional.log_softmax(clean_logits, dim=-1),
        reduction='none',
        log_target=True
    ).sum(dim=-1)

def metric_fn_acc(model, trg=None):
    """
    return 1 if the model's prediction is correct, 0 otherwise
    """
    if trg is None:
        raise ValueError("trg must be provided")
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    logits = module.output[torch.arange(trg[0].numel()), trg[0]]
    return (logits.argmax(dim=-1) == trg[1]).float()

def metric_fn_MRR(model, trg=None):
    """
    default : return 1/rank of the correct answer
    """
    if trg is None:
        raise ValueError("trg must be provided")
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    logits = module.output[torch.arange(trg[0].numel()), trg[0]]
    return 1 / (1 + (logits.argsort(dim=-1, descending=True) == trg[1].unsqueeze(-1)).float().argmax(dim=-1).float())
