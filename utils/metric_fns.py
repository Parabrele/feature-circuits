import torch

from nnsight.models.UnifiedTransformer import UnifiedTransformer

# TODO : clean_logits should have the same size as logits

def metric_fn_logit(model, kwargs={}):
    """
    return the target logit
    requires trg_idx and at least one of trg_pos or trg_neg
    """
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    trg_idx = kwargs.get('trg_idx', None)
    trg_pos = kwargs.get('trg_pos', None)
    trg_neg = kwargs.get('trg_neg', None)

    if trg_pos is None and trg_neg is None:
        raise ValueError("Wrong arguments in metric_fn_logit")

    logits = module.output[torch.arange(trg_idx.numel()), trg_idx]
    res = 0
    if trg_pos is not None:
        if isinstance(trg_pos, torch.Tensor):
            res += logits[trg_pos]
        elif isinstance(trg_pos, list):
            for i, tokens in enumerate(trg_pos):
                res += logits[i, tokens].sum()
    if trg_neg is not None:
        if isinstance(trg_neg, torch.Tensor):
            res -= logits[trg_neg]
        elif isinstance(trg_neg, list):
            for i, tokens in enumerate(trg_neg):
                res -= logits[i, tokens].sum()
    return res

    if len(trg) == 2:
        trg_idx, trg_token = trg
        if isinstance(trg_token, torch.Tensor):
            if clean_logits is None:
                return module.output[torch.arange(trg_idx.numel()), trg_idx, trg_token] # (b, s, d_vocab) -> (b)
            else:
                g = module.output[torch.arange(trg_idx.numel()), trg_idx, trg_token]
                m = clean_logits[torch.arange(trg_idx.numel()), trg_idx, trg_token]
                e = empty_logits[torch.arange(trg_idx.numel()), trg_idx, trg_token]
                return (g-e)/(m-e)
        elif isinstance(trg_token, list):
            res = []
            for i, tokens in enumerate(trg_token):
                if clean_logits is None:
                    res.append(module.output[i, trg_idx[i], tokens].sum())
                else:
                    g = module.output[i, trg_idx[i], tokens].sum()
                    m = clean_logits[i, trg_idx[i], tokens].sum()
                    e = empty_logits[i, trg_idx[i], tokens].sum()
                    res.append((g-e)/(m-e))
            return torch.stack(res)
        else:
            raise ValueError("This should not happen 5698456984")
    elif len(trg) == 3:
        trg_idx, trg_pos, trg_neg = trg
        if isinstance(trg_pos, torch.Tensor):
            if clean_logits is None:
                if trg_neg is not None:
                    return module.output[torch.arange(trg_idx.numel()), trg_idx, trg_pos] - module.output[torch.arange(trg_idx.numel()), trg_idx, trg_neg]
                else:
                    return module.output[torch.arange(trg_idx.numel()), trg_idx, trg_pos]
            else:
                if trg_neg is not None:
                    g = module.output[torch.arange(trg_idx.numel()), trg_idx, trg_pos] - module.output[torch.arange(trg_idx.numel()), trg_idx, trg_neg]
                    m = clean_logits[torch.arange(trg_idx.numel()), trg_idx, trg_pos] - clean_logits[torch.arange(trg_idx.numel()), trg_idx, trg_neg]
                    e = empty_logits[torch.arange(trg_idx.numel()), trg_idx, trg_pos] - empty_logits[torch.arange(trg_idx.numel()), trg_idx, trg_neg]
                else:
                    g = module.output[torch.arange(trg_idx.numel()), trg_idx, trg_pos]
                    m = clean_logits[torch.arange(trg_idx.numel()), trg_idx, trg_pos]
                    e = empty_logits[torch.arange(trg_idx.numel()), trg_idx, trg_pos]
                return (g-e)/(m-e)
        
        elif isinstance(trg_pos, list):
            res = []
            if clean_logits is None:
                if trg_neg is not None:
                    for i, (pos, neg) in enumerate(zip(trg_pos, trg_neg)):
                        res.append(module.output[i, trg_idx[i], pos].sum() - module.output[i, trg_idx[i], neg].sum())
                else:
                    for i, pos in enumerate(trg_pos):
                        res.append(module.output[i, trg_idx[i], pos].sum())
                return torch.stack(res)
            else:
                if trg_neg is not None:
                    for i, (pos, neg) in enumerate(zip(trg_pos, trg_neg)):
                        g = module.output[i, trg_idx[i].cpu(), pos].sum() - module.output[i, trg_idx[i].cpu(), neg].sum()
                        m = clean_logits[i, trg_idx[i].cpu(), pos].sum() - clean_logits[i, trg_idx[i].cpu(), neg].sum()
                        e = empty_logits[i, trg_idx[i].cpu(), pos].sum() - empty_logits[i, trg_idx[i].cpu(), neg].sum()
                        res.append((g-e)/(m-e))
                else:
                    for i, pos in enumerate(trg_pos):
                        g = module.output[i, trg_idx[i].cpu(), pos].sum()
                        m = clean_logits[i, trg_idx[i].cpu(), pos].sum()
                        e = empty_logits[i, trg_idx[i].cpu(), pos].sum()
                        res.append((g-e)/(m-e))
                return torch.stack(res)
    else:
        raise ValueError("This should not happen")

def metric_fn_KL(model, kwargs={}):
    """
    return the KL divergence between the current logits and a target clean logits
    requires trg_idx and clean_logits
    """
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out

    clean_logits = kwargs.get('clean_logits', None)
    trg_idx = kwargs.get('trg_idx', None)
    if trg_idx is None or clean_logits is None:
        raise ValueError("Wrong arguments in metric_fn_KL")
    
    batch_size = trg_idx.numel()

    logits = module.output[torch.arange(batch_size), trg_idx] # (b, s, d_model) -> (b, d_model)
    clean_logits = clean_logits[torch.arange(batch_size), trg_idx]
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits, dim=-1),
        torch.nn.functional.log_softmax(clean_logits, dim=-1),
        reduction='none',
        log_target=True
    ).sum(dim=-1)

def metric_fn_statistical_distance(model, kwargs={}):
    """
    return the statistical distance between the current logits and a target clean logits
    """
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    
    trg_idx = kwargs.get('trg_idx', None)
    clean_logits = kwargs.get('clean_logits', None)
    if trg_idx is None or clean_logits is None:
        raise ValueError("Wrong arguments in metric_fn_statistical_distance")
    
    batch_size = trg_idx.numel()
    logits = module.output[torch.arange(batch_size), trg_idx] # (b, s, d_model) -> (b, d_model)
    clean_logits = clean_logits[torch.arange(batch_size), trg_idx]
    sigma = torch.nn.functional.softmax
    return (0.5 * torch.abs(sigma(logits, dim=-1) - sigma(clean_logits, dim=-1))).sum(dim=-1)

def metric_fn_acc(model, trg=None, clean_logits=None):
    """
    return 1 if the model's prediction is correct, 0 otherwise
    """
    raise NotImplementedError("This fails for some reason, nnsight don't want to do the last part implicitely")
    if trg is None:
        raise ValueError("trg must be provided")
    if clean_logits is None:
        raise ValueError("clean_logits must be provided")
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    batch_size = trg[0].numel()
    logits = module.output[torch.arange(batch_size), trg[0]] # (b, s, d_model) -> (b, d_model)
    clean_logits = clean_logits[torch.arange(batch_size), trg[0]]
    if isinstance(trg[1], torch.Tensor):
        g = (logits.argmax(dim=-1) == trg[1]).float()
        m = (clean_logits.argmax(dim=-1) == trg[1]).float()
        return 1 - ((g + m) - 2 * (g * m))
    elif isinstance(trg[1], list):
        res = []
        for i, tokens in enumerate(trg[1]):
            g = -1
            m = -1
            for token in tokens:
                if g == -1:
                    g = (logits[i].argmax().item() == token)
                    m = (clean_logits[i].argmax().item() == token)
                else:
                    g = g or (logits[i].argmax().item() == token)
                    m = m or (clean_logits[i].argmax().item() == token)
            g = g.item()
            m = m.item()
            res.append( 1 - ((g + m) - 2 * (g * m)) )
        return torch.tensor(res).float()

def metric_fn_MRR(model, trg=None, clean_logits=None):
    """
    default : return 1/rank of the correct answer
    """
    raise NotImplementedError("This fails for some reason, nnsight don't want to do the last part implicitely")
    if trg is None:
        raise ValueError("trg must be provided")
    if clean_logits is None:
        raise ValueError("clean_logits must be provided")
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    batch_size = trg[0].numel()
    logits = module.output[torch.arange(batch_size), trg[0]]
    clean_logits = clean_logits[torch.arange(batch_size), trg[0]]
    if isinstance(trg[1], torch.Tensor):
        g = (logits.argsort(dim=-1, descending=True) == trg[1].unsqueeze(-1)).float().argmax(dim=-1).float()
        m = (clean_logits.argsort(dim=-1, descending=True) == trg[1].unsqueeze(-1)).float().argmax(dim=-1).float()
        return (m+1)/(g+1)
    else:
        res = []
        for i, tokens in enumerate(trg[1]):
            g = -1
            m = -1
            for token in tokens:
                if g == -1:
                    g = (logits[i].argsort(descending=True) == token).float().argmax().float()
                    m = (clean_logits[i].argsort(descending=True) == token).float().argmax().float()
                else:
                    g = min(g, (logits[i].argsort(descending=True) == token).float().argmax().float())
                    m = min(m, (clean_logits[i].argsort(descending=True) == token).float().argmax().float())
            res.append( (m + 1) / (g + 1) )
        return torch.tensor(res)
