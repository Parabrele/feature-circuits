import torch
import random

from datasets import load_from_disk, load_dataset

# TODO : can I make this file an "__init__" and make this folder a pip package ?
# TODO : same for other folders in this project

# TODO : classification, QA, SV_agreement, truth datasets

boolean_expressions_path = "data/datasets/boolean_expressions/"
gp_path = "data/datasets/gp/"
gt_path = "data/datasets/gt/"
ioi_path = "data/datasets/ioi/"

class TokenBatches:
    """
    This class allows to get tokenized batches of text data.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model, # language model
                 ctx_len=128, # length of each context
                 batch_size=8192, # size of batches in which to return activations
                 device='cpu', # device on which to store the activations
                 max_number_of_yields=None, # maximum number of activations yielded by the buffer
                 clean_field='text',
                 corr_field=None,
                 distractor_field=None,
                 ):
        self.data = data
        self.model = model

        self.ctx_len = ctx_len

        self.batch_size = batch_size

        self.max_number_of_yields = max_number_of_yields
        self.nb_yields = 0

        self.clean_field = clean_field
        self.corr_field = corr_field
        self.distractor_field = distractor_field
        
        self.device = device
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        if self.max_number_of_yields is not None and self.nb_yields >= self.max_number_of_yields:
            raise StopIteration("Maximum number of yields reached")
        with torch.no_grad():
            batch = self.text_batch(min(self.batch_size, self.max_number_of_yields - self.nb_yields))
            tokenizer = self.model.tokenizer

            if self.ctx_len is not None and self.ctx_len > 0:
                clean_tokens = tokenizer(batch["clean"], return_tensors='pt', padding='max_length', truncation=True, max_length=self.ctx_len, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
                trg_idx = torch.maximum(
                    self.ctx_len - 1 - torch.randn(clean_tokens.size(0), device=clean_tokens.device).abs() * 5,
                    torch.tensor([1]).to(clean_tokens.device).expand(clean_tokens.size(0))
                ).long()
            elif self.ctx_len is None or self.ctx_len <= 0:
                clean_tokens = tokenizer(batch["clean"], return_tensors='pt', padding='max_length', truncation=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
                trg_idx = torch.zeros(clean_tokens.size(0), device=clean_tokens.device).long() - 2
                trg = clean_tokens[torch.arange(clean_tokens.size(0)), trg_idx+1]
                if self.corr_field is not None:
                    corr_tokens = tokenizer(batch["corr"], return_tensors='pt', padding='max_length', truncation=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
                    if corr_tokens.shape != clean_tokens.shape:
                        raise ValueError(f"Shape of tokenized clean -{clean_tokens.shape}- and corr -{corr_tokens.shape}- don't match. Please check that counterfactuals always have the same length as the clean text.")
                    corr_trg = corr_tokens[torch.arange(corr_tokens.size(0)), trg_idx+1]
                elif self.distractor_field is not None:
                    corr_trg = tokenizer(batch["distractor"], return_tensors='pt', padding='max_length', truncation=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
            else:
                raise ValueError("ctx_len must be None or scalar")
            
            self.nb_yields += clean_tokens.size(0)
            res = {"clean": clean_tokens, "trg_idx": trg_idx, "trg": trg}
            if self.corr_field is not None:
                res["corr"] = corr_tokens
                res["corr_trg"] = corr_trg
            elif self.distractor_field is not None:
                res["corr_trg"] = corr_trg

            return res
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        bos_token = self.model.tokenizer.bos_token
        if batch_size is None:
            batch_size = self.load_buffer_batch_size
        try:
            res = {"clean": []}
            if self.corr_field is not None:
                res["corr"] = []
            elif self.distractor_field is not None:
                res["distractor"] = []
            for _ in range(batch_size):
                data = next(self.data)
                res["clean"].append(bos_token + data[self.clean_field])
                if self.corr_field is not None:
                    res["corr"].append(bos_token + data[self.corr_field])
                elif self.distractor_field is not None:
                    res["distractor"].append(data[self.distractor_field])
            
            return res
        except StopIteration:
            raise StopIteration("End of data stream reached")

class custom_iter:
    def __init__(self, data, text_field, corr_field=None):
        self.data = data
        self.text_field = text_field if isinstance(text_field, list) else [text_field]
        self.corr_field = corr_field if isinstance(corr_field, list) or corr_field is None else [corr_field]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = next(self.data)
        if self.corr_field is not None:
            return {
                'text': " ".join([data[field] for field in self.text_field]),
                'corr': " ".join([data[field] for field in self.corr_field]),
            }
        else:
            return {
                'text': " ".join([data[field] for field in self.text_field]),
            }

def bool_buffer(
        model,
        batch_size,
        device,
        ctx_len,
        split='train',
):
    bool_data = load_from_disk(boolean_expressions_path)[split]
    
    bool_data = custom_iter(bool_data, text_field=['input', 'target'])

    buffer = TokenBatches(
        bool_data,
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=bool_data.data.num_rows,
    )

    return buffer

def gp_buffer(
        model,
        batch_size,
        device,
        ctx_len,
        split='train',
):
    gp_data = load_from_disk(gp_path)[split]
    
    gp_data = custom_iter(gp_data, text_field=['prefix', 'pronoun'], corr_field=['corr_prefix', 'corr_pronoun'])

    buffer = TokenBatches(
        gp_data,
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=gp_data.data.num_rows,
        corr_field='corr',
    )

    return buffer

def gt_buffer(
        model,
        batch_size,
        device,
        ctx_len,
        split='train',
):
    gt_data = load_from_disk(gt_path)[split]

    gt_data = custom_iter(gt_data, text_field='prefix', corr_field='corr_prefix')

    buffer = TokenBatches(
        gt_data,
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=gt_data.data.num_rows,
        corr_field='corr',
    )
    
    return buffer

def ioi_buffer(
        model,
        batch_size,
        device,
        ctx_len,
        split='train',
):
    ioi_data = load_from_disk(ioi_path)[split]
    
    ioi_data = custom_iter(ioi_data, text_field='ioi_sentences', corr_field='corr_ioi_sentences')

    buffer = TokenBatches(
        ioi_data,
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=ioi_data.data.num_rows,
        corr_field='corr',
    )
    
    return buffer

def mixture_buffer(
        model,
        batch_size,
        device,
        ctx_len,
        split='train',
):
    gp_data = load_from_disk(gp_path)[split]
    gp_data = custom_iter(gp_data, text_field=['prefix', 'pronoun'], corr_field=['corr_prefix', 'corr_pronoun'])
    gt_data = load_from_disk(gt_path)[split]
    gt_data = custom_iter(gt_data, text_field='prefix', corr_field='corr_prefix')
    ioi_data = load_from_disk(ioi_path)[split]
    ioi_data = custom_iter(ioi_data, text_field='ioi_sentences', corr_field='corr_ioi_sentences')

    class random_iter:
        def __init__(self, datasets):
            self.datasets = datasets

        def __iter__(self):
            return self
        
        def __next__(self):
            return next(random.choice(self.datasets))
        
    buffer = TokenBatches(
        random_iter([gp_data, gt_data, ioi_data]),
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=min(gp_data.data.num_rows, gt_data.data.num_rows, ioi_data.data.num_rows) * 3,
        corr_field='corr',
    )

    return buffer

def wikipedia_buffer(
    model,
    batch_size,
    device,
    ctx_len,
    split='train',
):
    dataset = load_dataset(
        "wikipedia",
        language="en",
        date="20240401",
        split=split,
        streaming=True,
        trust_remote_code=True
    ).shuffle()
    dataset = iter(dataset)

    buffer = TokenBatches(
        dataset,
        model,
        ctx_len=ctx_len,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=2**20,
    )

    return buffer
