import torch
import random

from datasets import load_from_disk, load_dataset

# TODO : return attn mask !

boolean_expressions_path = "/home/pyllm/dhimoila/feature-circuits-1/data/datasets/boolean_expressions/"
gp_path = "/home/pyllm/dhimoila/feature-circuits-1/data/datasets/gp/"
gt_path = "/home/pyllm/dhimoila/feature-circuits-1/data/datasets/gt/"
ioi_path = "/home/pyllm/dhimoila/feature-circuits-1/data/datasets/ioi/"

class TokenBatches:
    """
    This class allows to get tokenized batches of text data.
    /!\ Tokenizer should pad on the LEFT ! The target token are supposed to be the last ones. /!\ 
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
                 good_field=None,
                 bad_field=None,
                 ):
        self.data = data
        self.model = model

        self.ctx_len = ctx_len

        self.batch_size = batch_size

        self.max_number_of_yields = max_number_of_yields
        self.nb_yields = 0

        self.clean_field = clean_field
        self.corr_field = corr_field
        self.good_field = good_field
        self.bad_field = bad_field
        
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

            # Deal with wikipedia or datasets with too long text
            if self.ctx_len is not None and self.ctx_len > 0:
                clean_tokens = tokenizer(batch["clean"], return_tensors='pt', padding='max_length', truncation=True, max_length=self.ctx_len, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
                trg_idx = torch.maximum(
                    self.ctx_len - 1 - torch.randn(clean_tokens.size(0), device=clean_tokens.device).abs() * 5,
                    torch.tensor([1]).to(clean_tokens.device).expand(clean_tokens.size(0))
                ).long()
                trg = clean_tokens[torch.arange(clean_tokens.size(0)), trg_idx+1]

            # Deal with templates and toy tasks
            elif self.ctx_len is None or self.ctx_len <= 0:
                clean_tokens = tokenizer(batch["clean"], return_tensors='pt', padding=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
                
                # good_field is used when there might be several correct answers. In this case, the clean tokens are a prefix without the answer appended.
                if self.good_field is None:
                    trg_idx = torch.zeros(clean_tokens.size(0), device=clean_tokens.device).long() - 2
                    trg = clean_tokens[torch.arange(clean_tokens.size(0)), trg_idx+1]
                else:
                    trg_idx = torch.zeros(clean_tokens.size(0), device=clean_tokens.device).long() - 1
                    trg = []
                    for i, good in enumerate(batch["good"]):
                        trg.append(tokenizer(good, return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)[:, -1])
                    # can't stack these as they may have different lengths
                
                # Deal with counterfactuals
                if self.corr_field is not None:
                    corr_tokens = tokenizer(batch["corr"], return_tensors='pt', padding=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)

                    # Check that the counterfactuals have the same length as the clean text, otherwise patching won't work
                    if corr_tokens.shape != clean_tokens.shape:
                        print(clean_tokens)
                        print(corr_tokens)
                        raise ValueError(f"Shape of tokenized clean {clean_tokens.shape} and corr {corr_tokens.shape} don't match. Please check that counterfactuals always have the same length as the clean text.")
                    
                    # bad_field is used when there might be several wrong answers. In this case, the counterfactual tokens are a prefix without the answer appended.
                    if self.bad_field is None:
                        corr_trg = corr_tokens[torch.arange(corr_tokens.size(0)), trg_idx+1]
                    else:
                        corr_trg = []
                        for i, bad in enumerate(batch["bad"]):
                            corr_trg.append(tokenizer(bad, return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)[:, -1])
                        # can't stack these as they may have different lengths

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
            if self.good_field is not None:
                res["good"] = []
            if self.bad_field is not None:
                res["bad"] = []
            for _ in range(batch_size):
                data = next(self.data)
                res["clean"].append(bos_token + data[self.clean_field])
                if self.corr_field is not None:
                    res["corr"].append(bos_token + data[self.corr_field])
                if self.good_field is not None:
                    if data["add_space"]:
                        res["good"].append([(" " + trg) for trg in data[self.good_field]])
                    else:
                        res["good"].append(data[self.good_field])
                if self.bad_field is not None:
                    if data["add_space"]:
                        res["bad"].append([(" " + trg) for trg in data[self.bad_field]])
                    else:
                        res["bad"].append(data[self.bad_field])
            
            return res
        except StopIteration:
            raise StopIteration("End of data stream reached")

def unpack_batch(batch):
    """
    Unpack a batch of activations
    """
    tokens = batch["clean"]
    trg_idx = batch["trg_idx"]
    trg = batch["trg"]
    corr = None
    if "corr" in batch:
        corr = batch["corr"]
    corr_trg = None
    if "corr_trg" in batch:
        corr_trg = batch["corr_trg"]
    return tokens, trg_idx, trg, corr, corr_trg

class custom_iter:
    def __init__(self, data, text_field, corr_field=None, good_field=None, bad_field=None, add_space=False):
        self.data = data
        self.text_field = text_field if isinstance(text_field, list) else [text_field]
        self.corr_field = corr_field if isinstance(corr_field, list) or corr_field is None else [corr_field]
        self.good_field = good_field
        self.bad_field = bad_field
        self.add_space = add_space
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = next(self.data)
        res = {
            'text': " ".join([data[field] for field in self.text_field]),
            'add_space': self.add_space,
        }
        if self.corr_field is not None:
            res['corr'] = " ".join([data[field] for field in self.corr_field])
        if self.good_field is not None:
            res['good'] = data[self.good_field] if isinstance(data[self.good_field], list) else [data[self.good_field]]
        if self.bad_field is not None:
            res['bad'] = data[self.bad_field] if isinstance(data[self.bad_field], list) else [data[self.bad_field]]
            
        return res

def bool_buffer(
        model,
        batch_size,
        device,
        ctx_len=None,
        split='train',
):
    bool_data = load_from_disk(boolean_expressions_path)[split].shuffle()
    bool_iter = custom_iter(iter(bool_data), text_field=['input', 'target'])

    print("Bool num rows", bool_data.num_rows)

    buffer = TokenBatches(
        bool_iter,
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=bool_data.num_rows,
    )

    return buffer

def gp_buffer(
        model,
        batch_size,
        device,
        ctx_len=None,
        split='train',
        perm = None,
):
    gp_data = load_from_disk(gp_path)[split].shuffle()
    gp_iter = custom_iter(iter(gp_data), text_field='prefix', corr_field='corr_prefix', good_field='pronoun', bad_field='corr_pronoun', add_space=True)#, text_field=['prefix', 'pronoun'], corr_field=['corr_prefix', 'corr_pronoun'])

    buffer = TokenBatches(
        gp_iter,
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=gp_data.num_rows,
        corr_field='corr',
        good_field='good',
        bad_field='bad',
    )

    return buffer

def gt_buffer(
        model,
        batch_size,
        device,
        ctx_len=None,
        split='train',
        perm = None,
):
    gt_data = load_from_disk(gt_path)[split].shuffle()
    gt_iter = custom_iter(iter(gt_data), text_field='prefix', corr_field='corr_prefix', good_field='good', bad_field='bad')

    buffer = TokenBatches(
        gt_iter,
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=gt_data.num_rows,
        corr_field='corr',
        good_field='good',
        bad_field='bad',
    )
    
    return buffer

def ioi_buffer(
        model,
        batch_size,
        device,
        ctx_len=None,
        split='train',
        perm = None,
):
    ioi_data = load_from_disk(ioi_path)[split]
    #print("IOI num rows", ioi_data.num_rows)
    if perm is None:
        ioi_data = ioi_data.shuffle()
    else:
        ioi_data = ioi_data.select(perm)
    ioi_iter = custom_iter(iter(ioi_data), text_field='ioi_sentences', corr_field='corr_ioi_sentences', good_field='a', bad_field='b', add_space=True)#, text_field=['ioi_sentences', 'a'], corr_field=['corr_ioi_sentences', 'b'])

    buffer = TokenBatches(
        ioi_iter,
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=ioi_data.num_rows,
        corr_field='corr',
        good_field='good',
        bad_field='bad',
    )
    
    return buffer

def mixture_buffer(
        model,
        batch_size,
        device,
        ctx_len=None,
        split='train',
):
    gp_data = load_from_disk(gp_path)[split].shuffle()
    gp_iter = custom_iter(iter(gp_data), text_field='prefix', corr_field='corr_prefix', good_field='pronoun', bad_field='corr_pronoun', add_space=True)#, text_field=['prefix', 'pronoun'], corr_field=['corr_prefix', 'corr_pronoun'])
    gt_data = load_from_disk(gt_path)[split].shuffle()
    gt_iter = custom_iter(iter(gt_data), text_field='prefix', corr_field='corr_prefix', good_field='good', bad_field='bad')
    ioi_data = load_from_disk(ioi_path)[split].shuffle()
    ioi_iter = custom_iter(iter(ioi_data), text_field='ioi_sentences', corr_field='corr_ioi_sentences', good_field='a', bad_field='b', add_space=True)#, text_field=['ioi_sentences', 'a'], corr_field=['corr_ioi_sentences', 'b'])

    class random_iter:
        def __init__(self, datasets):
            self.datasets = datasets

        def __iter__(self):
            return self
        
        def __next__(self):
            return next(random.choice(self.datasets))
        
    buffer = TokenBatches(
        random_iter([gp_iter, gt_iter, ioi_iter]),
        model,
        ctx_len=None,
        batch_size=batch_size,
        device=device,
        max_number_of_yields=min(gp_data.num_rows, gt_data.num_rows, ioi_data.num_rows) * 3,
        corr_field='corr',
        good_field='good',
        bad_field='bad',
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

if __name__ == "__main__":
    ioi_dataset = load_from_disk(ioi_path)
    print(ioi_dataset)
    print("Hehehe")
    print(ioi_dataset['train'])
    print("Hohoho")
    print(ioi_dataset['train'][0])

    gp_dataset = load_from_disk(gp_path)
    print(gp_dataset)
    print("Hehehe")
    print(gp_dataset['train'])
    print("Hohoho")
    print(gp_dataset['train'][0])

    gt_dataset = load_from_disk(gt_path)
    print(gt_dataset)
    print("Hehehe")
    print(gt_dataset['train'])
    print("Hohoho")
    print(gt_dataset['train'][0])