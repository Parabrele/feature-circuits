import torch

class TokenBuffer:
    def __init__(self, 
                 data, # generator which yields text data
                 model, # language model
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 load_buffer_batch_size=512, # size of batches in which to process the data when adding to buffer
                 return_batch_size=8192, # size of batches in which to return activations
                 device='cpu', # device on which to store the activations
                 max_number_of_yields=None, # maximum number of activations yielded by the buffer
                 discard_bos=True, # whether to discard the bos token from the activations, only use it to compute the others
                 ):
        self.tokens = torch.empty(0, ctx_len, device=device).long() # size (buffer_size, ctx_len), with buffer_size in [n_ctxs // 2, n_ctxs] approximately
        self.trg_idx = torch.empty(0, device=device).long()
        self.trg = torch.empty(0, device=device).long()

        self.read = torch.zeros(0).bool()

        self.data = data
        self.model = model
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.load_buffer_batch_size = load_buffer_batch_size
        self.return_batch_size = return_batch_size
        self.device = device
        self.max_number_of_yields = max_number_of_yields
        self.nb_yields = 0
        self.discard_bos = discard_bos
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        if self.max_number_of_yields is not None and self.nb_yields >= self.max_number_of_yields:
            raise StopIteration("Maximum number of yields reached")
        with torch.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs // 2:
                self.refill()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[torch.randperm(len(unreads), device=unreads.device)[:self.return_batch_size]]
            self.read[idxs] = True
            self.nb_yields += idxs.size(0)
            return self.tokens[idxs], self.trg_idx[idxs], self.trg[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        bos_token = self.model.tokenizer.bos_token
        if batch_size is None:
            batch_size = self.load_buffer_batch_size
        try:
            return [
                bos_token + next(self.data)['text'] for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def refill(self):
        self.tokens = self.tokens[~self.read]
        self.trg_idx = self.trg_idx[~self.read]
        self.trg = self.trg[~self.read]

        while len(self.tokens) < self.n_ctxs:
            tokenizer = self.model.tokenizer
            batch = self.text_batch()
            tokens = tokenizer(batch, return_tensors='pt', padding='max_length', truncation=True, max_length=self.ctx_len, return_attention_mask=False, return_token_type_ids=False)['input_ids'].to(self.device)
            # max(ctx_len - abs(randn, sigma = 5), 1 if self.discard_bos else 0)
            trg_idx = torch.maximum(
                self.ctx_len - 1 - torch.randn(tokens.size(0), device=tokens.device).abs() * 5,
                torch.tensor([1 if self.discard_bos else 0]).to(tokens.device).expand(tokens.size(0))
            ).long()
            
            self.tokens = torch.cat([self.tokens, tokens], dim=0)
            self.trg_idx = torch.cat([self.trg_idx, trg_idx], dim=0)
            self.trg = torch.cat([self.trg, tokens[torch.arange(tokens.size(0)), trg_idx+1]], dim=0)
            self.read = torch.zeros(len(self.tokens), dtype=torch.bool, device=self.device)

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()