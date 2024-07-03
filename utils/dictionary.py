"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn

class Dictionary(ABC):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """
    dict_size : int # number of features in the dictionary
    activation_dim : int # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass
    
    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))
    
    def decode(self, f):
        return self.decoder(f) + self.bias
    
    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None: # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat
        
        else: # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(f_ghost) # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

class LinearDictionary(Dictionary, nn.Module):
    """
    A linear dictionary, i.e. two matrices E and D.
    """
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.E = nn.Parameter(t.randn(dict_size, activation_dim))
        self.D = nn.Parameter(self.E.t())

    def encode(self, x):
        return t.matmul(x - self.bias, self.E.t())
    
    def decode(self, f):
        return t.matmul(f, self.D.t()) + self.bias
    
    def forward(self, x, output_features=False, ghost_mask=None):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

def whitening(cov):
    """
    x : Tensor
        input of dimension (n, d)
    """
    U, Lambda, _ = t.svd(cov)
    # if Lambda is below eps, set it to 0
    eps = 1e-6
    Lambda[abs(Lambda) < eps] = 0
    Lambda_sqrt_inv = t.zeros(Lambda.shape[0], Lambda.shape[0], device=Lambda.device)
    Lambda_sqrt_inv[
        t.arange(Lambda.shape[0], device=Lambda.device)[Lambda != 0],
        t.arange(Lambda.shape[0], device=Lambda.device)[Lambda != 0]
    ] = 1 / t.sqrt(Lambda[Lambda != 0])
    Lambda_sqrt = t.zeros(Lambda.shape[0], Lambda.shape[0], device=Lambda.device)
    Lambda_sqrt[
        t.arange(Lambda.shape[0], device=Lambda.device)[Lambda != 0],
        t.arange(Lambda.shape[0], device=Lambda.device)[Lambda != 0]
    ] = t.sqrt(Lambda[Lambda != 0])

    W = U @ Lambda_sqrt_inv @ U.T
    W_inv = U @ Lambda_sqrt @ U.T
    return W, W_inv
        
class WhiteDict(LinearDictionary):
    """
    Just another name for a linear dictionary, but to distinguish it.
    """
    def __init__(self, activation_dim, mean, cov):
        super().__init__(activation_dim, activation_dim)
        W, W_inv = whitening(cov)
        self.bias = nn.Parameter(mean)
        self.E = nn.Parameter(W)
        self.D = nn.Parameter(W_inv)

class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """
    def __init__(self, activation_dim):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x
    
    def decode(self, f):
        return f
    
    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x