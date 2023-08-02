import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import jit
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from typing import Optional


class TransformerModel(nn.Module):

    def __init__(self, n_input: int, n_output : int, d_model: int=256, nhead: int=4, d_hid: int=1024,
                nlayers: int=3, dropout: float = 0.1, batch_first: bool=False):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(n_input, d_model)
        self.d_model = d_model
        self.n_output = n_output
        self.n_input = n_input

        self.decoder = nn.Linear(d_model, n_output)
        self.batch_first = batch_first

        self.init_weights()

    def init_weights(self) -> None:
        initrange_enc = 1/math.sqrt(self.n_input)
        initrange_dec = 1/math.sqrt(self.d_model)

        self.encoder.weight.data.uniform_(-initrange_enc, initrange_enc)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange_dec, initrange_dec)

    #@jit.script_method
    def forward_batch_not_first(self, x: Tensor, src_mask: Optional[Tensor] = None):
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask = src_mask)
        output = self.decoder(output)
        return output

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, n_input]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        if self.batch_first:
            x = x.permute([1,0,2])

        output = self.forward_batch_not_first(x, src_mask)

        if self.batch_first:
            output = output.permute([1,0,2])
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    from torchinfo import summary

    d = 128 #n_input = n_output
    x = torch.rand(3,128,256) # Batch Channel/Feature SeqLength
    #Transformer with batch first takes Batch SeqLength Channel/Feature
    model = TransformerModel(d, d, batch_first=True, d_model=256, nhead=4, nlayers=3, d_hid=1024)
    
    print("input",x.shape)
    x = torch.permute(x,[0,2,1])
    input = x
    x = model(x)
    print("out of transformer",x.shape)
    x = torch.permute(x,[0,2,1])
    print("permute",x.shape)

    summary(model,input_data=input)