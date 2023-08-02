import math
from typing import Tuple

import torch
from .transformer import PositionalEncoding
from torch import nn, Tensor
import torch.nn.functional as F
from torch import jit
from torchaudio.models import Conformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from typing import Optional

'''

#Input [Batch, Time, Neuron]
torchaudio.models.Conformer(input_dim: int, 
                            num_heads: int,
                            ffn_dim: int,
                            num_layers: int,
                            depthwise_conv_kernel_size: int,
                            dropout: float = 0.0,
                            use_group_norm: bool = False,
                            convolution_first: bool = False)
'''

class ConformerModel(nn.Module):

    def __init__(self,
                 n_input: int,
                 n_output : int,
                 d_model: int=256,
                 ffn_dim: int=128,
                 depthwise_conv_kernel_size: int = 31,
                 nhead: int=4,
                 nlayers: int=3,
                 dropout: float = 0.1,
                 use_group_norm: bool = False,
                 convolution_first: bool = False):
        
        super().__init__()
        self.model_type = 'Conformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.conformer = Conformer(
                             input_dim=d_model,
                             num_heads=nhead,
                             ffn_dim=ffn_dim,
                             num_layers=nlayers,
                             depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                             use_group_norm = use_group_norm,
                             convolution_first = convolution_first
                         )
        self.d_model = d_model
        self.n_output = n_output
        self.n_input = n_input

        self.encoder = nn.Linear(n_input, d_model)
        self.decoder = nn.Linear(d_model, n_output)

        self.init_weights()

    def init_weights(self) -> None:
        initrange_enc = 1/math.sqrt(self.n_input)
        initrange_dec = 1/math.sqrt(self.d_model)

        self.encoder.weight.data.uniform_(-initrange_enc, initrange_enc)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange_dec, initrange_dec)

    #@jit.script_method
    def forward(self, x: Tensor):
        #x is [Batch, Time, Feature]
        #lengths is [Batch,]
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        B,T,_ = x.shape
        lengths = T*torch.ones(B,device = x.device)
        #print(f"Lengths: {lengths.shape}: {lengths}")
        output,_ = self.conformer(x,lengths)
        output = self.decoder(output)
        #Output is [Batch, Time, Feature]
        return output



if __name__ == "__main__":
    from torchinfo import summary

    d = 128 #n_input = n_output
    x = torch.rand(32,128,256, device = 'cuda') # Batch Channel/Feature SeqLength [Batch, Neuron, Time]
    #Transformer with batch first takes Batch SeqLength Channel/Feature
    model = ConformerModel(d, d, 
                           d_model = 256,
                           ffn_dim =128,
                           depthwise_conv_kernel_size = 31,
                           nhead =4,
                           nlayers =16,
                           dropout = 0.1,
                           use_group_norm  = False,
                           convolution_first = True).to('cuda')
    
    print("input",x.shape)
    x = torch.permute(x,[0,2,1])
    input = x
    x = model(x)
    print("out of conformer",x.shape)
    x = torch.permute(x,[0,2,1])
    print("permute",x.shape)

    summary(model,input_data=input)