import numpy as np
import os
import torch
import torch.nn as nn
from torchinfo import summary

from models.encoder_models import SpikingEncodecEncoder,RecursiveSpikingEncodecEncoder
from parser import make_parser,make_scaffold_parser,dotdict,override_args_from_dotdict
from model import SpikingAudioDiffusion


model_1 = dotdict({
    "encoder":["mu_encodec"],
    "model_dir":["dummy_model_dir"],
    "data_dirs":["dummy_data_dirs"],
    "bottleneck_dim":128,
    "transformer": False,
    "conformer": True,
    "rep_loss_type": ['brute'],
    "sequence_length": 32768
})


if __name__ == "__main__":

    '''
    
    #TEST FOR MU ENCODEC
    args = override_args_from_dotdict(model_1)
    model = SpikingAudioDiffusion(args).to('cuda')
    x = torch.randn(16, 1, 2**15).to('cuda')
    mu = torch.randint(32,(16,)).to('cuda')
    print(f"Input Shape: {x.shape}")
    print(f"mu: {mu}")
    B,C,T = x.shape
    summary(model,input_data = (x,mu))
    '''


    

    #x_context, x_diff = x[:,:,:T//2].to('cuda'), x[:,:,T//2:].to('cuda')
    #summary(model.autoencoder.encoder,input_data = (x_context,x_diff))
    '''
    x_in = torch.randn(16, 1, 2**16).to('cuda')
    s,e,i = model.autoencode(x_in)

    x = torch.randn(16,1,2**15)
    xpre = torch.randn(16,1,2**15)
    args = override_args_from_dotdict(model_1)
    model = RecursiveSpikingEncodecEncoder(args)

    output = model(x,xpre,with_info = True)'''