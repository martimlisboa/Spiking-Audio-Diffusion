import torch.nn as nn
import torch
import os

import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser

import numpy as np
from .rsnn import RSNNLayer, NaiveSpikeLayer
from .multitask_splitter import NormalizedMultiTaskSplitter
from typing import Tuple, Optional
import torchaudio.transforms as TT



from .encodec.modules import SEANetEncoder
from .encodec import quantization as qt
from .encodec.quantization.vq import QuantizedResult

class SpikingEncodecEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder = SEANetEncoder(dimension = args.encodec_dim,ratios = args.encodec_ratios)
        self.downsample_factor = 1
        for r in self.encoder.ratios: self.downsample_factor = self.downsample_factor*r 
    
        self.to_spikes = NaiveSpikeLayer(input_dim = self.encoder.dimension, embed_dim=args.bottleneck_dim)


        self.rep_loss_type = args.rep_loss_type[0]
        self.firing_rate_threshold = 0
        if self.rep_loss_type == "relu":
            self.firing_rate_threshold = args.firing_rate_threshold/args.bottleneck_dim
        self.out_channels = args.bottleneck_dim
        if args.lstm:
            self.lstm = nn.LSTM(
                    batch_first=True,
                    input_size=args.bottleneck_dim,
                    hidden_size=args.lstm_hidden_size,
                    num_layers=2,
                    bidirectional=False,
                    dropout=0.3
                    )
            self.out_channels = args.lstm_hidden_size
        else:
            self.lstm = nn.Identity()


        print("Quantizing Encodec Encoder")
        print(f"Out Channels:{self.out_channels}")
        print(f"Downsample Factor:{self.out_channels}")
        print(f"Encodec Dimension:{self.encoder.dimension}")
        print(f"Nr of Neurons:{args.bottleneck_dim}")
        print(f"Representation Loss Type:{self.rep_loss_type}")
        if self.rep_loss_type == "relu":
            print(f"Firing rate threshold per neuron: nu = {self.firing_rate_threshold}")
        print(f"LSTM:{args.lstm}")
        print("\n") 

    def to_spike(self,x):
        x = torch.permute(x,[0,2,1])
        x = self.to_spikes(x)
        x = torch.permute(x,[0,2,1])
        return x


    def spike_loss(self, x):
        #x [Batch,Neuron,Time]
        representation_loss = 0
        if self.rep_loss_type == 'mean':
            representation_loss = x.mean()
        elif self.rep_loss_type == 'mean2':
            squared_firing = x.mean(dim = 2)**2 
            representation_loss = squared_firing.mean()
        elif self.rep_loss_type == 'relu':
            firing_rate = x.mean(dim = 2) #average over time =  [Batch, Neuron Firing rates]
            relu = torch.maximum(torch.zeros_like(firing_rate), firing_rate - self.firing_rate_threshold) #Cap N \nu at firing rate threshold
            representation_loss = relu.mean()
            #print(f"Firing Rate: {firing_rate}")

            #print(f"Loss: {representation_loss}")
        return representation_loss

    def to_lstm(self,x):
        x = torch.permute(x,[0,2,1])
        x,_ = self.lstm(x)
        x = torch.permute(x,[0,2,1])
        return x
    def forward(self,x ,with_info: bool=False):
        #Encodec Encoder
        x = self.encoder(x)
        #print(f"shape after encoder: {x.shape}")

        #Naive Spiking Module
        x = self.to_spike(x)
        rep_loss = self.spike_loss(x)
        info = {'spikes': x,'rep_loss':rep_loss} #Store the spikes in info



        x = self.to_lstm(x)
        #print(f"Shape after LSTM: {x.shape}")


        return (x, info) if with_info else x
    

class QuantizingEncodecEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder = SEANetEncoder(dimension = args.encodec_dim,ratios = args.encodec_ratios)
        self.quantizer = qt.ResidualVectorQuantizer(dimension = args.bottleneck_dim,n_q = 8)

        self.out_channels = self.quantizer.dimension
        self.downsample_factor = 1
        for r in self.encoder.ratios: self.downsample_factor = self.downsample_factor*r 

        self.frame_rate = int(args.sample_rate/self.downsample_factor)

        print("Quantizing Encodec Encoder")
        print(f"Out Channels:{self.out_channels}")
        print(f"Downsample Factor:{self.out_channels}")
        print(f"Frame Rate:{self.frame_rate} frames/s")
        print("\n")




    def forward(self,x,with_info: bool=False):
        #Encodec Encoder:
        x = self.encoder(x)
        #Residual Vector Quantization
        q_result = self.quantizer(x,frame_rate = self.frame_rate) #Frame Rate is completely useless here but has to be passed

        x = q_result.quantized
        info = {"codes": q_result.codes, "rep_loss": q_result.penalty}

        return (x, info) if with_info else x


        
