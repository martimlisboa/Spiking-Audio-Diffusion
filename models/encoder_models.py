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


from .transformer import TransformerModel, generate_square_subsequent_mask
from .binary_quantizer import BinaryQuantizer
from .encodec.modules import SEANetEncoder
from .encodec import quantization as qt
from .encodec.quantization.vq import QuantizedResult


def raise_error_if_nan(tensor,module):
    if torch.any(torch.isnan(tensor)): 
        print(f"Found NaN in {module}")
        raise ValueError()


def spike_loss(x,rep_loss_type,threshold = 0., epsilon = 1e-4):
    #x [Batch,Neuron,Time]
    representation_loss = 0
    if rep_loss_type == 'mean':
        representation_loss = x.mean()
    elif rep_loss_type == 'mean2':
        squared_firing = x.mean(dim = 2)**2 
        representation_loss = squared_firing.mean()
    elif rep_loss_type == 'relu':
        firing_rate = x.mean(dim = (1,2)) #average over time and neuron =  [Batch]
        phi = max(torch.quantile(firing_rate,0.5), threshold) # quantile trick
        relu = torch.maximum(torch.zeros_like(firing_rate), firing_rate - phi) #Cap N \nu at firing rate threshold with the qunatile trick
        
        #max_nu = torch.max(firing_rate)
        #min_nu = torch.min(firing_rate)
        #avg_nu = firing_rate.mean()
        representation_loss = relu.mean()

        #print(f"Loss: {representation_loss}, min,max : [{min_nu},{max_nu}], avg: {avg_nu}")

    elif rep_loss_type == 'relu2':
        firing_rate = x.mean(dim = (1,2)) #average over time and neuron=  [Batch,]
        r = np.random.uniform(0.25,0.5)
        phi = max(torch.quantile(firing_rate,r), threshold) # quantile trick over both batch and neuron
        relu = torch.maximum(torch.zeros_like(firing_rate), firing_rate - phi)**2 #Cap N \nu at firing rate threshold with the qunatile trick
        representation_loss = relu.mean()
    
    elif rep_loss_type == 'trough':
        firing_rate = x.mean(dim = (1,2))

        #Now the trough function:
        trough = (firing_rate-threshold).pow(2)/(threshold*(firing_rate + epsilon))

        #nu_batch = x.mean(dim = (1,2))
        #max_nu = torch.max(nu_batch)
        #min_nu = torch.min(nu_batch)
        #avg_nu = firing_rate

        representation_loss = trough.mean() # Mean over Batch
        #print(f"Loss: {representation_loss}, min,max : [{min_nu},{max_nu}], avg: {avg_nu}")


    return representation_loss

class SpikingEncodecEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder = SEANetEncoder(dimension = args.encodec_dim,ratios = args.encodec_ratios)
        self.downsample_factor = 1
        for r in self.encoder.ratios: self.downsample_factor = self.downsample_factor*r

        self.to_spikes = NaiveSpikeLayer(input_dim = self.encoder.dimension, embed_dim=args.bottleneck_dim,spike_fn=args.spike_function)
        self._batch_norm = args.batch_norm
        if args.batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.encoder.dimension)

        self.rep_loss_type = args.rep_loss_type[0]
        self.firing_rate_threshold = 0
        if self.rep_loss_type in ["relu","relu2","trough"]:
            self.firing_rate_threshold = args.firing_rate_threshold/args.bottleneck_dim
        self.out_channels = args.bottleneck_dim

        self._lstm = args.lstm
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
        
        self._transformer = args.transformer
        if args.transformer:
            self.transformer_internal_dim = args.transformer_internal_dim
            self.transformer = TransformerModel(args.bottleneck_dim, #input to transformer is spikes so ninput = bottleneck dim
                                                args.transformer_output_dim,  # output dimension
                                                batch_first=True, 
                                                d_model=args.transformer_internal_dim, #Internal transformer model dimension
                                                nhead=args.transformer_nhead, #Number of attention heads
                                                nlayers=args.transformer_nlayers, #Number of encoder layers
                                                d_hid=args.transformer_hidden_dim) #Hidden dimension inside feed forward networks of Transformer Encoder
            self.out_channels = args.transformer_output_dim
        else:
            self.transformer = nn.Identity()


        print("Spiking Encodec Encoder")
        print(f"Out Channels: {self.out_channels}")
        print(f"Downsample Factor: {self.downsample_factor}")
        print(f"Encodec Dimension: {self.encoder.dimension}")
        print(f"Nr of Neurons: {args.bottleneck_dim}")
        print(f"Representation Loss Type: {self.rep_loss_type}")
        if self.rep_loss_type in ["relu","relu2","trough"]:
            print(f"Firing rate threshold per neuron: nu = {self.firing_rate_threshold}")
        print(f"Spiking Function: {args.spike_function}")
        print(f"LSTM: {args.lstm}")
        print(f"Transformer: {args.transformer}")
        print(f"Batch Norm: {args.batch_norm}")


        print("\n") 

    def to_spike(self,x):
        if self._batch_norm:
            x = self.batch_norm(x)
        x = torch.permute(x,[0,2,1]) #Permute to [Batch, Time, Neuron]
        x = self.to_spikes(x)
        x = torch.permute(x,[0,2,1])
        return x

    def to_lstm(self,x):
        x = torch.permute(x,[0,2,1])
        x,_ = self.lstm(x)
        x = torch.permute(x,[0,2,1])
        return x

    def to_transformer(self,x):
        x = torch.permute(x,[0,2,1])
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x)
        #print(mask)
        x = self.transformer(x,src_mask = mask)
        x = torch.permute(x,[0,2,1])
        return x
    def forward(self,x ,with_info: bool=False):
        #Encodec Encoder
        x = self.encoder(x)
        raise_error_if_nan(x,"Encodec Encoder")

        #print(f"shape after encoder: {x.shape}")

        #Naive Spiking Module
        x = self.to_spike(x)
        raise_error_if_nan(x,"Naive Spiking Module")
        rep_loss = spike_loss(x,self.rep_loss_type, threshold = self.firing_rate_threshold)
        info = {'spikes': x,'rep_loss':rep_loss} #Store the spikes in info


        if self._lstm:
            x = self.to_lstm(x)
            raise_error_if_nan(x,"LSTM")
            #print(f"Shape after LSTM: {x.shape}")

        if self._transformer:
            x = self.to_transformer(x)
            raise_error_if_nan(x,"Transformer")
            #print(f"Shape after Transformer: {x.shape}")

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
        print(f"Out Channels: {self.out_channels}")
        print(f"Downsample Factor: {self.downsample_factor}")
        print(f"Frame Rate: {self.frame_rate} frames/s")
        print("\n")




    def forward(self,x,with_info: bool=False):
        #Encodec Encoder:
        x = self.encoder(x)
        #Residual Vector Quantization
        q_result = self.quantizer(x,frame_rate = self.frame_rate) #Frame Rate is completely useless here but has to be passed

        x = q_result.quantized
        info = {"codes": q_result.codes, "rep_loss": q_result.penalty}

        return (x, info) if with_info else x

class ResidualSpikingEncodecEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder = SEANetEncoder(dimension = args.encodec_dim,ratios = args.encodec_ratios)
        self.downsample_factor = 1
        for r in self.encoder.ratios: self.downsample_factor = self.downsample_factor*r

        self.spikes = BinaryQuantizer(args.encodec_dim,args.bottleneck_dim,lr_internal = args.binary_quantizer_lr_internal) #Spiking module is a binary quantizer here


        self.rep_loss_type = args.rep_loss_type[0]
        self.firing_rate_threshold = 0
        if self.rep_loss_type == "relu":
            self.firing_rate_threshold = args.firing_rate_threshold/args.bottleneck_dim
        self.out_channels = args.encodec_dim

        self._lstm = args.lstm
        if args.lstm:
            self.lstm = nn.LSTM(
                    batch_first=True,
                    input_size=args.encodec_dim,
                    hidden_size=args.lstm_hidden_size,
                    num_layers=2,
                    bidirectional=False,
                    dropout=0.3
                    )
            self.out_channels = args.lstm_hidden_size
        else:
            self.lstm = nn.Identity()

        self._transformer = args.transformer
        if args.transformer:
            self.transformer_internal_dim = args.transformer_internal_dim
            self.transformer = TransformerModel(args.encodec_dim, #input to transformer is inversed spikes so ninput = encodec dim
                                                args.transformer_output_dim,  # output dimension
                                                batch_first=True, 
                                                d_model=args.transformer_internal_dim, #Internal transformer model dimension
                                                nhead=args.transformer_nhead, #Number of attention heads
                                                nlayers=args.transformer_nlayers, #Number of encoder layers
                                                d_hid=args.transformer_hidden_dim) #Hidden dimension inside feed forward networks of Transformer Encoder
            self.out_channels = args.transformer_output_dim
        else:
            self.transformer = nn.Identity()


        print("Binary Quantizer Spiking Encodec Encoder")
        print(f"Out Channels: {self.out_channels}")
        print(f"Downsample Factor: {self.downsample_factor}")
        print(f"Encodec Dimension: {self.encoder.dimension}")
        print(f"Nr of Neurons: {args.bottleneck_dim}")
        print(f"Representation Loss Type: {self.rep_loss_type}")
        if self.rep_loss_type == "relu":
            print(f"Firing rate threshold per neuron: nu = {self.firing_rate_threshold}")
        print(f"LSTM: {self._lstm}")
        print(f"Transformer: {self._transformer}")

        print("\n") 

    def to_spikes(self,x):
        x = torch.permute(x,[0,2,1])
        x, z_binary, z = self.spikes(x)
        z = torch.view(z_binary.shape)
        x = torch.permute(x,[0,2,1])
        z_binary = torch.permute(z_binary,[0,2,1])
        z = torch.permute(z,[0,2,1])

        return x, z_binary, z

    def to_lstm(self,x):
        x = torch.permute(x,[0,2,1])
        x,_ = self.lstm(x)
        x = torch.permute(x,[0,2,1])
        return x

    def to_transformer(self,x):
        x = torch.permute(x,[0,2,1])
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x)
        #print(mask)
        x = self.transformer(x,src_mask = mask)
        x = torch.permute(x,[0,2,1])
        return x

    def forward(self,x ,with_info: bool=False):
        #Encodec Encoder
        x = self.encoder(x)
        raise_error_if_nan(x, "Encodec Encoder")
        #print(f"shape after encoder: {x.shape}")

        #Binary Quantizer Spiking Module
        x, z_binary, z = self.to_spikes(x)
        raise_error_if_nan(x, "Binary Quantizer")
        rep_loss = spike_loss(z,self.rep_loss_type, threshold = self.firing_rate_threshold)
        info = {'spikes': z_binary,'rep_loss':rep_loss} #Store the spikes and the loss in info
        #print(f"shape after spikes: {x.shape}")

        if self._lstm:
            x = self.to_lstm(x)
            raise_error_if_nan(x, "LSTM")

            #print(f"Shape after LSTM: {x.shape}")

        if self._transformer:
            x = self.to_transformer(x)
            raise_error_if_nan(x, "Transformer")
            #print(f"Shape after Transformer: {x.shape}")

        return (x, info) if with_info else x