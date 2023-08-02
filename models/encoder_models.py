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
from collections import OrderedDict


from .transformer import TransformerModel, generate_square_subsequent_mask
from .conformer import ConformerModel
from .binary_quantizer import BinaryQuantizer
from .encodec.modules import SEANetEncoder
from .encodec import quantization as qt
from .encodec.quantization.vq import QuantizedResult
from .bitrate_counting import compN_bps,compT_bps,clist_bps

def raise_error_if_nan(tensor,module):
    if torch.any(torch.isnan(tensor)): 
        print(f"Found NaN in {module}")
        raise ValueError()


class MLP(nn.Module):
    def __init__(self,input_dim,
                 output_dim,
                 n_hidden_layers,
                 hidden_dim):
        super().__init__()
        
        self.model = nn.Sequential()
        self.model.add_module("input", nn.Linear(input_dim,hidden_dim))
        self.model.add_module("act_in", nn.ReLU())
        for k in range(n_hidden_layers):
            self.model.add_module(f"hlayer{k+1}", nn.Linear(hidden_dim,hidden_dim))
            self.model.add_module(f"act{k+1}", nn.ReLU())
        self.model.add_module("output",nn.Linear(hidden_dim,output_dim))
        self.model.add_module("act_output",nn.ReLU())
    
    def forward(self,x):
        return self.model(x)

        





class IntegrateContext(nn.Module):
    def __init__(self,args):
        super().__init__()
        self._lstm = args.lstm
        self._transformer = args.transformer
        self._conformer = args.conformer
        self.conv =  nn.Conv1d(in_channels = args.bottleneck_dim,
                               out_channels = args.bottleneck_dim,
                               groups = args.bottleneck_dim,
                               kernel_size = args.transformer_conv_kernel,
                               padding = (args.transformer_conv_kernel -1)//2)
        self.out_channels=args.bottleneck_dim

        #Case LSTM
        if self._lstm:
            self._transformer = False
            self._conformer = False
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
        

        #Case Conformer
        if self._conformer:
            self._transformer = False
            self._lstm = False
            self.conformer = ConformerModel(args.bottleneck_dim, #Input dimension
                                            args.conformer_output_dim, #Output Dimension
                                            d_model = args.conformer_d_model,
                                            ffn_dim = args.conformer_ffn_dim,
                                            depthwise_conv_kernel_size = args.conformer_depthwise_conv_kernel_size,
                                            nhead = args.conformer_nhead,
                                            nlayers =args.conformer_nlayers,
                                            dropout = args.conformer_dropout,
                                            use_group_norm  = args.conformer_use_group_norm,
                                            convolution_first = args.conformer_convolution_first)

            self.out_channels = args.conformer_output_dim
        else:
            self.conformer = nn.Identity()
        

        #Case Transformer
        if self._transformer:
            self._lstm = False
            self._conformer = False

            
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


    def to_lstm(self,x):
        x = torch.permute(x,[0,2,1])#Permute to [Batch, Time, Neuron]
        x,_ = self.lstm(x)
        x = torch.permute(x,[0,2,1])#Permute to [Batch, Neuron, Time]
        return x

    def to_transformer(self,x):
        x = torch.permute(x,[0,2,1]) #Permute to [Batch, Time, Neuron]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x)
        #print(mask)
        x = self.transformer(x,src_mask = mask)
        x = torch.permute(x,[0,2,1]) #Permute to [Batch, Neuron, Time]
        return x
    
    def to_conformer(self,x):
        x = torch.permute(x,[0,2,1])#Permute to [Batch, Time, Neuron]
        x = self.conformer(x)
        x = torch.permute(x,[0,2,1]) #Permute to [Batch, Neuron, Time]
        return x
    
    def forward(self,x):
 

        #First a convolution
        x = self.conv(x)

        if self._lstm:
            x = self.to_lstm(x)
            raise_error_if_nan(x,"LSTM")
            #print(f"Shape after LSTM: {x.shape}")

        elif self._conformer:
            x = self.to_conformer(x)
            raise_error_if_nan(x,"Conformer")
            #print(f"Shape after Conformer: {x.shape}")
        elif self._transformer:
            x = self.to_transformer(x)
            raise_error_if_nan(x,"Transformer")
            #print(f"Shape after Transformer: {x.shape}")
        else:
            return x


        return x
    


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
        self.firing_rate_threshold = args.firing_rate_threshold
        if self.rep_loss_type in ["relu","relu2","trough","frying_pan"]:
            self.firing_rate_threshold /= args.bottleneck_dim
    
        self.B0 = args.B0 # threshold bits per frame used only for brute loss
        
        self.integrate_context = IntegrateContext(args)

        self.out_channels = self.integrate_context.out_channels
        print("Spiking Encodec Encoder")
        print(f"Out Channels: {self.out_channels}")
        print(f"Downsample Factor: {self.downsample_factor}")
        print(f"Encodec Dimension: {self.encoder.dimension}")
        print(f"Nr of Neurons: {args.bottleneck_dim}")
        print(f"Representation Loss Type: {self.rep_loss_type}")
        if self.rep_loss_type in ["relu","relu2","trough"]:
            print(f"Firing rate threshold per neuron: nu = {self.firing_rate_threshold}")
        print(f"Spiking Function: {args.spike_function}")
        print(f"LSTM: {self.integrate_context._lstm}")
        print(f"Transformer: {self.integrate_context._transformer}")
        print(f"Conformer: {self.integrate_context._conformer}")

        print(f"Batch Norm: {args.batch_norm}")

        print("\n") 


    def spike_loss(self,x,rep_loss_type,threshold = 0., epsilon = 1e-4):
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
            representation_loss = trough.mean() # Mean over Batch

        elif rep_loss_type == 'frying_pan':
            firing_rate = x.mean()/threshold

            #Now the loss:
            upper = torch.maximum(torch.zeros_like(firing_rate), firing_rate - 2)
            lower = torch.maximum(torch.zeros_like(firing_rate), 1/16 - firing_rate)

            representation_loss = upper + lower

        elif rep_loss_type == 'brute':
            # threshold argument interpreted as fraction of target bitrate to beat
            # threshold = 0.9 means that batch average bitrate = 0.9B0
            B,N,T = x.shape
            S = x.sum(dim = (1,2)) 
            N,T = torch.tensor(N,dtype = torch.float),torch.tensor(T,dtype=torch.float)
            clist = clist_bps(N,T,S)
            compN = compN_bps(N,T,S)
            compT = compT_bps(N,T,S)
            #naive = N*T*torch.ones_like(clist)
            #print(f"clist {clist}")
            #print(f"compN {compN}")
            #print(f"compT {compT}")
            #print(f"naive {naive}")

            sparse_factor,_ = torch.min(torch.stack([clist,compN,compT],dim = 0),dim=0)
            sparse_factor /= (self.B0*T)
            #print(sparse_factor)
            sparse_factor = sparse_factor.mean()
            #print(sparse_factor)
            upper = torch.maximum(torch.zeros_like(sparse_factor), sparse_factor - threshold)
            #lower = torch.maximum(torch.zeros_like(sparse_factor), sparse_factor)
            representation_loss = upper.mean()
            #print(f"rep loss: {representation_loss} {threshold}")

        return representation_loss


    def to_spike(self,x):
        if self._batch_norm:
            x = self.batch_norm(x)
        x = torch.permute(x,[0,2,1]) #Permute to [Batch, Time, Neuron]
        x = self.to_spikes(x)
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
        rep_loss = self.spike_loss(x,self.rep_loss_type, threshold = self.firing_rate_threshold)
        info = {'spikes': x,'rep_loss':rep_loss} #Store the spikes in info

        #Integrate the context with either LSTM, Transformer or Conformer
        x = self.integrate_context(x)

        return (x, info) if with_info else x

class QuantizingEncodecEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder = SEANetEncoder(dimension = args.encodec_dim,ratios = args.encodec_ratios)
        self.quantizer = qt.ResidualVectorQuantizer(dimension = args.bottleneck_dim,
                                                    n_q = args.n_q,
                                                    bins = args.q_bins,)

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

class MuSpikingEncodecEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.encoder = SEANetEncoder(dimension = args.encodec_dim,ratios = args.encodec_ratios)
        self.downsample_factor = 1
        for r in self.encoder.ratios: self.downsample_factor = self.downsample_factor*r

        self.spiking_layer = NaiveSpikeLayer(input_dim = self.encoder.dimension, embed_dim=args.bottleneck_dim,spike_fn=args.spike_function)
        self._batch_norm = args.batch_norm
        if args.batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.encoder.dimension)

    
        self.B0 = args.B0 # threshold bits per frame used only for brute loss
        
        self.mu_embed_pre  = nn.Embedding(args.nr_mu_embeddings, 128)
        self.mu_embed_post = nn.Embedding(args.nr_mu_embeddings, 128)


        self.integrate_context_pre = IntegrateContext(args)
        self.integrate_context_post = IntegrateContext(args)

        self.out_channels = self.integrate_context_post.out_channels
        print("Mu Spiking Encodec Encoder")
        print(f"Out Channels: {self.out_channels}")
        print(f"Downsample Factor: {self.downsample_factor}")
        print(f"Encodec Dimension: {self.encoder.dimension}")
        print(f"Nr of Neurons: {args.bottleneck_dim}")
        print(f"Spiking Function: {args.spike_function}")
        print(f"B0: {self.B0}")
        print(f"LSTM: Pre: {self.integrate_context_pre._lstm}, Post: {self.integrate_context_post._lstm}")
        print(f"Transformer: Pre: {self.integrate_context_pre._transformer}, Post: {self.integrate_context_post._transformer}")
        print(f"Conformer: Pre: {self.integrate_context_pre._conformer}, Post: {self.integrate_context_post._conformer}")

        print(f"Batch Norm: {args.batch_norm}")

        print("\n") 



    def spike_loss(self,x,mu):
        # mu argument interpreted as fraction of target bitrate to beat
        # mu = 0.9 means that batch average bitrate = 0.9B0. 
        # B0 is sent as argument
        B,N,T = x.shape
        S = x.sum(dim = (1,2)) 
        N,T = torch.tensor(N,dtype = torch.float),torch.tensor(T,dtype=torch.float)
        clist = clist_bps(N,T,S)
        compN = compN_bps(N,T,S)
        compT = compT_bps(N,T,S)
        #naive = N*T*torch.ones_like(clist)
        #print(f"clist {clist}")
        #print(f"compN {compN}")
        #print(f"compT {compT}")
        #print(f"naive {naive}")
        #sparse_factor,_ = torch.min(torch.stack([clist,compN,compT],dim = 0),dim=0)
        #print(sparse_factor)

        event_rate = S/(self.B0*T)
        Bmu = 2**(-mu/(self.mu_embed_post.num_embeddings//10))
        #print(f"Bmu: {Bmu}")
        #print(f"S: {event_rate}")

        upper = torch.maximum(torch.zeros_like(event_rate), event_rate - Bmu)
        #lower = torch.maximum(torch.zeros_like(sparse_factor), sparse_factor)
        representation_loss = upper.mean()
        #print(f"rep loss: {representation_loss}")

        return representation_loss
    def to_spike(self,x):
        if self._batch_norm:
            x = self.batch_norm(x)
        x = torch.permute(x,[0,2,1]) #Permute to [Batch, Time, Neuron]
        x = self.spiking_layer(x)
        x = torch.permute(x,[0,2,1])
        return x
    def forward(self,x,mu = 0, with_info: bool = False):
  
        assert x.shape[0] == mu.shape[0] , "MuEncodec x, and mu should have the same Batch dimension"
        #Encodec Encoder
        x = self.encoder(x)
        raise_error_if_nan(x,"Encodec Encoder")
        #print(f"shape after encoder: {x.shape}")

        
        #MLP for the sparsity level
        mu_bias_pre = self.mu_embed_pre(mu.int())
        #print(f"After pre bias x shape: {x.shape} mu bias shape {mu_bias_pre.shape}")
        
        #Integrate the context with either LSTM, Transformer or Conformer
        x = self.integrate_context_pre(x+mu_bias_pre[:,:,None])
        #print(f" shape afer pre context: {x.shape}")

        #Naive Spiking Module
        x = self.to_spike(x)
        raise_error_if_nan(x,"Naive Spiking Module")
        rep_loss = self.spike_loss(x,mu)
        info = {'spikes': x,'rep_loss':rep_loss} #Store the spikes in info


        mu_bias_post = self.mu_embed_post(mu.int())
        #print(f"After post bias x shape: {x.shape} mu bias shape {mu_bias_post.shape}")

        #Integrate the context with either LSTM, Transformer or Conformer
        x = self.integrate_context_post(x+mu_bias_post[:,:,None])
        #print(f" shape afer post context: {x.shape}")
        return (x, info) if with_info else x




class RecursiveSpikingEncodecEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.encoder = SEANetEncoder(dimension = args.encodec_dim,ratios = args.encodec_ratios)
        self.context_encoder = SEANetEncoder(dimension = args.encodec_dim,ratios = args.encodec_ratios)

        self.downsample_factor = 1
        for r in self.encoder.ratios: self.downsample_factor = self.downsample_factor*r

        self.spiking_layer = NaiveSpikeLayer(input_dim = self.encoder.dimension, embed_dim=args.bottleneck_dim,spike_fn=args.spike_function)
        self._batch_norm = args.batch_norm
        if args.batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.encoder.dimension)

        self.rep_loss_type = args.rep_loss_type[0]
        self.B0 = args.B0 # Base number of bits per frame to beat
        self.firing_rate_threshold = args.firing_rate_threshold # fraction of B0 to beat
        # if for example B0 = 80 and firing_rate_threshold = 0.75 the storage cost to beat is 60*T
        
        self.integrate_context = IntegrateContext(args)
        self.out_channels = self.integrate_context.out_channels

        d_in = self.context_encoder.dim
        self.pre_mlp = MLP(input_dim = d_in,output_dim=d_in, hidden_dim=128, n_hidden_layers=2)
        self.post_mlp = MLP(input_dim = d_in,output_dim=self.out_channels, hidden_dim=128, n_hidden_layers=2)

        self.train_sequence_length= args.sequence_length

        print("Recursive Spiking Encodec Encoder")
        print(f"Out Channels: {self.out_channels}")
        print(f"Downsample Factor: {self.downsample_factor}")
        print(f"Encodec Dimension: {self.encoder.dimension}")
        print(f"Nr of Neurons: {args.bottleneck_dim}")
        print(f"Representation Loss Type: {self.rep_loss_type}")
        print(f"B0: {self.B0}")
        print(f"threshold: {self.firing_rate_threshold}")
        print(f"Spiking Function: {args.spike_function}")
        print(f"LSTM: {self.integrate_context._lstm}")
        print(f"Transformer: {self.integrate_context._transformer}")
        print(f"Transformer Conv1D: {self.integrate_context._transformer_conv}")
        print(f"Conformer: {self.integrate_context._conformer}")

        print(f"Batch Norm: {args.batch_norm}")

        print("\n") 



    def spike_loss(self,x):
        if self.rep_loss_type == 'brute':
            B,N,T = x.shape
            S = x.sum(dim = (1,2)) 
            N,T = torch.tensor(N,dtype = torch.float),torch.tensor(T,dtype=torch.float)
            clist = clist_bps(N,T,S)
            compN = compN_bps(N,T,S)
            compT = compT_bps(N,T,S)
            #naive = N*T*torch.ones_like(clist)
            #print(f"clist {clist}")
            #print(f"compN {compN}")
            #print(f"compT {compT}")
            #print(f"naive {naive}")

            sparse_factor,_ = torch.min(torch.stack([clist,compN,compT],dim = 0),dim=0)
            sparse_factor /= (self.B0*T)
            #print(sparse_factor)
            upper = torch.maximum(torch.zeros_like(sparse_factor), sparse_factor - self.firing_rate_threshold)
            #lower = torch.maximum(torch.zeros_like(sparse_factor), sparse_factor)
            representation_loss = upper.mean()
            #print(f"rep loss: {representation_loss}")
        else:
            representation_loss = torch.Tensor([0.],device = x.device)

        return representation_loss
    def to_spike(self,x):
        if self._batch_norm:
            x = self.batch_norm(x)
        x = torch.permute(x,[0,2,1]) #Permute to [Batch, Time, Neuron]
        x = self.spiking_layer(x)
        x = torch.permute(x,[0,2,1])
        return x
    def forward(self, x, xpre, with_info: bool = False):
  
        #Encodec Encoder
        x = self.encoder(x)
        raise_error_if_nan(x,"Encodec Encoder")
        #print(f"shape after encoder: {x.shape}")

        #Identical module for the context part
        xpre = self.context_encoder(xpre)
        raise_error_if_nan(xpre,"Context Encodec Encoder")
        #print(f"shape after context encoder: {xpre.shape}")

        #Collapse the time dimension:
        xpre = xpre.mean(dim = -1)
        raise_error_if_nan(xpre,"Collapse Time")
        #print(f"shape after colapsing time: {xpre.shape}")

        #MLPs
        xpre_bias_spikes  = self.pre_mlp(xpre)
        raise_error_if_nan(xpre_bias_spikes,"Bias Pre Spikes")
        #print(f"shape after pre mlp: {xpre_bias_spikes.shape}")

        xpre_bias_decoder = self.post_mlp(xpre)
        #Permutation required before the post integrate_context linear projection
  

        raise_error_if_nan(xpre_bias_decoder,"Bias Decoder")
        #print(f"shape after post mlp: {xpre_bias_decoder.shape}")

        #Inject pre bias with the random mask

        if self.training:
            p = torch.rand(x.shape[0],device = x.device)
            #The model has to learn how to encode without context!
            xpre = torch.where(p<0.1,torch.zeros_like(xpre), xpre).to(x.device)
            #print(mask)
        
        #print(f"shape after injecting and masking: {x.shape}")
       

        #Naive Spiking Module
        x = self.to_spike(x)
        raise_error_if_nan(x,"Naive Spiking Module")
        rep_loss = self.spike_loss(x)
        info = {'spikes': x,'rep_loss':rep_loss} #Store the spikes in info

        #Integrate the context with either LSTM, Transformer or Conformer
        x = self.integrate_context(x)


        #print(f"output shape: {x.shape}")

        return (x, info) if with_info else x



