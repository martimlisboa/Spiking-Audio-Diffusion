import numpy as np
import math
import time
import os
import torch.nn as nn
from tqdm import tqdm

import torch
import torchaudio as T
import torchaudio.transforms as TT
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
import matplotlib.pyplot as plt
from dataset_utils import from_maestro
from model import SpikingAudioDiffusion
from learner import _nested_map

#Computes the Mel Spectrogram input oudio audio. Parameters for the 
class MelSpectrogram(nn.Module):
    def __init__(self,args):
        super(MelSpectrogram, self).__init__()
        self.mel_args = {
          'sample_rate': args.sample_rate,
          'win_length': 1024,
          'hop_length': 256,
          'n_fft': 2048,
          'n_mels': 128,
          'power': 1.0,# magnitude of stft
          'normalized': True,
        } 
        self.mel_spec_transform = TT.MelSpectrogram(**(self.mel_args))
    def forward(self,x):
        x = self.mel_spec_transform(x)
        return x 


class Inferer:
    def __init__(self,args,device):
        super(Inferer,self).__init__()
        self.device = device
        self.dataloader = self.make_dataloader(args)
        self.model = self.load_model(args,device)

        self.mel_spec = MelSpectrogram(args).to(self.device)
    #Load the model into memory
    def load_model(self,args,device):
        t0 = time.time()
        
        print(f"loading model from {args.model_dir}")
        if os.path.exists(f'{args.model_dir}/weights.pt'):
            checkpoint = torch.load(f'{args.model_dir}/weights.pt')
        else:
            checkpoint = torch.load(args.model_dir)
        
        model = SpikingAudioDiffusion(args).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        dt = time.time()-t0
        print(f"Loading Time: {dt}")
        return model

    # Make the Dataloader with the given arguments
    def make_dataloader(self,args):
        if args.data_dirs[0] == 'maestro':
            dataloader = from_maestro(args)
        else:
            raise ValueError('NO DATASET.')
        return dataloader
    #Overide encode and decode methods:
    def encode(self,audio):
        latent,info = self.model.encode(audio)
        return latent,info
    def decode(self,latent,num_steps=10,show_progress=False):
        sample = self.model.decode(latent,num_steps,show_progress)
        return sample
    #Method to autoencode a given input audio
    def autoencode(self,audio,num_steps=10,show_progress=False):
        latent,info = self.encode(audio)
        audio_inf = self.decode(latent,num_steps=num_steps,show_progress=show_progress)
        return audio_inf, latent, info
    #Count number of bits per unit time
    def count_bps(self,bottleneck_output): #Returns bits per second
        nr_spikes  = bottleneck_output.sum(dim = (1,2))
        time_frames = bottleneck_output.shape[-1]
        nr_neurons = bottleneck_output.shape[-2]
        nr_time_bits = math.ceil(math.log2(time_frames))
        nr_neuron_bits = math.ceil(math.log2(nr_neurons))
        bpf = nr_spikes *(nr_time_bits + nr_neuron_bits)/time_frames

        bpf = torch.where(bpf < nr_neurons,bpf,nr_neurons)
        #print(f"Final Bit rate {bpf} bit/frame")

        return bpf*self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor #convert to bits/s
    #Method to compute wav MSE
    def wavMSE(self,audio_in, audio_out):
        return (audio_in-audio_out).pow(2).mean(dim = (1,2))  
    #Method to compute a spectrogram MSE
    def specMSE(self,audio_in, audio_out):
        spec_in = self.mel_spec(audio_in)
        spec_out = self.mel_spec(audio_out)
        return ((spec_in - spec_out)).pow(2).mean(dim=(1,2,3))
    def SISNR(self,audio_in,audio_out):
        if audio_in.shape[1] == 1:
            return scale_invariant_signal_noise_ratio(audio_out,audio_in).squeeze(1) #Audio is of size [Batch, Channel, Time] so we squeeze the channel dim
        else:
            return scale_invariant_signal_noise_ratio(audio_out,audio_in)
    def inf_specMSE(self):
        MSE = torch.empty(0,device = self.device)
        for features in tqdm(self.dataloader):  
            features = _nested_map(features, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)            
            audio_in = features["audio"]
            audio_out, _, _ = self.autoencode(audio_in,num_steps=100)
            MSE = torch.cat((MSE,self.specMSE(audio_in,audio_out)),dim = 0)
        
        return MSE
    def inf_SISNR(self):
        SISNR = torch.empty(0,device = self.device)   
        for features in tqdm(self.dataloader):  
            features = _nested_map(features, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)            
            audio_in = features["audio"]
            audio_out, _, _ = self.autoencode(audio_in,num_steps=100)
            SISNR = torch.cat((SISNR,self.SISNR(audio_in,audio_out)),dim = 0)
        
        return SISNR
    
    @torch.no_grad()
    def infer(self): #method to infer a dictionary containing all possible metrics
        sMSE = torch.empty(0,device = self.device)
        wMSE = torch.empty(0,device = self.device)
        sisnr = torch.empty(0,device = self.device)
        bps = torch.empty(0,device = self.device)

        for features in tqdm(self.dataloader):  
                    features = _nested_map(features, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)            
                    audio_in = features["audio"]
                    audio_out, _, info = self.autoencode(audio_in,num_steps=100)
                    sMSE = torch.cat((sMSE,self.specMSE(audio_in,audio_out)),dim = 0)
                    wMSE = torch.cat((wMSE,self.wavMSE(audio_in,audio_out)),dim = 0)
                    sisnr = torch.cat((sisnr,self.SISNR(audio_in,audio_out)),dim = 0)
                    if self.model.args.encoder[0] == "encodec":
                        bps = torch.cat((bps,self.count_bps(info["spikes"])),dim = 0)
        
        if self.model.args.encoder[0] == "q_encodec":
            bps = torch.tensor(self.model.autoencoder.encoder.quantizer.n_q * math.ceil(math.log2(self.model.autoencoder.encoder.quantizer.bins))*self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor, device = self.device)
        elif self.model.args.encoder[0] == "mel":
            bps = torch.tensor(self.model.autoencoder.encoder.out_channels*32*self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor, device = self.device)
        return {"specMSE": sMSE,
                "wavMSE": wMSE,
                "sisnr": sisnr,
                "bps": bps}

        

            







