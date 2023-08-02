import numpy as np
import math
import time
import os
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchaudio as T
import torchaudio.transforms as TT
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio,scale_invariant_signal_distortion_ratio

class MelSpectrogram(nn.Module):
    def __init__(self,sampling_rate,win_length,n_mels):
        super(MelSpectrogram, self).__init__()
        self.mel_args = {
          'sample_rate': sampling_rate,
          'win_length': win_length,
          'hop_length': win_length//4,
          'n_fft': 2048,
          'n_mels': n_mels,
          'power': 1.0,# magnitude of stft
          'normalized': True,
        } 
        self.mel_spec_transform = TT.MelSpectrogram(**(self.mel_args))
        self.B = 1.

    def forward(self,x):
        spec = self.mel_spec_transform(x)
        #print(f"specmin = {spec.min()}, specmax = {spec.max()}, specmean = {spec.mean()}")
        x = torch.log10(1 + self.B*spec)
        return x 
    
class MultiMelSpectrogram(nn.Module):
    def __init__(self,sampling_rate,win_lengths,mel_bins):
        super(MultiMelSpectrogram, self).__init__()    
        self.mel_transforms = nn.ModuleList([MelSpectrogram(sampling_rate,w,n) for (w,n) in zip(win_lengths,mel_bins)])
        self.L1loss = nn.L1Loss()
    def forward(self,x):
        specs = []
        if x.shape[0] == 1:
            x = x.unsqueeze(1)
        for mel in self.mel_transforms:
            specs.append(mel(x))
        return specs
    
    def losses(self,xin,xout):
        losses = []
        if xin.shape[0] == 1:
            xin = xin.unsqueeze(1)
            #print(f"{xin.shape} {xout.shape}")
        if xout.shape[0] == 1:
            xout = xout.unsqueeze(1)
            #print(f"{xin.shape} {xout.shape}")

        for mel in self.mel_transforms:
            #print(f"{torch.abs(mel(xout)-mel(xin)).shape}")
            mel_out = mel(xout)
            mel_in = mel(xin)
            loss = torch.abs(mel_out-mel_in).mean(dim = (1,2,3))
            losses.append(loss)
        return torch.stack(losses,dim = 0).permute([1,0])
    def sisnr_losses(self,xin,xout):
        losses = []
        if xin.shape[0] == 1:
            xin = xin.unsqueeze(1)
            #print(f"{xin.shape} {xout.shape}")
        if xout.shape[0] == 1:
            xout = xout.unsqueeze(1)
            #print(f"{xin.shape} {xout.shape}")

        for mel in self.mel_transforms:
            #print(f"{torch.abs(mel(xout)-mel(xin)).shape}")
            mel_out = mel(xout)
            mel_in = mel(xin)
            loss = scale_invariant_signal_distortion_ratio(mel_out,mel_in).mean(dim = (1,2))
            losses.append(loss)
        return torch.stack(losses,dim = 0).permute([1,0])
    
    def loss(self,xin,xout):
        return self.losses(xin,xout).mean(dim = 1)
    
    def sisnr_loss(self,xin,xout):
        return self.sisnr_losses(xin,xout).mean(dim = 1)
    

def read_dir(dir):
    file_list = os.listdir(dir)
    file_list = [f for f in file_list if f[-4:] == ".wav"]
    file_list = [f for f in file_list if f[-9:] != "48000.wav"]
    
    audio_names = [f[:-4] for f in file_list]
    audios = [T.load(dir+f)[0] for f in file_list]
    audios = torch.stack(audios,dim = 0).to("cuda")
    audios.unsqueeze(1)
    sr = T.load(dir+file_list[0])[1] #return sampling rate here
    return audios, audio_names, sr
   

if __name__ == "__main__":
    win_lengths = [2**i for i in range(5,12)]
    mel_bins = [5*2**i for i in range(7)]

    multi_scale_mel = MultiMelSpectrogram(22050,win_lengths,mel_bins).to('cuda')

    
    audio_in_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/auto/audio_8.wav"
    audio_opus_path = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/opus6/audio_8_opus6.wav"
    audio_sparse_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/sparse_conv/audio_8_sparse_conv.wav"
    audio_free_path = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/FREE/FREE_audio_8.wav"
    audio_rvq_path = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/RVQ/RVQ_audio_8.wav"


    audio_in,sr = T.load(audio_in_path)
    audio_free,sr = T.load(audio_free_path)
    audio_sparse,sr = T.load(audio_sparse_path)
    audio_opus,sr = T.load(audio_opus_path)

    audio_in = audio_in.to('cuda')
    audio_free = audio_free.to('cuda')
    audio_sparse = audio_sparse.to('cuda')
    audio_opus = audio_opus.to('cuda')


    specs = multi_scale_mel(audio_in)
    fig,ax = plt.subplots(len(specs),figsize = (30,21))
    for i,spec in enumerate(specs):
        print(f"Spec {i} w = {win_lengths[i]}, n_mels = {mel_bins[i]}: max = {torch.max(spec[0])}, min = {torch.min(spec[0])}, shape = {spec.shape} ")
        ax[i].imshow(spec[0,0].cpu().numpy(),interpolation="none",aspect='auto')    
    plt.savefig('test_multi_scale_spec_input.png')
    loss = multi_scale_mel.loss(audio_in,audio_in)
    print(f"Loss Original: {loss} (Sanity Check) ")


    specs = multi_scale_mel(audio_free)
    for i,spec in enumerate(specs):
        #print(f"Spec {i} w = {win_lengths[i]}, n_mels = {mel_bins[i]}: max = {torch.max(spec[0])}, min = {torch.min(spec[0])} ")
        ax[i].imshow(spec[0,0].cpu().numpy(),interpolation="none",aspect='auto')    
    plt.savefig('test_multi_scale_spec_free.png')
    losses = multi_scale_mel.losses(audio_in,audio_free)
    print(f"Losses FREE: {losses}")
    loss = multi_scale_mel.loss(audio_in,audio_free)
    print(f"Loss FREE: {loss}")


    specs = multi_scale_mel(audio_sparse)
    for i,spec in enumerate(specs):
        #print(f"Spec {i} w = {win_lengths[i]}, n_mels = {mel_bins[i]}: max = {torch.max(spec[0])}, min = {torch.min(spec[0])} ")
        ax[i].imshow(spec[0,0].cpu().numpy(),interpolation="none",aspect='auto')    
    plt.savefig('test_multi_scale_spec_sparse.png')
    losses = multi_scale_mel.losses(audio_in,audio_sparse)
    print(f"Losses SPARSE: {losses}")
    loss = multi_scale_mel.loss(audio_in,audio_sparse)
    print(f"Loss SPARSE: {loss}")

    specs = multi_scale_mel(audio_opus)
    for i,spec in enumerate(specs):
        #print(f"Spec {i} w = {win_lengths[i]}, n_mels = {mel_bins[i]}: max = {torch.max(spec[0])}, min = {torch.min(spec[0])} ")
        ax[i].imshow(spec[0,0].cpu().numpy(),interpolation="none",aspect='auto')    
    plt.savefig('test_multi_scale_spec_opus6.png')
    losses = multi_scale_mel.losses(audio_in,audio_opus)
    print(f"Losses OPUS: {losses}")
    loss = multi_scale_mel.loss(audio_in,audio_opus)
    print(f"Loss OPUS: {loss}")

   
    basedir = '/lcncluster/lisboa/spikes_audio_diffusion/experiment'
    audios_original,_,sr = read_dir(f"{basedir}/originals/") 
    audios_opus6,_,sr = read_dir(f"{basedir}/opus6/")
    audios_opus12,_,sr = read_dir(f"{basedir}/opus12/") 
    audios_free,_,sr = read_dir(f"{basedir}/FREE/") 
    audios_rvq,_,sr = read_dir(f"{basedir}/RVQ/") 
    audios_sparse,_,sr = read_dir(f"{basedir}/SPARSE/") 



    loss = multi_scale_mel.loss(audios_original,audios_free) 
    print(f"FREE  Loss: {loss}")
    
    loss = multi_scale_mel.loss(audios_original,audios_opus6) 
    print(f"Opus6  Loss: {loss}")
    
    loss = multi_scale_mel.loss(audios_original,audios_opus12) 
    print(f"Opus12  Loss: {loss}")
    
    loss = multi_scale_mel.loss(audios_original,audios_rvq) 
    print(f"RVQ  Loss: {loss}")
    
    loss = multi_scale_mel.loss(audios_original,audios_sparse) 
    print(f"SPARSE  Loss: {loss}")