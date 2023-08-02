import numpy as np
import math
import time
import os
import torch.nn as nn
from tqdm import tqdm

import torch
import torchaudio as T
import torchaudio.transforms as TT
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio,scale_invariant_signal_distortion_ratio
import matplotlib.pyplot as plt
from dataset_utils import from_maestro
from model import SpikingAudioDiffusion
from learner import _nested_map
from models import compN_bps,compT_bps,clist_bps

from models.multi_scale_mel import MultiMelSpectrogram

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
        win_lengths = [2**i for i in range(5,12)]
        mel_bins = [5*2**i for i in range(7)]
        self.multi_scale_mel = MultiMelSpectrogram(args.sample_rate,win_lengths,mel_bins).to(device)
    #Load the model into memory
    def load_model(self,args,device):
        t0 = time.time()
        
        print(f"loading model from {args.model_dir}")
        if os.path.exists(f'{args.model_dir}/weights.pt'):
            checkpoint = torch.load(f'{args.model_dir}/weights.pt')
        else:
            checkpoint = torch.load(args.model_dir)

        #This code is to help patch different versions of the integrate_context:
        keys = list(checkpoint["model"].keys())
        for key in keys:
            
            prefix = "autoencoder.encoder"
            prefixes = [f"{prefix}.{module}" for module in ["transformer","conformer","lstm"]]
            
            for p in prefixes:
                if key.startswith(p):
                    #print(key)
                    suffix = key.removeprefix(prefix)
                    #print(suffix)
                    newkey = f"{prefix}.integrate_context{suffix}"
                    #print(newkey)
                    # print('\n')
                    checkpoint['model'][newkey] = checkpoint["model"].pop(key)
                    



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
    def encode(self,audio,mu = 0):
        latent,info = self.model.encode(audio,mu = mu)
        return latent,info
    def decode(self,latent,num_steps=10,show_progress=False):
        sample = self.model.decode(latent,num_steps,show_progress)
        return sample
    #Method to autoencode a given input audio
    def autoencode(self,audio,mu =0, num_steps=10,show_progress=False):
        audio_inf,latent,info = self.model.autoencode(audio,mu = mu, num_steps = num_steps,show_progress=show_progress)
        return audio_inf, latent, info
    #Count number of bits per unit time
    def count_bps(self,bottleneck_output): #Returns bits per second
        B,N,T = bottleneck_output.shape
        S = bottleneck_output.sum(dim = (1,2)) 
        N,T = torch.tensor(N,dtype = torch.float,device = self.device),torch.tensor(T,dtype=torch.float,device = self.device)

        clist = clist_bps(N,T,S)/T
        compN = compN_bps(N,T,S)/T
        compT = compT_bps(N,T,S)/T
        #print(clist,compN,compT)

        bpf,_ = torch.min(torch.stack([clist,compN,compT],dim = 0),dim=0)
        bpf = torch.where(bpf < N,bpf,N)
        #print(f"Final Bit rate {bpf} bit/frame")
        return bpf*self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor #convert to bits/s
    #Method to compute wav MSE
    def wavMSE(self,audio_in, audio_out):
        return (audio_in-audio_out).pow(2).mean(dim = (1,2)) 
     
    #Method to compute the multiscale log mel spectrogram MAE
    def multimelspecMAE(self,audio_in, audio_out):
        return self.multi_scale_mel.loss(audio_in,audio_out)
    def multimelspecSISNR(self,audio_in, audio_out):
        return self.multi_scale_mel.sisnr_loss(audio_in,audio_out)
    
    def SISNR(self,audio_in,audio_out):
        if audio_in.shape[1] == 1:
            return scale_invariant_signal_distortion_ratio(audio_out,audio_in).squeeze(1) #Audio is of size [Batch, Channel, Time] so we squeeze the channel dim
        else:
            return scale_invariant_signal_distortion_ratio(audio_out,audio_in)
    def inf_multimelspecMAE(self):
        MAE = torch.empty(0,device = self.device)
        for features in tqdm(self.dataloader):  
            features = _nested_map(features, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)            
            audio_in = features["audio"]
            audio_out, _, _ = self.autoencode(audio_in,num_steps=100)
            MAE = torch.cat((MAE,self.multimelspecMAE(audio_in,audio_out)),dim = 0)
        
        return MAE
    
    def inf_multimelspecSISNR(self):
        MAE = torch.empty(0,device = self.device)
        for features in tqdm(self.dataloader):  
            features = _nested_map(features, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)            
            audio_in = features["audio"]
            audio_out, _, _ = self.autoencode(audio_in,num_steps=100)
            MAE = torch.cat((MAE,self.multimelspecSISNR(audio_in,audio_out)),dim = 0)
        
        return MAE
    
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
        msMAE = torch.empty(0,device = self.device)
        wMSE = torch.empty(0,device = self.device)
        sisnr = torch.empty(0,device = self.device)
        msSISNR = torch.empty(0,device = self.device)
        bps = torch.empty(0,device = self.device)

        for features in tqdm(self.dataloader):  
                    features = _nested_map(features, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)            
                    audio_in = features["audio"]
                    audio_out, _, info = self.autoencode(audio_in,num_steps=100)
                    msMAE = torch.cat((msMAE,self.multimelspecMAE(audio_in,audio_out)),dim = 0)
                    msSISNR = torch.cat((msSISNR,self.multimelspecSISNR(audio_in,audio_out)),dim = 0)
                    wMSE = torch.cat((wMSE,self.wavMSE(audio_in,audio_out)),dim = 0)
                    sisnr = torch.cat((sisnr,self.SISNR(audio_in,audio_out)),dim = 0)
                    if self.model.args.encoder[0] == "encodec":
                        bps = torch.cat((bps,self.count_bps(info["spikes"])),dim = 0)
        
        if self.model.args.encoder[0] == "q_encodec":
            bps = torch.tensor(self.model.autoencoder.encoder.quantizer.n_q * math.ceil(math.log2(self.model.autoencoder.encoder.quantizer.bins))*self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor, device = self.device)
        elif self.model.args.encoder[0] == "mel":
            bps = torch.tensor(self.model.autoencoder.encoder.out_channels*32*self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor, device = self.device)
        return {"multispecMAE": msMAE,
                "multispecSISNR":msSISNR,
                "wavMSE": wMSE,
                "sisnr": sisnr,
                "bps": bps}
    
    

    def parse_midi(self,midi_args):
        i_song,sample_start,sample_end = midi_args.numpy()  
        note_event_list = self.dataloader.dataset.parse_midi(i_song,sample_start,sample_end)
        return note_event_list
    
    def make_note_roll(self,note_events,n_frames):
        note_roll = np.zeros((n_frames, 128))
        onset_roll = np.zeros((n_frames, 128))
        for (note_start,note_end,note,velocity) in note_events:
            #for t in range(note_start,note_end):
            #    note_roll[t,note] = 1
            onset_roll[note_start,note] = 1
        return torch.from_numpy(note_roll),torch.from_numpy(onset_roll)
    

    def midi_batch(self,midi_args,n_frames):
        note_roll_batch = []
        onset_roll_batch = []
        for m in midi_args:
            events = self.parse_midi(m)
            notes, onsets = self.make_note_roll(events,n_frames)
            note_roll_batch.append(notes)
            onset_roll_batch.append(onsets)

        note_roll_batch = torch.stack(note_roll_batch,dim = 0).to(self.device)
        onset_roll_batch = torch.stack(onset_roll_batch,dim = 0).to(self.device)
        return note_roll_batch,onset_roll_batch
    
    def correlate(self,spikes,midi):
        #Ensure spikes and midi are of shape [Batch,Time,Neuron/Note]
        assert spikes.shape[0] == midi.shape[0]
        assert spikes.shape[1] == midi.shape[1]
        
        N_neurons = spikes.shape[-1]
        N_notes = midi.shape[-1]

        N_batch = spikes.shape[0]
        N_frames = spikes.shape[-2]
        
        lags = range(int(-N_frames/2),int(N_frames/2+1),1)
        N_lags_oneside = int((len(lags)-1)/2)

        corr = torch.empty((N_neurons,N_notes,len(lags))).to(self.device)

        #Correlation loop:
        for lag in lags:
            if lag>=0:
                spikes_crop = spikes[:,0:N_frames-lag,:]
                midi_crop = midi[:,lag:,:]

            else:
                spikes_crop = spikes[:,-lag:,:] 
                midi_crop = midi[:,0:N_frames+lag,:]

            
            corr[:,:,N_lags_oneside+lag] = torch.einsum("btz,btn->zn",spikes_crop.to(torch.double),midi_crop.to(torch.double))/N_batch

        return corr

    @torch.no_grad()
    def midi_correlate(self):
        count = 0
        nD = len(self.dataloader)
        corr = torch.empty(0,device = self.device) #Initialize outside the loop 

        for features in tqdm(self.dataloader):  
            audio_in = features["audio"].to(self.device)
            _,info = self.encode(audio_in)
            spikes = torch.permute(info['spikes'],[0,2,1])
            #print(f"Spikes shape: {spikes.shape}")
            midi_args = features["midi_args"]
            _,onsets = self.midi_batch(midi_args,n_frames = spikes.shape[1])
            #print(f"notes shape: {notes.shape}, onsets shape: {onsets.shape}")
            #t = time.time()
            #Loop to calculate average inline
            if count == 0:
                corr = self.correlate(spikes,onsets)/nD
            else:
                corr += self.correlate(spikes,onsets)/nD #In place addition like this to prevent memory leakage
            

            #dt = time.time() - t
            #print(f"time to correlate: {dt}   corr shape: {corr.shape}, corr device: {corr.device}")
            count = count + 1
        
        
        return(corr)





            







