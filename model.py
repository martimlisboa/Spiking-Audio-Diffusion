import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler,DiffusionVocoder
from audio_encoders_pytorch import MelE1d, TanhBottleneck
from audio_diffusion_pytorch.utils import exists
from parser import make_parser
from dataset_utils import MaestroDataset,from_maestro

from models import SpikingEncodecEncoder, QuantizingEncodecEncoder,MuSpikingEncodecEncoder,RecursiveSpikingEncodecEncoder


class SpikingAudioDiffusion(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        #Encoder Stuff
        if args.encoder[0] in ["mel", "encodec","q_encodec","mu_encodec","rec_encodec"]:
            self.autoencoder = DiffusionAE(
                encoder= self._build_encoder(),
                inject_depth=args.inject_depth, # Depth at which to inject the Auto Encoded Context: In the Paper: 4
                net_t=UNetV0, # The model type used for diffusion upsampling
                in_channels=1, # U-Net: number of input/output (audio) channels
                channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer  In the Paper: [256,512,512,512,1024,1024,1024]
                factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer : In the Paper [1,2,2,2,2,2,2]
                items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer : In the Paper [1,2,2,2,2,2,2]
                diffusion_t=VDiffusion, # The diffusion method used
                sampler_t=VSampler, # The diffusion sampler used
            )
        elif args.encoder[0] in ["vocoder"]:
            self.autoencoder = DiffusionVocoder(
                mel_n_fft=1024, # Mel-spectrogram n_fft
                mel_channels=80, # Mel-spectrogram channels
                mel_sample_rate=args.sample_rate, # Mel-spectrogram sample rate
                mel_normalize_log=True, # Mel-spectrogram log normalization (alternative is mel_normalize=True for [-1,1] power normalization)
                net_t=UNetV0, # The model type used for diffusion vocoding
                channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
                factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
                items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
                diffusion_t=VDiffusion, # The diffusion method used
                sampler_t=VSampler, # The diffusion sampler used
            )
    def get_device(self):
        p = next(self.parameters())
        return p.device

    def forward(self,audio,mu = 0,**kwargs):
        if self.args.encoder[0] in ["encodec","q_encodec","mel","vocoder"]:
            loss,info = self.autoencoder(audio,with_info = True)
        elif self.args.encoder[0] in ["mu_encodec"]:
            #Overload the forward function in DiffusionAE to accomodate for mu
            # Encode input to latent channels
            latent, info = self.encode(audio, mu = mu)
            channels = [None] * self.autoencoder.inject_depth + [latent]
            # Adapt input to diffusion if adapter provided
            audio = self.autoencoder.adapter.encode(audio) if exists(self.autoencoder.adapter) else audio
            # Compute diffusion loss
            loss = super(DiffusionAE,self.autoencoder).forward(audio, channels=channels, **kwargs)
        
        elif self.args.encoder[0] in ["rec_encodec"]:
            # Here we have to break the input audio in 2: the first half serves as context, 
            # The second half is what the diffusion model attempts to reconstruct
            B,C,T = audio.shape
            audio_context, audio_rec = audio[:,:,:T//2], audio[:,:,T//2:]

            latent,info = self.autoencoder.encoder(audio_rec,audio_context,with_info = True)
            channels = [None] * self.autoencoder.inject_depth + [latent]
            # Adapt input to diffusion if adapter provided
            audio_rec = self.autoencoder.adapter.encode(audio_rec) if exists(self.autoencoder.adapter) else audio_rec
            # Compute diffusion loss
            loss = super(DiffusionAE,self.autoencoder).forward(audio_rec, channels=channels, **kwargs)
        
        return loss,info
    
    def encode(self,audio,mu = 0):
        #Encoding  Override Diffusion AE method
        if self.args.encoder[0] in ["mel"]:
            encoding = self.autoencoder.encode(audio)
            return encoding, {}

        elif self.args.encoder[0] in ["vocoder"]:
            encoding = self.autoencoder.to_spectrogram(audio)
            return encoding, {}
            
        elif self.args.encoder[0] in ["encodec","q_encodec"]:
            encoding, info = self.autoencoder.encoder(audio,with_info = True)
            return encoding, info
        elif self.args.encoder[0] in ["mu_encodec"]:
            encoding, info = self.autoencoder.encoder(audio,mu = mu,with_info = True)
            return encoding, info
        elif self.args.encoder[0] in ["rec_encodec"]:
            B,C,T = audio.shape
            audio_context, audio_rec = audio[:,:,:T//2], audio[:,:,T//2:]
            encoding,info = self.autoencoder.encoder(audio_rec,audio_context,with_info = True)
            return encoding,info               

    def decode(self,encoding, num_steps = 10,show_progress = False):
        if self.args.encoder[0] in ["mel", "encodec","q_encodec","mu_encodec"]:
            #Decoding - Override inner method
            sample = self.autoencoder.decode(encoding, num_steps= num_steps,show_progress = show_progress)
        elif self.args.encoder[0] in ["vocoder"]:
            sample = self.autoencoder.sample(encoding, num_steps= num_steps,show_progress = show_progress)
        
        elif self.args.encoder[0] in ["rec_encodec"]:
            sample = self.autoencoder.decode(encoding, num_steps= num_steps,show_progress = show_progress)
        
        return sample
    
    def autoencode(self,audio,mu = 0, num_steps = 100, show_progress = False):
        if self.args.encoder[0] in ["mel", "vocoder", "encodec","q_encodec","mu_encodec"]:
            encoding, info = self.encode(audio,mu=mu)
            sample = self.decode(encoding,num_steps=num_steps,show_progress=show_progress)
        
        elif self.args.encoder[0] in ["rec_encodec"]:
            B,C,T = audio.shape
            S = 2*T//(self.autoencoder.encoder.train_sequence_length)
            Th = self.autoencoder.encoder.train_sequence_length//2
            audio = audio.view(B,C,S,Th)
            audio = torch.permute(audio, [2,0,1,3]) #Permute to have slice in dimension 0


            sample = []
            encoding = []
            info = {"spikes": [],"rep_loss": []}
            audio_inf = torch.zeros(B,C,Th).to(audio.device)
            #Recursive Loop over time slices
            #audio is [S,B,C,Th]
            for slice in audio:
                #Concatenate the previously diffused audio with the next audio so that it serves the context
                audio_in = torch.cat((audio_inf, slice), dim = -1)
                latent, information = self.encode(audio_in)
                audio_inf = self.decode(latent, num_steps=num_steps,show_progress=show_progress)
                sample.append(audio_inf) # dim: B x C x Th
                encoding.append(latent)
                info["rep_loss"].append(information["rep_loss"])
                info["spikes"].append(information["spikes"])

            #Stacking and reshaping
            sample = torch.cat(sample,dim=2) # dim: B x C x T
            encoding = torch.cat(encoding,dim=2)
            info["rep_loss"] = torch.mean(info["rep_loss"])
            info["spikes"] = torch.cat(info["spikes"],dim = 2)
            
            #print(f"sample shape {sample.shape}")
            #print(f"encoding shape {encoding.shape}")
            #print(f"spikes shape {info['spikes'].shape}")
            #print(f"rep_loss shape {info['rep_loss'].shape}")
        return sample, encoding, info
    def _build_encoder(self):
        if self.args.encoder == ["mel"]:
            return MelE1d( # The encoder used, in this case a mel-spectrogram encoder
                in_channels=self.args.in_channels,
                channels=512,
                multipliers=[1, 1],
                factors=[2],
                num_blocks=[12],
                out_channels=32,
                mel_channels=80,
                mel_sample_rate=self.args.sample_rate,
                mel_normalize_log=True,
                bottleneck=TanhBottleneck(),
            )
        elif self.args.encoder == ["encodec"]:
            return SpikingEncodecEncoder(self.args)
        elif self.args.encoder == ["q_encodec"]:
            return QuantizingEncodecEncoder(self.args)
        elif self.args.encoder == ["mu_encodec"]:
            return MuSpikingEncodecEncoder(self.args)
        elif self.args.encoder == ["rec_encodec"]:
            return RecursiveSpikingEncodecEncoder(self.args)

      
if __name__=="__main__":
    from torchinfo import summary
    args = make_parser().parse_args()
    model = SpikingAudioDiffusion(args)
    
    dataloader = from_maestro(args)
    count = 0
    for feature in dataloader:
        print("In")
        print(f"Feature {count}: Shape {feature['audio'].shape}")
        count +=1
        audio = feature["audio"]
    print(f"audio shape {audio.shape}")
    # Train autoencoder with audio samples
    loss = model(audio)
    print(f"Loss: {loss}")
    #loss.backward()
    
    #print(model.autoencoder.encoder)
    summary(model, input_data = audio)

    print(f"Model Device {model.get_device()}")

