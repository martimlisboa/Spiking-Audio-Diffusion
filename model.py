import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler,DiffusionVocoder
from audio_encoders_pytorch import MelE1d, TanhBottleneck

from parser import make_parser
from dataset_utils import MaestroDataset,from_maestro

from models import SpikingEncodecEncoder, QuantizingEncodecEncoder


class SpikingAudioDiffusion(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        #Encoder Stuff
        if args.encoder[0] in ["mel", "encodec","q_encodec"]:
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

    def forward(self,audio):       
        loss,info = self.autoencoder(audio,with_info = True)
        return loss,info
    
    def encode(self,audio):
        #Encoding  Override Diffusion AE method
        if self.args.encoder[0] in ["mel"]:
            encoding = self.autoencoder.encode(audio)
            return encoding

        elif self.args.encoder[0] in ["vocoder"]:
            encoding = self.autoencoder.to_spectrogram(audio)
            return encoding
            
        elif self.args.encoder[0] in ["encodec","q_encodec"]:
            encoding, info = self.autoencoder.encoder(audio,with_info = True)
            return encoding, info

    def decode(self,encoding, num_steps = 10,show_progress = False):
        if self.args.encoder[0] in ["mel", "encodec","q_encodec"]:
            #Decoding - Override inner method
            sample = self.autoencoder.decode(encoding, num_steps= num_steps,show_progress = show_progress)
        elif self.args.encoder[0] in ["vocoder"]:
            sample = self.autoencoder.sample(encoding, num_steps= num_steps,show_progress = show_progress)
        return sample
    
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

