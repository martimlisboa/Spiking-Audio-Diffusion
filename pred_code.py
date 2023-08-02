import numpy as np
import math
import soundfile as sf
import subprocess
import os
import matplotlib.pyplot as plt
from analysis import Inferer
import torch.nn as nn


import torch
import torchaudio as T
import torchaudio.transforms as TT
import torch.nn.functional as F

from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from parser import dotdict, override_args_from_dotdict

from model_list import model_80brute50,model_80brute,model_80free,model_sparse

def read_audios(input_path):
    files = os.listdir(input_path) 
    audios = []
    audio_names = []
    for name in files:
        print(name)
        audio_names.append(name[:-4])
        audio_path = input_path+name
        audio, sr = T.load(audio_path)
        audios.append(audio)

    audios = torch.stack(audios,dim = 0)
    print(f"Audios Shape: {audios.shape}")
    #print(f"Audio shape : {audio.shape}\n")
    t = audios.shape[-1]
    target_t = 2 ** math.ceil(math.log2(t))

    audios = F.pad(audios,(0,target_t-t))
    #audio = audio[:,:2**16]
    audios = audios.to(torch.device('cuda'))
    print(f"Audios shape : {audios.shape}\n")
    return audios, audio_names

class MelSpectrogram(nn.Module):
    def __init__(self):
        super(MelSpectrogram, self).__init__()
        self.mel_args = {
          'sample_rate': 22050,
          'win_length': 2048,
          'hop_length': 512,
          'n_fft': 2048,
          'n_mels': 128,
          'power': 1.0,# magnitude of stft
          'normalized': True,
        } 
        self.mel_spec_transform = TT.MelSpectrogram(**(self.mel_args))
    def forward(self,x):
        x = self.mel_spec_transform(x)
        return x 


models = [model_80free,model_80brute50,model_80brute,model_sparse]
def main():
    input_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/pred/"
    output_path = "/lcncluster/lisboa/spikes_audio_diffusion/data/"
    # Read the pred/unpred samples
    audios,audio_names = read_audios(input_path)
    mel_spec= MelSpectrogram().to('cuda')
    # Autoencode them with each model
    with torch.no_grad():
        for model in models:
            args = override_args_from_dotdict(model)
            out_path = f"{output_path}/{args.name}"
            os.makedirs(out_path,exist_ok=True)
            inf = Inferer(args,device = torch.device('cuda'))
            audios_inf, _, info = inf.autoencode(audios,num_steps = 100)
            sisnr = inf.SISNR(audios,audios_inf)
            print(f"Audios inf: Shape {audios_inf.shape}")
            print(f"SI-SNR: {sisnr}")
            if args.encoder[0] == "encodec":
                bps = inf.count_bps(info["spikes"])
                print(f"Bit Rates: {bps}")
                latent = info["spikes"]
            elif args.encoder[0] == "q_encodec":
                n_q = inf.model.autoencoder.encoder.quantizer.n_q
                bpf_per_quantizer = math.ceil(math.log2(inf.model.autoencoder.encoder.quantizer.bins))
                conv_bps = inf.model.args.sample_rate/inf.model.autoencoder.encoder.downsample_factor
                bps = n_q * bpf_per_quantizer * conv_bps * torch.ones_like(sisnr)
                latent = torch.permute(info['codes'],[1,0,2])

            #spectrograms
            spec_in = mel_spec(audios)
            spec_out = mel_spec(audios_inf)


            # Save latent + input and output spectrogram
            pred_file = out_path + "/pred_code.npz"
            with open(pred_file,'wb') as f:
                np.savez(f,
                         latent = latent.detach().cpu().numpy(),
                         spec_in = spec_in.detach().cpu().numpy(),
                         spec_out = spec_out.detach().cpu().numpy(),
                         names = audio_names,
                         bps = bps.detach().cpu().numpy(),
                         sisnr = sisnr.detach().cpu().numpy())

if __name__ == "__main__":
    main()
