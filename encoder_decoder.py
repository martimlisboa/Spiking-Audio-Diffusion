import numpy as np
import math
import torch
import torchaudio as T
import torchaudio.transforms as TT
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

import os
import sys

from model import SpikingAudioDiffusion
from parser import make_parser
import time

Models = {}

def count_bpf(bottleneck_output, conv = 1): #Returns bits per frame
    
    print(f"bottleneck shape: {bottleneck_output.shape}")
    nr_spikes  = bottleneck_output.sum()
    print(f"nr_spikes: {nr_spikes}")
    time_frames = bottleneck_output.shape[-1]
    print(f"time_frames: {time_frames}")

    nr_neurons = bottleneck_output.shape[-2]
    print(f"nr_neurons: {nr_neurons}")

    nr_time_bits = math.ceil(math.log2(time_frames))
    print(f"nr_time_bits: {nr_time_bits}")

    nr_neuron_bits = math.ceil(math.log2(nr_neurons))
    print(f"nr_neuron_bits: {nr_neuron_bits}")

    bpf = nr_spikes *(nr_time_bits + nr_neuron_bits)/time_frames
    print(f"SPARSE? bits per frame: {bpf}")

    firing_rate = bottleneck_output.mean()
    print(f"mean firing rate: {firing_rate}")

    bpf = bpf if bpf < nr_neurons else nr_neurons
    print(f"Final Bit rate {bpf} bit/frame = {bpf*conv} bit/s")
    return bpf



def load_model(args, model_name, device):
    t0 = time.time()
    if not model_name in Models:
        print(f"loading model {model_name} from {args.model_dir}")
        if os.path.exists(f'{args.model_dir}/weights.pt'):
            checkpoint = torch.load(f'{args.model_dir}/weights.pt')
        else:
            checkpoint = torch.load(args.model_dir)
        
        model = SpikingAudioDiffusion(args).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        Models[model_name] = model
    dt = time.time()-t0
    print(f"Loading Time: {dt}")

def main(args):
    names = ["bach","nocturne","scriabin","beethoven"]
    input_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/"
    audios = []
    _, sr = T.load(input_path+"bach.wav")
    for name in names:
        audio_path =input_path+name+".wav"
        audio, sr = T.load(audio_path)
        audio = audio.unsqueeze(0)
        audios.append(audio)


    audios = tuple(audios)
    audios = torch.cat(audios,dim = 0)
    print(f"Audios shape: {audios.shape}")
    
    #audio, sr = T.load(args.input)
    #print(f"Audio shape : {audio.shape}\n")
    t = audios.shape[-1]
    target_t = 2 ** math.ceil(math.log2(t))

    audios = F.pad(audios,(0,target_t-t))
    #audio = audio[:,:2**16]
    audios = audios.to(torch.device('cuda'))
    print(f"Audios shape : {audios.shape}\n")
    
    #Load Model
    model_name = "model_1"
    load_model(args, model_name, device = torch.device('cuda'))

    for name in names:
        out_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/"+name
        os.makedirs(out_path,exist_ok=True)
        out_path += "/"+ args.output
        os.makedirs(out_path,exist_ok=True)


    model = Models[model_name]
    #Encode
    t0 = time.time()
    print("\n")
    if args.encoder[0] in ["mel","vocoder"]:
        latent,_ = model.encode(audios) # Encode
    elif args.encoder[0] in ["encodec"]:
        latent, info = model.encode(audios) # Encode
        spikes = info["spikes"]
        print(f"spikes shape: {spikes.shape}")
    elif args.encoder[0] in ["b_encodec"]:
        latent, info = model.encode(audios) # Encode
        spikes = torch.where(info["spikes"],1.,0.)
        print(f"spikes shape: {spikes.shape}")
    elif args.encoder[0] in ["q_encodec"]:
        latent, info = model.encode(audios) # Encode
        codes = torch.permute(info['codes'],[1,0,2])
        print(f"codes shape: {codes.shape}")
    print(f"latent shape: {latent.shape}")
    dt = time.time() - t0
    print(f"Encoding Time: {dt}\n")

    #Decode
    t0 = time.time()
    audio_inf = model.decode(latent, num_steps=100) # Decode by sampling diffusion model conditioning on latent
    print(f"output audios shape: {audio_inf.shape}")
    dt = time.time() - t0
    print(f"Decoding Time: {dt}")
    

    sisnr = scale_invariant_signal_noise_ratio(audio_inf,audios).squeeze(1)
   


    for i,name in enumerate(names):
        output_audio_filename = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/"+name+"/"+ args.output+"/audio_"+args.encoder[0]+"_inf.wav";
        T.save(output_audio_filename,audio_inf[i].cpu(),sample_rate = sr) #save output audio
        outfile = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/"+name+"/"+ args.output+"/archive.npz"
        with open(outfile,'wb') as f:
            np.savez(f,bottleneck = latent[i].detach().cpu().numpy().T)

    #Plotting
                
    if args.plot_wavs:
        fig, ax = plt.subplots(figsize=(16,9))
        for i,audio in enumerate(audio_inf.cpu()):
            ax.plot(audio[0])
            fig.tight_layout()
            filename = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/"+names[i]+"/"+ args.output +"/"+"plot_wavs_"+args.encoder[0]+".png";
            plt.savefig(filename)
            plt.cla()
 

    if args.plot_latent:
        fig, ax = plt.subplots(figsize=(16,9))
        for i,l in enumerate(latent.detach().cpu()):
            im = ax.imshow(l,aspect='auto')
            plt.colorbar(im, ax=ax)
            fig.tight_layout()
            filename = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/"+names[i]+"/"+ args.output +"/"+"plot_latent_"+args.encoder[0]+".png";
            plt.savefig(filename)
            plt.cla()

    if args.plot_spikes:
        fig, ax = plt.subplots(figsize=(16,9))
        for i,z in enumerate(spikes.detach().cpu()):
            print(f"\nSample:{names[i]}")
            print(f"SI-SNR: {sisnr[i]}")
            print(f"Bit Rate")
            count_bpf(z, conv = args.sample_rate/model.autoencoder.encoder.downsample_factor)
            ax.imshow(z,cmap="Greys",interpolation="none",aspect='auto')
            fig.tight_layout()
            filename = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/"+names[i]+"/"+ args.output +"/"+"plot_spikes_"+args.encoder[0]+".png";
            plt.savefig(filename)
            plt.cla()

    if args.plot_codes:
        fig, ax = plt.subplots(figsize=(16,9))
        for i,z in enumerate(codes.detach().cpu()):
            print(f"\nSample:{names[i]}")
            print(f"SI-SNR: {sisnr[i]}")
            ax.imshow(z,interpolation="none",aspect='auto')
            fig.tight_layout()
            filename = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/"+names[i]+"/"+ args.output +"/"+"plot_codes_"+args.encoder[0]+".png";
            plt.savefig(filename)
            plt.cla()

      
if __name__ == '__main__':
  parser = make_parser()
  #Extra Arguments
  parser.add_argument("--input",'-i',default="wavs_test/audio_spikingmusic.wav",
        help = "input wav file name")
  parser.add_argument('--output', '-o', default='wavs_test',
      help='output dir name')
  parser.add_argument('--plot_wavs', action = 'store_true', default=False,
      help='plot the wavs') 
  parser.add_argument('--plot_latent',  action = 'store_true', default=False,
      help='plot the latent') 
  parser.add_argument('--plot_spikes', action = 'store_true', default=False,
      help='plot the spikes') 
  parser.add_argument('--plot_codes', action = 'store_true', default=False,
      help='plot the codes') 

  main(parser.parse_args())