import numpy as np
import math
import soundfile as sf
import subprocess
import os, shutil
import matplotlib.pyplot as plt

import torch
import torchaudio as T
import torchaudio.transforms as TT
import torch.nn.functional as F

from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

from dataset_utils import MaestroDataset
from parser import dotdict, override_args_from_dotdict
from analysis import Inferer
from model_list import model_sparse
test_model = dotdict({
    "name":"spikes_transformer_128",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/encodec_transformer",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":128,
    "transformer": True,

    "split":"validation",
    "clip_limit":None,
})
#List of indices infered form the list. These excerpts are not too crazy
sample_list = [2725, 2750, 4063, 4610,4695, 4950, 5086, 11056]
#models = [model_80free,model_80trough4,model_1024troughA4,model_4096troughA4,model_q,model_mel]
models = [model_sparse]

def list_dataset(dataset):
    N = len(dataset.index_table)
    prev_index,_,_ = dataset.index_table[0]
    last_i = 0
    f = open(f"list_MAESTRO_"+dataset.split+".txt", "a")

    for i in range(N):
        i_song, _, _ = dataset.index_table[i]
        if i_song != prev_index:
            f_out = f"Song: {dataset.get_title(i-1)}     {last_i} - {i-1}\n"
            #print(f_out)
            f.write(f_out)
            prev_index = i_song
            last_i = i
    f.close()
    return

def resample_dir(dir,sampling_rate):
    for file in sorted(os.listdir(dir)):
        filename = os.fsdecode(file)
        a = filename.endswith(".wav")
        b = not filename.endswith(f'{sampling_rate}.wav')
        if a and b :
            new_filename = f'{filename[:-4]}_{sampling_rate}.wav'
            print(f'{filename}->{new_filename}')
            audio, sr = T.load(f'{dir}/{filename}')
            audio_resampled = F.resample(audio,sr,sampling_rate)
            out_file = f'{dir}/{new_filename}'
            print(f"Audio Shape: {audio.shape}, Audio R Shape {audio_resampled.shape}")
            T.save(out_file,audio_resampled,sample_rate = sampling_rate,
                   bits_per_sample = 16) #save output audio
            #_, srF = T.load(out_file)
            #print(srF)

def make_originals():
    print("Making originals")
    args = override_args_from_dotdict(test_model)
    dataset = MaestroDataset(args)
    #list_dataset(dataset)
    #print("Listed DATASET in file")
    originals_dir = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/originals"
    os.makedirs(originals_dir,exist_ok=True)
    originals_dir_mp3 = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/originals/mp3"
    os.makedirs(originals_dir_mp3,exist_ok=True)
    audios = []
    audio_names =[]
    for i,sample_idx in enumerate(sample_list):
        print(f"{sample_idx} - Song: {dataset.get_title(sample_idx)}")
        item = dataset.__getitem__(sample_idx)
        audio = item["audio"] 
        audio = torch.clamp(torch.Tensor(audio),min = -1., max = 1.)
        audio = audio.unsqueeze(0)
        #print(audio.shape)
        name = "audio_"+str(i+1)
        audio_names.append(name)
        audios.append(audio)

        output_audio_filename = f"{originals_dir}/{name}.wav";
        T.save(output_audio_filename,audio.cpu(),sample_rate = dataset.sampling_rate) #save output audio
        subprocess.call(f'ffmpeg -y -i {originals_dir}/{name}.wav -b:a 192k {originals_dir_mp3}/{name}.mp3 -loglevel quiet', shell=True) #Convert to mp3 for gorilla. -y flag automates overwrite

    audios = torch.stack(audios,dim = 0).to("cuda")
    return audios,audio_names,dataset.sampling_rate


def read_originals():
    originals_dir = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/originals/"
    file_list = os.listdir(originals_dir)
    file_list = [f for f in file_list if f[-4:] == ".wav"]
    file_list = [f for f in file_list if f[-9:] != "48000.wav"]
    
    audio_names = [f[:-4] for f in file_list]
    audios = [T.load(originals_dir+f)[0] for f in file_list]
    audios = torch.stack(audios,dim = 0).to("cuda")
    audios.unsqueeze(1)
    sr = T.load(originals_dir+file_list[0])[1] #return sampling rate here
    return audios, audio_names, sr


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

def mix_with_randn(audios,m):
    mean2=audios.pow(2).mean(2).unsqueeze(1)
    print(f"mean2 shape : {mean2.shape}")
    noise = torch.mul(torch.sqrt(mean2),torch.randn_like(audios))#2*torch.rand_like(audios)-1
    noise = torch.clamp(noise,min=-1,max = 1)
    print(f"Noise Shape: {noise.shape}")
    #mix audios with noise
    anchors = m*noise + (1-m)*audios
    return anchors

def quantize_wav(audios,n):
    std=torch.sqrt(audios.pow(2).mean(2).unsqueeze(1))

    anchors = torch.round(audios*n/std,decimals = 0)
    anchors = anchors*std/n
    #Round to nearest integer

    return anchors

def make_anchors():
    audios,audio_names,sr = read_originals()
    anchors = quantize_wav(audios,0.5)
    anchors = mix_with_randn(anchors,0.2)
    sisnr = scale_invariant_signal_noise_ratio(audios.squeeze(1),anchors.squeeze(1))

    print("Shapes")
    print(f"Audio: {audios.shape}, Anchor: {anchors.shape}, sisnr: {sisnr.shape}")

    for i,name in enumerate(audio_names):
        anchor = anchors[i]
        print(f"{name} anchor: SI-SNR = {sisnr[i]}")

        out_path = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/anchors"
        os.makedirs(out_path,exist_ok=True)
        out_path_mp3 = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/anchors/mp3"
        os.makedirs(out_path_mp3,exist_ok=True)

        output_anchor_filename = out_path+"/anchor_"+name+".wav";
        T.save(output_anchor_filename,anchor.cpu(),sample_rate = sr) #save output audio
        subprocess.call(f'ffmpeg -y -i experiment/anchors/anchor_{name}.wav -b:a 192k experiment/anchors/mp3/anchor_{name}.mp3 -loglevel quiet', shell=True) #Convert to mp3 for gorilla. -y flag automates overwrite
    
    print("Made Anchors")
    return

def make_audio_inf(model,audios,audio_names,sr):
    args = override_args_from_dotdict(model)
    out_path = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/"+args.name
    os.makedirs(out_path,exist_ok=True)
    out_path_mp3 = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/"+args.name+'/mp3'
    os.makedirs(out_path_mp3,exist_ok=True)
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

    if args.name not in ["mel","MEL"]:
        metrics_file = out_path + "/metrics.npz"
        with open(metrics_file,'wb') as f:
            np.savez(f,bottleneck = latent.detach().cpu().numpy(), bps = bps.detach().cpu().numpy(), sisnr = sisnr.detach().cpu().numpy())
    
    for i,name in enumerate(audio_names):
        path = out_path+f"/{args.name}_{name}.wav"
        path_mp3 = out_path_mp3+f"/{args.name}_{name}.mp3"
        T.save(path,audios_inf[i].cpu(),sample_rate = sr) #save output audio
        print(f"Making mp3: {args.name}")
        subprocess.call(f'ffmpeg -y -i {path} -b:a 192k {path_mp3} -loglevel quiet', shell=True)
        
    #resample_dir(out_path,48000)


def make_opus(opus):
    opus_dir = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/"+opus
    os.makedirs(opus_dir,exist_ok=True)
    opus_dir_mp3 = f"/lcncluster/lisboa/spikes_audio_diffusion/experiment/{opus}/mp3"
    os.makedirs(opus_dir_mp3,exist_ok=True)
    opus_bps = int(opus[4:])

    originals_dir = "/lcncluster/lisboa/spikes_audio_diffusion/experiment/originals/"

    for audio_name in os.listdir(originals_dir):
        if audio_name[-4:] == '.wav':
            audio_path = originals_dir+audio_name
            opus_path = f"{opus_dir}/{audio_name[:-4]}.opus"
            out_path = f"{opus_dir}/{audio_name[:-4]}_{opus}.wav"
            print(audio_name)
            print(audio_path)
            print(opus_path)
            print(out_path)
            print("\n")

            os.system(f"opusenc --bitrate {opus_bps} {audio_path} {opus_path} --hard-cbr")
            #decode the opus to wav
            #print("Dec")
            os.system(f"opusdec {opus_path} {out_path}")
            #print("Done")

def gather_mp3():
        rootdir = "/lcncluster/lisboa/spikes_audio_diffusion/experiment"
        dirs = os.listdir(rootdir)
        mp3dir = rootdir + '/mp3'
        os.makedirs(mp3dir,exist_ok=True)
        for dir in dirs:
            if dir not in ['mp3','data']:
                in_dir = f'{rootdir}/{dir}/mp3/'
                mp3s = os.listdir(in_dir)
                for mp3 in mp3s:
                    subprocess.call(f'cp {in_dir}/{mp3} {mp3dir}', shell=True) #Convert to mp3 for gorilla. -y flag automates overwrite
        print(f'Copied all mp3 files to {mp3dir}')

def clear_dir():
    folder = "/lcncluster/lisboa/spikes_audio_diffusion/experiment"
    for filename in os.listdir(folder):
        if filename != "data": #Never clear the data folder
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    print('Cleared Root Dir')

    
def main():
    
    #clear_dir()
    #audios, audio_names,sr = make_originals()
    
    #make_anchors()
    #exit(0)
    
    audios, audio_names,sr = read_originals()
    print(f"AUDIOS: Shape: {audios.shape}, device: {audios.device}\n\n")
    print(f"Audio Names: {audio_names}")

    with torch.no_grad():
        for model in models:
            make_audio_inf(model,audios,audio_names,sr)

    #Gather all mp3 in a single folder:
    
    make_opus('opus12')
    gather_mp3()
        

if __name__ == "__main__":
        main()