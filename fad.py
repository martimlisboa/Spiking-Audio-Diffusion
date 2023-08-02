import pytorch_fad
import numpy as np
import math
import soundfile as sf
import subprocess
import os, sys, shutil
import matplotlib.pyplot as plt

import torch
import torchaudio as T
import torchaudio.transforms as TT
import torch.nn.functional as F
import time
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio



from dataset_utils import MaestroDataset
from parser import dotdict, override_args_from_dotdict

from model_list import model_80brute50,model_80free,model_RVQ8x10,model_80brute,model_bad,model_mel,model_80trough4,model_RVQ_50

from analysis import Inferer



model_list = [model_bad,model_80trough4,model_RVQ_50,model_mel,model_80brute50,model_80free,model_RVQ8x10]
#model_list = [model_bad,model_80brute]

test_model = dotdict({
    "name":"spikes_transformer_128",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/encodec_transformer",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":128,
    "transformer": True,

    "split":"train",
    "clip_limit":None,
})

def make_base(nr_audios,verbose = False):
    print("Making base")
    args = override_args_from_dotdict(test_model)
    dataset = MaestroDataset(args)
    #list_dataset(dataset)
    #print("Listed DATASET in file")
    base_dir = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/fad"
    os.makedirs(base_dir,exist_ok=True)
    N = len(dataset.index_table)

    sample_list = np.random.choice(range(N),nr_audios,replace = False)
    print(sample_list)

    for i,sample_idx in enumerate(sample_list):
        if verbose:
            print(f"{sample_idx} - Song: {dataset.get_title(sample_idx)}")
        item = dataset.__getitem__(sample_idx)
        audio = item["audio"] 
        audio = torch.clamp(torch.Tensor(audio),min = -1., max = 1.)
        audio = audio.unsqueeze(0)
        #print(audio.shape)
        name = "audio_"+str(i+1)

        output_audio_filename = f"{base_dir}/{name}.wav";
        T.save(output_audio_filename,audio.cpu(),sample_rate = dataset.sampling_rate) #save output audio

    return

def make_temp(model_inf,tmp_dir, nr_audios, batch_size):
    N = len(model_inf.dataloader.dataset.index_table) 
    n_batch = nr_audios//batch_size
    sample_list = np.random.choice(range(N),nr_audios,replace = False)
    sample_list = [[s for s in sample_list[k*batch_size:(k+1)*batch_size]] for k in range(n_batch)]

    for b,batch in enumerate(sample_list):
        print(f"batch {b}/{n_batch}")
        b_audios = []
        b_audio_names =[]
        for i,sample_idx in enumerate(batch):
            #print(f"{sample_idx} - Song: {dataset.get_title(sample_idx)}")
            item = model_inf.dataloader.dataset.__getitem__(sample_idx)
            audio = item["audio"] 
            audio = torch.clamp(torch.Tensor(audio),min = -1., max = 1.)
            audio = audio.unsqueeze(0)
            #print(audio.shape)
            name = "audio_"+str(i + b*batch_size)
            b_audio_names.append(name)
            b_audios.append(audio)

        b_audios = torch.stack(b_audios,dim = 0).to("cuda")       
        audios_inf, _, _ = model_inf.autoencode(b_audios,num_steps = 100)
        for i,audio in enumerate(audios_inf):
            output_audio_filename = f"{tmp_dir}/{b_audio_names[i]}.wav";
            T.save(output_audio_filename,audio.cpu(),sample_rate = model_inf.model.args.sample_rate) #save output audio
       

        del b_audios
        del audios_inf
        del b_audio_names

    return

def make_base_metric(base_path):
    metric = pytorch_fad.FADMetric(device='cuda', base_path = base_path)
    m,s = metric.base_m, metric.base_s

    metrics_file = f"{base_path}/metrics.npz"
    with open(metrics_file,'wb') as f:
        np.savez(f,mu = m, sigma = s)
    
def fad(base_path, target_path):
    metric = pytorch_fad.FADMetric(device='cuda', base_path = base_path)
    fad_score = metric.compare_base_to_path(target_path)
    print(f"FAD: {fad_score}")

    return fad_score


def FAD(model,base_path,target_path,nr_iters,nr_audios=500,batch_size=50):

    fads = []
    t0 = time.time()
    tstart = t0
    tmp_dir = target_path
    print(f"Making tmp {model.name}")
    args = override_args_from_dotdict(model)
    #list_dataset(dataset)
    #print("Listed DATASET in file")
    inf = Inferer(args,device = torch.device('cuda'))

    for iter in range(nr_iters):
        #Make the tmp dir
        print(f"Making tmp {model.name}: {iter+1}/{nr_iters}")
        os.makedirs(tmp_dir,exist_ok=True)
        #Fill the temp dir up with audios
        make_temp(inf,tmp_dir = tmp_dir,nr_audios=nr_audios,batch_size=batch_size)
        
        #Compute the FAD distance for this pool of audios
        fads.append(fad(base_path,target_path))

        #Remove the tmp dir
        # Try to remove the tree; if it fails, throw an error using try...except.
        try:
            shutil.rmtree(tmp_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        
        t = time.time()
        print(f"Time iter {iter+1}: {t-t0}")
        t0 = t

    tend = time.time()
    print(f"Time Total: {tend-tstart}")
    return np.array(fads)

def main():
    #make_base_metric(base_path="/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/fad")
    #make_temp(model_80free,tmp_dir = target_path,nr_audios = 300,batch_size = 50)
    
    #Base already stored in fad/metrics.npz
    base_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/fad/metrics.npz"
    target_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/tmp"

    for model in model_list:
        fads = FAD(model,base_path,target_path, nr_iters=20, nr_audios=500,batch_size=25)
        print(f"FADs : {fads}")
        disp = fads.std()**2/fads.mean()
        print(f"dispersion: {disp}")

        dir_path =f"/lcncluster/lisboa/spikes_audio_diffusion/data/{model.name}"
        os.makedirs(dir_path,exist_ok=True)

        data_file = dir_path + "/fads.npz"
        with open(data_file,'wb') as f:
            np.savez(f,fads = fads)

if __name__ == "__main__":
    main()