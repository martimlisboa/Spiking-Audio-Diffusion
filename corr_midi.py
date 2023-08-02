from analysis import Inferer
from parser import dotdict,override_args_from_dotdict
import numpy as np
import os
import torch
import torch.nn as nn

from model_list import model_80brute50

model_test = dotdict({
    "name":"spikes_transformer_80",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80full4/weights-200000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "spike_function":"full",
    "transformer": True,
    "batch_norm": True,


    "split":"validation",
    "clip_limit":64,
    "midi_sampling_rate":43,
})
models = [model_80brute50]
def main():
    for model in models:
        args = override_args_from_dotdict(model)
        out_path = "/lcncluster/lisboa/spikes_audio_diffusion/data/"+args.name
        os.makedirs(out_path,exist_ok=True)
        inf = Inferer(args,device = torch.device('cuda'))
        corr = inf.midi_correlate()
        
        midi_corr_path = f"{out_path}/midi_corr.npz"
        with open(midi_corr_path,'wb') as f:
            np.savez(f,corr = corr.cpu().numpy())
 

if __name__ == "__main__":
    main()