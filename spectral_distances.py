from analysis import Inferer
from parser import dotdict,override_args_from_dotdict
import numpy as np
import os
import torch
import torch.nn as nn


model_t80 = dotdict({
    "name":"spikes_transformer_80",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/spikes_transformer_80/weights-300000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,

    "split":"validation",
    "clip_limit":None,
})

model_q = dotdict({
    "name": "quant_8x10",
    "encoder": ["q_encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/quant_8x10/weights-300000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,

    "split":"validation",
    "clip_limit":None,
})

models = [model_t80,model_q]
if __name__ == "__main__":

    for model in models:
        args = override_args_from_dotdict(model)
        out_path = "/lcncluster/lisboa/spikes_audio_diffusion/data/"+args.name
        os.makedirs(out_path,exist_ok=True)
        inf = Inferer(args,device = torch.device('cuda'))
        specMSE = inf.inf_specMSE()

        outfile = out_path+'/specMSE.npz'
        with open(outfile,'wb') as f:
            np.savez(f,specMSE = specMSE.cpu().numpy())  
        
        print(f"Spec MSE: {specMSE.mean()} +- {specMSE.std()}")