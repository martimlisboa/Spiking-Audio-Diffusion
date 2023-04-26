from analysis import Inferer
from parser import dotdict,override_args_from_dotdict
import numpy as np
import os
import torch
import torch.nn as nn

from model_list import model_t80,model_q,model_t128,model_bad,model_mel

model_test = dotdict({
    "name":"spikes_transformer_80",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/spikes_transformer_80/weights-300000.pt",
    "data_dirs":["maestro"],
    "batch_size":5,
    "bottleneck_dim":80,
    "transformer": True,

    "split":"validation",
    "clip_limit":10,
})

models = [model_mel]

if __name__ == "__main__":

    for model in models:
        args = override_args_from_dotdict(model)
        out_path = "/lcncluster/lisboa/spikes_audio_diffusion/data/"+args.name
        os.makedirs(out_path,exist_ok=True)
        inf = Inferer(args,device = torch.device('cuda'))
        #Get the metrics using the inferer object
        data = inf.infer()
        specMSE = data["specMSE"]
        wavMSE = data["wavMSE"]
        sisnr = data["sisnr"]
        bps = data["bps"]


        outfile = out_path+'/metrics.npz'
        with open(outfile,'wb') as f:
            np.savez(f,
                     specMSE = specMSE.cpu().numpy(),
                     wavMSE = wavMSE.cpu().numpy(),
                     sisnr = sisnr.cpu().numpy(),
                     bps = bps.cpu().numpy())  
        
        for k,v in data.items():
            print(f"{k}: {v.mean()} +- {v.std()}")