import numpy as np
import os
import torch
import torch.nn as nn
from torchinfo import summary

from models.encoder_models import SpikingEncodecEncoder,RecursiveSpikingEncodecEncoder
from parser import make_parser,make_scaffold_parser,dotdict,override_args_from_dotdict
from model import SpikingAudioDiffusion
from analysis import Inferer
from make_experiment_audio import read_originals

model_mu = dotdict({
    "name":"mu",
    "encoder":["mu_encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/mu_encodec_0.1",
    "data_dirs":["maestro"],
    "bottleneck_dim":128,
    "transformer": True,
    "transformer_conv": True,
    "sequence_length": 32768,
    "clip_limit":10,
})


def main():
    args = override_args_from_dotdict(model_mu)
    inf = Inferer(args,device = 'cuda')
    audios,_,sr = read_originals()
    B,C,T = audios.shape
    mu_values = torch.arange(0.25,1.05,0.05)
    print(mu_values)
    for mu0 in mu_values:
        print(f"mu = {mu0}")
        audios_inf, latent, info = inf.autoencode(audios, mu = mu0*torch.ones(B,1,1,device = inf.device),num_steps=100,show_progress=False)
        rep = info["spikes"]
        bps = inf.count_bps(rep)
        mus = bps/(inf.model.autoencoder.encoder.B0 * sr / inf.model.autoencoder.encoder.downsample_factor)
        print(mus)
if __name__ == "__main__":
    main()    