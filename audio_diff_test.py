import numpy as np
import os
import torch
import torch.nn as nn
from torchinfo import summary

from models.encoder_models import SpikingEncodecEncoder
from parser import make_parser,make_scaffold_parser,dotdict,override_args_from_dotdict
from model import SpikingAudioDiffusion


model_1 = dotdict({
    "encoder":["encodec"],
    "model_dir":["dummy_model_dir"],
    "data_dirs":["dummy_data_dirs"],
    "spike_function": "moving",
    "bottleneck_dim":80,
    "transformer": True
})



if __name__ == "__main__":

    args = override_args_from_dotdict(model_1)
    model = SpikingAudioDiffusion(args)
    x = torch.randn(3, 1, 2**15)
    print(f"Input Shape: {x.shape}")
    loss,info = model(x)
    print(f"Representation Loss in Learner: {info['rep_loss']}")


    summary(model.autoencoder,input_data = x)

