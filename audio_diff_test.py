import numpy as np
import os
import torch
import torch.nn as nn
from torchinfo import summary

from models.encoder_models import SpikingEncodecEncoder
from parser import make_parser
from model import SpikingAudioDiffusion

if __name__ == "__main__":
    args = make_parser().parse_args()
    model = SpikingAudioDiffusion(args)
    x = torch.randn(3, 1, 2**15)
    print(f"Input Shape: {x.shape}")
    loss,info = model(x)
    print(f"Representation Loss in Learner: {info['rep_loss']}")


    summary(model.autoencoder,input_data = x)

