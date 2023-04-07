import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self

params = AttrDict(
  # Training params
  batch_size=32,
  learning_rate=1e-4,
  betas= [0.95,0.99],
  eps = 1e-6,
  w_decay = 1e-3,
  max_grad_norm=None,
  goal_beta = 1.,
  loss_coeff = 1.,
  save_step = 50,
  save_model_step = 25000,
  
  #Encoder Parameters
  encoder_params = AttrDict(

    #bottleneck dim -> nr of neurons (spiking bottlenecks)/ codewords(vq-vae)
    bottleneck_dim = 128, 
    lstm_hidden_size = 128,
    encodec_dim = 128,

    inject_depth = 6,

    lstm = False,

    encodec_ratios = [8,8,4,2],
    
    firing_rate_threshold = 1e-1, #spikes/frame
    in_channels = 1
  ),

  dataset_params = AttrDict(
    dataset_folder="/lcncluster/datasets/maestro",
    sample_rate = 22050, 
    sequence_time = 2**17/22050, #number of seconds of audio in each of the samples from the dataset
    download=True,
    split="train",
    midi_sampling_rate=100,
    data_limit = None,
    clip_limit = None,
  )
)