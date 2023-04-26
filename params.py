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
    bottleneck_dim = 80, #By default equal to the number of bpf of quantizer 
    encodec_dim = 128,

    inject_depth = 6,

    lstm = False,
    transformer = True,
    batch_norm = True,
    encodec_ratios = [8,8,4,2],
    
    firing_rate_threshold = 4, # Nr_neurons * \nu_t
    spike_function = "free",
    in_channels = 1,

    lstm_hidden_size = 128,

    transformer_hidden_dim = 1024,
    transformer_output_dim = 128,
    transformer_internal_dim = 256,
    transformer_nhead = 4,
    transformer_nlayers = 3,

    binary_quantizer_lr_internal = 0.3

  ),

  dataset_params = AttrDict(
    dataset_folder="/lcncluster/datasets/maestro",
    sample_rate = 22050, 
    sequence_length = 2**17, #number of audio samples loaded by data loader
    download=True,
    split="train",
    midi_sampling_rate=100,
    data_limit = None,
    clip_limit = None,
  )
)