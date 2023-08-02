from parser import dotdict




model_80free = dotdict({
    "name":"FREE",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80free",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_80brute = dotdict({
    "name":"80brute",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80brute",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["brute"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_80brute50 = dotdict({
    "name":"80brute50",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/annealing/A0.01slow",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["brute"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_sparse = dotdict({
    "name":"SPARSE",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sparse_conv",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 3,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["brute"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_RVQ8x10 = dotdict({
    "name": "RVQ8x10",
    "encoder": ["q_encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/RVQs/RVQ8x10",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim": 128,

    "split":"validation",
    "clip_limit":None,    
})

model_bad = dotdict({
    "name": "bad",
    "encoder": ["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/trough/80full/weights-10000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"full",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,
  
})


model_RVQ_50 = dotdict({
    "name": "RVQ_50",
    "encoder": ["q_encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/RVQs/RVQ8x10/weights-50000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim": 128,

    "split":"validation",
    "clip_limit":None,    
})



model_8842i6 = dotdict({
    "name":"8842i6",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sample_rate/8842i6_conv",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 3,
    "encodec_ratios":[8,8,4,2],
    "inject_depth":6,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["brute"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_8842i6_free = dotdict({
    "name":"8842i6_free",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sample_rate/8842i6_free",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 3,
    "encodec_ratios":[8,8,4,2],
    "inject_depth":6,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_8841i5 = dotdict({
    "name":"8841i5",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sample_rate/8841i5_B0_conv",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 5,
    "encodec_ratios":[8,8,4,1],
    "inject_depth":5,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["brute"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_8841i5_free = dotdict({
    "name":"8841i5_free",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sample_rate/8841i5_free",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 5,
    "encodec_ratios":[8,8,4,1],
    "inject_depth":5,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})


model_8822i5 = dotdict({
    "name":"8822i5",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sample_rate/8822i5_B0_conv",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 5,
    "encodec_ratios":[8,8,2,2],
    "inject_depth":5,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["brute"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_8822i5_free = dotdict({
    "name":"8822i5_free",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sample_rate/8822i5_free",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 5,
    "encodec_ratios":[8,8,2,2],
    "inject_depth":5,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_8821i4 = dotdict({
    "name":"8821i4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sample_rate/8821i4_B0_conv",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 9,
    "encodec_ratios":[8,8,2,1],
    "inject_depth":4,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["brute"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_8821i4_free = dotdict({
    "name":"8821i4_free",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/sample_rate/8821i4_free",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "transformer_conv": True,
    "transformer_conv_kernel": 9,
    "encodec_ratios":[8,8,2,1],
    "inject_depth":4,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})


model_mel = dotdict({
    "name":"MEL",
    "encoder":["mel"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/mel_default",
    "data_dirs":["maestro"],
    "batch_size":32,

    "split":"validation",
    "clip_limit":None,   
})




'''
model_80free = dotdict({
    "name":"80free",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80free/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})


model_80trough4 = dotdict({
    "name":"80trough4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80trough4/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})
model_80trough1A4 = dotdict({
    "name":"80trough0.1A4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80trough0.1A4/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
})

model_80trough2A4 = dotdict({
    "name":"80trough0.01A4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80trough0.01A4/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
})

model_80reset4 = dotdict({
    "name":"80reset4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80reset4/weights-200000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"reset",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
})

model_80adapt4 = dotdict({
    "name":"80adapt4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80adapt4/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"adapt",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
})

model_80refractory4 = dotdict({
    "name":"80refractory4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80refractory4/weights-200000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"refractory",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
})


model_80full4 = dotdict({
    "name":"80full4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80full4/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"full",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,
})

model_80full = dotdict({
    "name":"80full",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80full/weights-200000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"full",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
})

model_256trough4 = dotdict({
    "name":"256troughA4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/256troughA4/weights-200000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":256,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
})

model_1024troughA4 = dotdict({
    "name":"1024troughA4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/1024troughA4/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":1024,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})


model_4096troughA4 = dotdict({
    "name":"4096troughA4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/4096troughA4/weights-200000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":4096,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
})

model_8841i5 = dotdict({
    "name":"8841i5",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/8841i5/weights.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 2,
    "inject_depth": 5,
    "encodec_ratios": [8,8,4,1],

    "split":"validation",
    "clip_limit":None,
})

model_8822i5 = dotdict({
    "name":"8822i5",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/8822i5/weights.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 2,
    "inject_depth": 5,
    "encodec_ratios": [8,8,2,2],

    "split":"validation",
    "clip_limit":None,
})

model_8821i4 = dotdict({
    "name":"8821i4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/8821i4",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 1,
    "inject_depth": 4,
    "encodec_ratios": [8,8,2,1],

    "split":"validation",
    "clip_limit":None,
})

# Frying pan models:
model_128frying_pan4 = dotdict({
    "name":"128frying_pan4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/128frying_pan4/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":128,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["frying_pan"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,
})

# Frying pan models:
model_80fpfull4 = dotdict({
    "name":"80fpfull4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80fpfull4/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"full",
    "rep_loss_type": ["frying_pan"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,
})




#Quantizer Model

model_q = dotdict({
    "name": "quant_8x10",
    "encoder": ["q_encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/q_encodec/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim": 128,
    "split":"validation",
    "clip_limit":None,
})


#Mel Spectrogram Model
model_mel = dotdict({
    "name":"mel",
    "encoder":["mel"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/mel_default/weights-500000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,

    "split":"validation",
    "clip_limit":None,
})


model_80free_midi = dotdict({
    "name":"80free",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80free",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_80trough4_midi = dotdict({
    "name":"80trough4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80trough4",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})

model_80full4_midi = dotdict({
    "name":"80full4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80full4",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"full",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,
})

model_1024troughA4_midi = dotdict({
    "name":"1024troughA4",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/1024troughA4",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":1024,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,

})



#Models for EXP:

model_80freeEXP = dotdict({
    "name":"FREE",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80free",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["none"],

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,
})

model_QEXP = dotdict({
    "name": "RVQ",
    "encoder": ["q_encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/quant_8x10",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim": 128,
    "split":"validation",
    "clip_limit":None,    
})

model_80trough4EXP = dotdict({
    "name":"SPARSE",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/80trough4",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,
    "batch_norm": True,
    "spike_function":"free",
    "rep_loss_type": ["trough"],
    "firing_rate_threshold": 4,

    "split":"validation",
    "clip_limit":None,
    "midi_sampling_rate":43,    
})

model_Q50EXP = dotdict({
    "name": "RVQ_50",
    "encoder": ["q_encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/quant_8x10/weights-50000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim": 128,
    "split":"validation",
    "clip_limit":None,      
})

model_melEXP = dotdict({
    "name":"MEL",
    "encoder":["mel"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/mel_default",
    "data_dirs":["maestro"],
    "batch_size":32,

    "split":"validation",
    "clip_limit":None,   
})
'''