from parser import dotdict

model_t80 = dotdict({
    "name":"spikes_transformer_80",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/spikes_transformer_80",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,

    "split":"validation",
    "clip_limit":None,
})

model_t128 = dotdict({
    "name":"spikes_transformer_128",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/encodec_transformer",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":128,
    "transformer": True,

    "split":"validation",
    "clip_limit":None,
})

model_q = dotdict({
    "name": "quant_8x10",
    "encoder": ["q_encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/quant_8x10",
    "data_dirs":["maestro"],
    "batch_size":32,

    "split":"validation",
    "clip_limit":None,
})

model_bad = dotdict({
    "name":"bad",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/spikes_transformer_80/weights-10000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,

    "split":"validation",
    "clip_limit":None,
})

model_mel = dotdict({
    "name":"mel",
    "encoder":["mel"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/mel_default",
    "data_dirs":["maestro"],
    "batch_size":32,

    "split":"validation",
    "clip_limit":None,
})