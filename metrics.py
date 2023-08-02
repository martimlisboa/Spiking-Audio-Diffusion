from analysis import Inferer
from parser import dotdict,override_args_from_dotdict
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio,scale_invariant_signal_distortion_ratio

import numpy as np
import os
import shutil
from tqdm import tqdm
import torchaudio as T

import torch
import torch.nn as nn
from dataset_utils import from_maestro
from learner import _nested_map
from models import MultiMelSpectrogram
import argparse

#from model_list import model_80free, model_q, \
#                       model_80trough4,model_1024troughA4,model_4096troughA4,model_256trough4,model_mel, \
#                       model_80adapt4,model_80refractory4,model_80reset4,model_80full4, model_80trough1A4,model_80trough2A4,\
#                       model_8821i4,model_8822i5,model_8841i5, model_128frying_pan4,model_80fpfull4, \
#                       model_80freeEXP,model_Q50EXP,model_QEXP,model_80trough4EXP,model_melEXP

from model_list import model_RVQ8x10,model_80brute50,model_mel,model_80free,model_80brute,model_RVQ_50, \
                       model_8821i4,model_8822i5,model_8841i5,model_8842i6,model_sparse, \
                       model_8842i6_free, model_8841i5_free, model_8822i5_free, model_8821i4_free
model_test = dotdict({
    "name":"spikes_transformer_80",
    "encoder":["encodec"],
    "model_dir":"/lcncluster/lisboa/runs/audio_diffusion/spikes_transformer_80/weights-300000.pt",
    "data_dirs":["maestro"],
    "batch_size":32,
    "bottleneck_dim":80,
    "transformer": True,

    "split":"validation",
})

#models = [model_80free, model_q, \
#          model_80trough4,model_1024troughA4,model_4096troughA4,model_256trough4,model_mel, \
#          model_80trough1A4,model_80trough2A4,\
#          model_80adapt4,model_80refractory4,model_80reset4,model_80full4, model_80full, \
#          model_8821i4,model_8822i5,model_8841i5]

models = [model_RVQ8x10,model_80brute50,model_mel,model_80free,model_80brute,model_RVQ_50, \
          model_8821i4,model_8822i5,model_8841i5,model_8842i6,model_sparse, \
          model_8842i6_free, model_8841i5_free, model_8822i5_free, model_8821i4_free]
opus_models = ["opus6","opus12"]

def make_dataloader(args):
    if args.data_dirs[0] == 'maestro':
        dataloader = from_maestro(args)
    else:
        raise ValueError('NO DATASET.')
    return dataloader


def rm_dir(dir_path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def opus_autoencode_batch(audio_in,sample_rate,opus_bps,tmp_opus_dir,tmp_opus_wav_dir):
    audio_out = []
    for i,audio in enumerate(audio_in):
        input_audio_path = f"{tmp_opus_dir}/audio_{i}.wav"
        opus_audio_path =  f"{tmp_opus_dir}/audio_{i}.opus"
        output_audio_path =  f"{tmp_opus_wav_dir}/audio_{i}.wav"
        #put the audios in cpu:
        T.save(input_audio_path,audio.cpu(),sample_rate = sample_rate) #save output audio
        #encode to opus:
        #print("Enc")
        os.system(f"opusenc --quiet --bitrate {opus_bps} {input_audio_path} {opus_audio_path} --hard-cbr")
        #decode the opus to wav
        #print("Dec")
        os.system(f"opusdec --quiet {opus_audio_path} {output_audio_path}")
        #print("Done")
        
        #output batching
        out,sr = T.load(output_audio_path)
        audio_out.append(out)

        # Remove the input wav and the opus file
        #os.remove(input_audio_path)
        #os.remove(opus_audio_path)
        #os.remove(output_audio_path)

    
    audio_out = torch.stack(audio_out,dim = 0).to("cuda")

    return audio_out
def SISNR(audio_in,audio_out):
    if audio_in.shape[1] == 1:
        return scale_invariant_signal_distortion_ratio(audio_out,audio_in).squeeze(1) #Audio is of size [Batch, Channel, Time] so we squeeze the channel dim
    else:
        return scale_invariant_signal_distortion_ratio(audio_out,audio_in)
    

def wavMSE(audio_in, audio_out):
    return (audio_in-audio_out).pow(2).mean(dim = (1,2))  

def inf_opus(data_dir,opus_bps):
    args = override_args_from_dotdict(model_test)
    dataloader = make_dataloader(args)
    tmp_opus_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/tmp_opus"
    tmp_opus_dir_path = f"{tmp_opus_path}/opus"
    tmp_wav_dir_path = f"{tmp_opus_path}/wav"

    os.makedirs(tmp_opus_path,exist_ok=True)

    os.makedirs(tmp_opus_dir_path,exist_ok=True)
    os.makedirs(tmp_wav_dir_path,exist_ok=True)
    os.makedirs(data_dir,exist_ok=True)



    #Batch in GPU to CPU
    #Batch -> Opus Batch
    #Opus Batch -> Wav/Opus batch
    # Wav/Opus in CPU->GPU
    # Infer MAE,SISNR,wavMSE
    #Repeat 

    device = torch.device('cuda')

    msMAE = torch.empty(0,device = device)
    msSISNR = torch.empty(0,device = device)
    wMSE = torch.empty(0,device = device)
    sisnr = torch.empty(0,device = device)
    win_lengths = [2**i for i in range(5,12)]
    mel_bins = [5*2**i for i in range(7)]

    multi_scale_mel = MultiMelSpectrogram(args.sample_rate,win_lengths,mel_bins).to(device)
    for features in tqdm(dataloader):  
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)            
        audio_in = features["audio"]
        audio_out = opus_autoencode_batch(audio_in, args.sample_rate, opus_bps,tmp_opus_dir_path,tmp_wav_dir_path)
        msMAE = torch.cat((msMAE,multi_scale_mel.loss(audio_in,audio_out)),dim = 0)
        msSISNR = torch.cat((msSISNR,multi_scale_mel.sisnr_loss(audio_in,audio_out)),dim = 0)

        wMSE = torch.cat((wMSE,wavMSE(audio_in,audio_out)),dim = 0)
        sisnr = torch.cat((sisnr,SISNR(audio_in,audio_out)),dim = 0)


    outfile = data_dir+'/metrics.npz'
    with open(outfile,'wb') as f:
        np.savez(f, multispecSISNR = msSISNR.cpu().numpy(),
                    multispecMAE = msMAE.cpu().numpy(),
                    wavMSE = wMSE.cpu().numpy(),
                    sisnr = sisnr.cpu().numpy(),
                    bps = opus_bps)  
        
        
    rm_dir(tmp_opus_dir_path)
    rm_dir(tmp_wav_dir_path)
    rm_dir(tmp_opus_path)


def inf_model(model):
    args = override_args_from_dotdict(model)
    out_path = "/lcncluster/lisboa/spikes_audio_diffusion/data/"+args.name
    os.makedirs(out_path,exist_ok=True)
    inf = Inferer(args,device = torch.device('cuda'))
    #Get the metrics using the inferer object
    data = inf.infer()


    multispecMAE = data["multispecMAE"]
    multispecSISNR = data["multispecSISNR"]
    wavMSE = data["wavMSE"]
    sisnr = data["sisnr"]
    bps = data["bps"]


    outfile = out_path+'/metrics.npz'
    with open(outfile,'wb') as f:
        np.savez(f,
                    multispecMAE = multispecMAE.cpu().numpy(),
                    multispecSISNR = multispecSISNR.cpu().numpy(),
                    wavMSE = wavMSE.cpu().numpy(),
                    sisnr = sisnr.cpu().numpy(),
                    bps = bps.cpu().numpy())  
    
    for k,v in data.items():
        print(f"{k}: {v.mean()} +- {v.std()}")

def main():
    _models = True
    _opus = False
    if _models:
        print("models")
        for model in models:
            inf_model(model)

    if _opus:
        for opus in opus_models:  
            data_dir = "/lcncluster/lisboa/spikes_audio_diffusion/data/"+opus
            opus_bps=int(opus[4:])
            print(f"infering {opus}")
            inf_opus(data_dir,opus_bps)
            
if __name__ == "__main__":
    main()

    '''    parser = argparse.ArgumentParser(description="Opus or Models")
    parser.add_argument('--opus', action='store_true', default=False,help='Opus Metrics')
    parser.add_argument('--models',action='store_true', default=False,help='Models Metrics')
    main_args = parser.parse_args()
    main(main_args)'''