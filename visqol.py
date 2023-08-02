import torch
import torchaudio as T
import torchaudio.functional as F
import torchaudio.transforms as TT
import os
import shutil
import numpy as np
import pandas as pd

from dataset_utils import MaestroDataset
from parser import dotdict, override_args_from_dotdict
from model_list import model_80brute50,model_80free,model_RVQ8x10,model_80brute,model_bad,model_mel,model_80trough4,model_RVQ_50
from analysis import Inferer

#model_list = [model_80brute50,model_80free,model_RVQ8x10,model_80brute,model_bad,model_mel,model_80trough4,model_RVQ_50]

test_model = dotdict({
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




def make_base(nr_audios,verbose = False):
    print("Making base")
    visqol_sr = 48000
    args = override_args_from_dotdict(test_model)
    dataset = MaestroDataset(args)
    base_dir = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/visqol"
    base_dir48 = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/visqol48"
    os.makedirs(base_dir,exist_ok=True)
    os.makedirs(base_dir48,exist_ok=True)

    N = len(dataset.index_table)

    sample_list = np.random.choice(range(N),nr_audios,replace = False)
    print(sample_list)

    for i,sample_idx in enumerate(sample_list):
        if verbose:
            print(f"{sample_idx} - Song: {dataset.get_title(sample_idx)}")
        item = dataset.__getitem__(sample_idx)
        audio = item["audio"] 
        audio = torch.clamp(torch.Tensor(audio),min = -1., max = 1.)
        audio = audio.unsqueeze(0)
        #print(audio.shape)
        name = "audio_"+str(i+1)

        output_audio_filename = f"{base_dir}/{name}.wav";
        T.save(output_audio_filename,audio.cpu(),sample_rate = dataset.sampling_rate) #save output audio

        #Resample  
        audio_resampled = F.resample(audio.cpu(),dataset.sampling_rate,visqol_sr) 
        output_audio_filename = f"{base_dir48}/{name}.wav";
        T.save(output_audio_filename,audio_resampled,sample_rate = visqol_sr,
                   bits_per_sample = 16) #save output audio

    return

def make_temp(model_inf, tmp_dir, base_dir, batch_size):
    visqol_sr = 48000
    N = len(model_inf.dataloader.dataset.index_table) 
    sample_list = os.listdir(base_dir)
    nr_audios = len(sample_list)
    n_batch = nr_audios//batch_size

    sample_list = [[s for s in sample_list[k*batch_size:(k+1)*batch_size]] for k in range(n_batch)]

    print(f"Samples: {sample_list}")
    print(f"# audios: {nr_audios}")
    print(f"# batch : {n_batch}")
    

    os.makedirs(tmp_dir,exist_ok=True)


    for b,batch in enumerate(sample_list):
        print(f"batch {b}/{n_batch}")
        b_audios = []
        b_audio_names =[]
        sr0 = 22050 #just to define it before the batching loop
        #Batching
        for i,sample_name in enumerate(batch):
            #print(f"{sample_idx} - Song: {dataset.get_title(sample_idx)}")
            audio,sr0 = T.load(f"{base_dir}/{sample_name}")
            audio = torch.clamp(torch.Tensor(audio),min = -1., max = 1.)
            #audio = audio.unsqueeze(0)
            #print(audio.shape)
            #name = "audio_"+str(i + b*batch_size)
            b_audio_names.append(sample_name)
            b_audios.append(audio)

        b_audios = torch.stack(b_audios,dim = 0).to("cuda")
        #Autoencoding batch
        audios_inf, _, _ = model_inf.autoencode(b_audios,num_steps = 100)
        #Resampling batch
        audios_inf = F.resample(audios_inf,sr0,visqol_sr)
        #Saving audios one by one
        for i,audio in enumerate(audios_inf):
            output_audio_filename = f"{tmp_dir}/{b_audio_names[i]}";
            T.save(output_audio_filename,audio.cpu(),sample_rate = visqol_sr,
                   bits_per_sample = 16) #save output audio
       

        del b_audios
        del audios_inf
        del b_audio_names

    print(f"List tmp dir: {os.listdir(tmp_dir)}")

    return

def make_opus(opus_dir,base_dir, opus_bps):
    os.makedirs(opus_dir,exist_ok=True)

    sample_list = os.listdir(base_dir)
    nr_audios = len(sample_list)
    
    opus_audio_path = f"{opus_dir}/tmp.opus"


    for sample_name in sample_list:
        input_audio_path = f"{base_dir}/{sample_name}"
        output_audio_path = f"{opus_dir}/{sample_name}"

        os.system(f"opusenc --quiet --bitrate {opus_bps} {input_audio_path} {opus_audio_path} --hard-cbr")
        os.system(f"opusdec --quiet {opus_audio_path} {output_audio_path}")

    
    os.remove(opus_audio_path)



def make_input_csv(name,base_dir,tmp_dir):
    base_list = os.listdir(base_dir)
    tmp_list = os.listdir(tmp_dir)
    data = pd.DataFrame({'reference':[f"{base_dir}/{name}" for name in base_list],'degraded':[f"{tmp_dir}/{name}" for name in tmp_list]})
    data.to_csv(name,index = False)

def compute_visqol(input_csv, results_csv):
    cwd = os.getcwd()
    os.chdir(r"/lcncluster/lisboa/ViSQOL/visqol")
    os.system(f"./bazel-bin/visqol --batch_input_csv {input_csv} --results_csv {results_csv}")
    os.chdir(f"{cwd}")

def clear_tmp(input_csv,tmp_dir):
    os.remove(input_csv)
    try:
        shutil.rmtree(tmp_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def ViSQOL(model,batch_size):
    tmp_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/tmp_visqol"
    base_path = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/visqol"
    base_path48 = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/visqol48"
    
    #Make tmp file with all the audios 16bit@48kHz ready for ViSQOL
    args = override_args_from_dotdict(model)
    inf = Inferer(args,device = torch.device('cuda'))
    make_temp(inf,tmp_path,base_path, batch_size = batch_size)

    #Input csv with pairs of references in base48 and degraded in tmp
    input_csv = "/lcncluster/lisboa/spikes_audio_diffusion/data/visqol.csv"
    make_input_csv(input_csv,base_path48,tmp_path)
    
    #make data dir if it doesnt exist
    os.makedirs(f"/lcncluster/lisboa/spikes_audio_diffusion/data/{model.name}",exist_ok=True)
    #path to csv where results are stored
    results_csv = f"/lcncluster/lisboa/spikes_audio_diffusion/data/{model.name}/visqol_{test_model.split}.csv"
    if os.path.exists(results_csv):
        os.remove(results_csv) #Remove the visqol.csv if there is already one in that dir
    #ViSQOL computation
    compute_visqol(input_csv = input_csv, results_csv=results_csv)
    #Clear the tmp stuff 
    clear_tmp(input_csv,tmp_path)

    return


def ViSQOL_opus():
    input_csv = "/lcncluster/lisboa/spikes_audio_diffusion/data/visqol.csv"
    base_path48 = "/lcncluster/lisboa/spikes_audio_diffusion/wav_inputs/visqol48"

    opus_dirs = ["opus6","opus12","opus24","opus64"]
    for opus_dir in opus_dirs:
        opus_path = f"/lcncluster/lisboa/spikes_audio_diffusion/wav_outputs/{opus_dir}"
        data_dir =  f"/lcncluster/lisboa/spikes_audio_diffusion/data/{opus_dir}"
        os.makedirs(data_dir,exist_ok=True)
        results_csv = f"/lcncluster/lisboa/spikes_audio_diffusion/data/{opus_dir}/visqol_{test_model.split}.csv"

        opus_bps = int(opus_dir[4:])
        print(f"{opus_dir} @ {opus_bps}Kbps")
        make_opus(opus_path,base_path48,opus_bps)
        make_input_csv(input_csv,base_path48,opus_path)
        compute_visqol(input_csv = input_csv, results_csv=results_csv)
        os.remove(input_csv)
    return


def resample_dir(dir,sampling_rate):
    for file in sorted(os.listdir(dir)):
        filename = os.fsdecode(file)
        a = filename.endswith(".wav")
        b = not filename.endswith(f'{sampling_rate}.wav')
        if a and b :
            new_filename = f'{filename[:-4]}_{sampling_rate}.wav'
            print(f'{filename}->{new_filename}')
            audio, sr = T.load(f'{dir}/{filename}')
            audio_resampled = F.resample(audio,sr,sampling_rate)
            out_file = f'{dir}/{new_filename}'
            print(f"Audio Shape: {audio.shape}, Audio R Shape {audio_resampled.shape}")
            T.save(out_file,audio_resampled,sample_rate = sampling_rate,
                   bits_per_sample = 16) #save output audio
            #_, srF = T.load(out_file)
            #print(srF)

def delete_with_sr(dir,sampling_rate):
    for file in sorted(os.listdir(dir)):
        filename = os.fsdecode(file)
        b = filename.endswith(f'{sampling_rate}.wav')
        if b:
            print(f"Removing {filename}")
            file_path = f'{dir}/{filename}'
            os.remove(file_path)           

def main():

    #make_base(200,verbose=True)
    #for model in model_list:
    #    ViSQOL(model,batch_size=25)
    
    ViSQOL_opus()


if __name__ =='__main__':
    main()