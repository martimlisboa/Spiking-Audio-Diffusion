import numpy as np
import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import MultiMelSpectrogram

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import torchaudio as T
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio,scale_invariant_signal_distortion_ratio

from dataset_utils import from_maestro
import time

from model import SpikingAudioDiffusion

from models import compN_bps,compT_bps,clist_bps



verbose = False

#Training Functions
def train(args): #Train on 1 GPU
  if args.data_dirs[0] == 'maestro':
    dataloader = from_maestro(args)
  else:
    raise ValueError('NO DATASET.')

  #Model
  model = SpikingAudioDiffusion(args).cuda()

  #print(model)
  _train_impl(0, model, dataloader, args)


def train_distributed(replica_id, replica_count, port, args): #Train on multiple GPU
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.data_dirs[0] == 'maestro':
    dataloader = from_maestro(args,is_distributed= True)
  else:
    raise ValueError('NO DATASET.')

  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = SpikingAudioDiffusion(args).cuda()

  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataloader, args)


def _train_impl(replica_id, model, dataloader, args):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,betas = tuple(args.betas),eps = args.eps, weight_decay =args.w_decay)

  learner = SpikingAudioDiffusionLearner(args.model_dir, model, dataloader, opt, args, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  _restore = learner.restore_from_checkpoint()
  print(f"Restoring from checkpoint: {_restore}")
  learner.train(max_steps=args.max_steps)




class SpikingAudioDiffusionLearner: #Object to handle the training loop and the saving of data
  def __init__(self, model_dir, model, dataloader, optimizer, args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.args = args
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True
    self.summary_writer = None
    win_lengths = [2**i for i in range(5,12)]
    mel_bins = [5*2**i for i in range(7)]

    self.multi_scale_mel = MultiMelSpectrogram(args.sample_rate,win_lengths,mel_bins).to(self.model.get_device())




  def train(self, max_steps=None):
    self.model.train()
    device = next(self.model.parameters()).device
    #Start Time:
    t0 = time.time()
    while True:
      for features in tqdm(self.dataloader, desc=f'Epoch {self.step // len(self.dataloader)}') if self.is_master else self.dataloader:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        
        loss,rep_loss,beta = self.train_step(features)
        
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:

          if self.step % self.args.save_step == 0:
            self._write_summary(self.step, loss,rep_loss,beta)

          if self.step % self.args.test_step == 0:
            self.model.eval()
            self._test(self.step, features)
            self.model.train()

          if self.step % self.args.save_model_step  == 0 or self.step == 10000:
            self.save_to_checkpoint()
            t = (time.time() - t0)
            print(f"Saved to checkpoint: {self.step} steps: ")
            print(f"Training Time:  {t/3600} hours")   
            self.model.eval()
            self.encode_save_decode(input_path=self.args.wav_input_path, output_path=self.args.wav_output_path)         
            self.model.train()
        self.step += 1


  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    
    with self.autocast:
      if self.args.encoder[0] in ["encodec","q_encodec","mel","vocoder","rec_encodec"]:
        loss, info = self.model(audio)
      elif self.args.encoder[0] in ["mu_encodec"]:
        # random level of sparsity instantiated here
        mu = torch.randint(self.args.nr_mu_embeddings,(self.args.batch_size,), device = self.model.get_device())
        
        loss, info = self.model(audio,mu = mu)

    if self.args.encoder[0] in ["encodec","q_encodec","mu_encodec","rec_encodec"]:
      rep_loss = info["rep_loss"]
      beta = 1
      if self.args.annealing:
        beta = geometric(self.step,self.args.r_value,self.args.log_slope,self.args.goal_beta,self.args.warm_up_steps,self.args.annealing_steps)
      loss = loss + beta*rep_loss
    else:
      rep_loss = 0
      beta = 0

    #Backprop
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()

    return loss,rep_loss,beta

 #Functions to handle Checkpoints
  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
        'scaler': self.scaler.state_dict(),
    }
  
  def load_state_dict(self, state_dict):
    #This code is to help patch different versions of the integrate_context:
    keys = list(state_dict["model"].keys())
    for key in keys:
        
        prefix = "autoencoder.encoder"
        prefixes = [f"{prefix}.{module}" for module in ["transformer","conformer","lstm"]]
  
        for p in prefixes:
            if key.startswith(p):
                #print(key)
                suffix = key.removeprefix(prefix)
                #print(suffix)
                newkey = f"{prefix}.integrate_context{suffix}"
                #print(newkey)
                # print('\n')
                state_dict['model'][newkey] = state_dict["model"].pop(key)


    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']
    print(f"Successfully loaded from checkpoint {self.step}")
  
  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      try:
        os.symlink(save_basename, link_name)
      except OSError:
        print(f"Link {save_basename} to  {link_name} failed at step {self.step}")
        pass
  
  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

# Writing the summary
  def _write_summary(self, step, loss,rep_loss,beta):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/rep_loss', rep_loss, step)
    writer.add_scalar('train/rec_loss', loss-beta*rep_loss, step)
    writer.add_scalar('train/beta', beta, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer
  


  def count_bps(self,bottleneck_output): #Returns bits per second
    B,N,T = bottleneck_output.shape
    S = bottleneck_output.sum(dim = (1,2)) 
    N,T = torch.tensor(N,dtype = torch.float,device = self.model.get_device()),torch.tensor(T,dtype=torch.float,device = self.model.get_device())

    clist = clist_bps(N,T,S)/T
    compN = compN_bps(N,T,S)/T
    compT = compT_bps(N,T,S)/T

    bpf,_ = torch.min(torch.stack([clist,compN,compT],dim = 0),dim=0)
    bpf = torch.where(bpf < N,bpf,N)
    #print(f"Final Bit rate {bpf} bit/frame")

    return bpf*self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor #convert to bits/s
  
  def SISNR(self,audio_in,audio_out):
    if audio_in.shape[1] == 1:
        return scale_invariant_signal_distortion_ratio(audio_out,audio_in).squeeze(1) #Audio is of size [Batch, Channel, Time] so we squeeze the channel dim
    else:
        return scale_invariant_signal_distortion_ratio(audio_out,audio_in)
  
  @torch.no_grad()
  def _test(self, step, features):
    audio_in = features['audio']
    if self.args.encoder[0] in ["encodec","q_encodec","rec_encodec","mel","vocoder"]:
      audio_out,latent,info = self.model.autoencode(audio_in,num_steps = 100,show_progress=False)
    elif self.args.encoder[0] in ["mu_encodec"]:
      mu = torch.randint(self.args.nr_mu_embeddings,(audio_in.shape[0],), device = self.model.get_device())
      audio_out,latent, info = self.model.autoencode(audio_in,mu=mu,num_steps = 100,show_progress=False)


    sisnr = self.SISNR(audio_in,audio_out).mean()
    msMAE = self.multi_scale_mel.loss(audio_in,audio_out).mean()
    msSISNR = self.multi_scale_mel.sisnr_loss(audio_in,audio_out).mean()

    if self.args.encoder[0] in ["encodec","mu_encodec","rec_encodec"]:
      bps = self.count_bps(info["spikes"]).mean()
    elif self.args.encoder[0] in ["q_encodec"]:
      n_q = self.model.autoencoder.encoder.quantizer.n_q
      bpf_per_quantizer = math.ceil(math.log2(self.model.autoencoder.encoder.quantizer.bins))
      conv_bps = self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor
      bps = n_q * bpf_per_quantizer * conv_bps * torch.ones_like(sisnr)

    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('train/sisnr', sisnr, step)
    writer.add_scalar('train/bps', bps, step)
    writer.add_scalar('train/msMAE', msMAE, step)
    writer.add_scalar('train/msSISNR', msSISNR, step)


    writer.flush()
    self.summary_writer = writer

  @torch.no_grad()
  def encode_save_decode(self,input_path,output_path):
    #Get the list of audios to run the tests on
    files = os.listdir(input_path) 
    audios = []
    audio_names = []
    model_name = os.path.basename(os.path.normpath(self.model_dir))
    for name in files:
        audio_names.append(name[:-4])
        audio_path =input_path+name
        audio, sr = T.load(audio_path)
        audios.append(audio)

    audios = torch.stack(audios,dim = 0).to(self.model.get_device())
    print(f"Audios Shape: {audios.shape}")
    #Encode
    t0 = time.time()
    print("\n")
    if self.args.encoder[0] in ["mel","vocoder"]:
        latent,_ = self.model.encode(audios) # Encode
    elif self.args.encoder[0] in ["encodec"]:
        latent, info = self.model.encode(audios) # Encode
        rep = info["spikes"]
        print(f"spikes shape: {rep.shape}")
    elif self.args.encoder[0] in ["mu_encodec"]:
        mu = torch.randint(self.args.nr_mu_embeddings,(audios.shape[0],), device = self.model.get_device())
        latent, info = self.model.encode(audios,mu = mu) # Encode
        rep = info["spikes"]
        print(f"spikes shape: {rep.shape}")

    elif self.args.encoder[0] in ["q_encodec"]:
        latent, info = self.model.encode(audios) # Encode
        rep = torch.permute(info['codes'],[1,0,2])
        print(f"codes shape: {rep.shape}")
    
    dt = time.time() - t0
    print(f"Encoding Time: {dt}\n")

    #Decode
    t0 = time.time()
    if self.args.encoder[0] in ["mel","vocoder","mu_encodec","q_encodec","encodec"]:
      audios_inf = self.model.decode(latent, num_steps=100) # Decode by sampling diffusion model conditioning on latent
    
    elif self.args.encoder[0] in ["rec_encodec"]:
      audios_inf, latent, info = self.model.autoencode(audios,num_steps = 100)
      rep = info["spikes"]
      print(f"spikes shape: {rep.shape}")
      
    print(f"latent shape: {latent.shape}")
    print(f"output audios shape: {audios_inf.shape}")
    dt = time.time() - t0
    print(f"Decoding Time: {dt}")

    #Compute SI-SNR
    sisnr = self.SISNR(audios,audios_inf)
    print(f"SI-SNR Rates: {sisnr}")

    #Compute Bit Rates
    if self.args.encoder[0] in ["encodec","mu_encodec","rec_encodec"]:
        bps = self.count_bps(info["spikes"])
        print(f"Bit Rates: {bps}")
    elif self.args.encoder[0] == "q_encodec":
        n_q = self.model.autoencoder.encoder.quantizer.n_q
        bpf_per_quantizer = math.ceil(math.log2(self.model.autoencoder.encoder.quantizer.bins))
        conv_bps = self.model.args.sample_rate/self.model.autoencoder.encoder.downsample_factor
        bps = n_q * bpf_per_quantizer * conv_bps * torch.ones_like(sisnr)
    else: 
      bps = 44100

    out_dir = f"{output_path}/{model_name}"
    os.makedirs(out_dir,exist_ok=True)

    archive_path = f"{out_dir}/archive.npz"
    with open(archive_path,'wb') as f:
      np.savez(f,rep = rep.detach().cpu().numpy(), bps = bps.detach().cpu().numpy(), sisnr = sisnr.detach().cpu().numpy())
     
    for i,name in enumerate(audio_names):
      output_audio_filename = f"{out_dir}/{name}_{model_name}.wav"
      T.save(output_audio_filename,audios_inf[i].cpu(),sample_rate = sr) #save output audio

    
    fig, ax = plt.subplots(figsize=(16,9))
    for i,l in enumerate(latent.detach().cpu()):
        im = ax.imshow(l,aspect='auto')
        #fig.colorbar(im, ax=ax)
        fig.tight_layout()
        filename = f"{out_dir}/{audio_names[i]}_{model_name}_latent.png"
        plt.savefig(filename)
        plt.cla()
    plt.close(fig = fig)

    if self.args.encoder[0] in ["encodec","mu_encodec","rec_encodec"]:
        fig, ax = plt.subplots(figsize=(16,9))
        for i,z in enumerate(rep.detach().cpu()):
            ax.imshow(z,cmap="Greys",interpolation="none",aspect='auto')
            fig.tight_layout()
            filename = f"{out_dir}/{audio_names[i]}_{model_name}_spikes.png"
            plt.savefig(filename)
            plt.cla()
        plt.close(fig = fig)


    elif self.args.encoder[0] in ["q_encodec"]:
        fig, ax = plt.subplots(figsize=(16,9))
        for i,z in enumerate(rep.detach().cpu()):
            ax.imshow(z,interpolation="none",aspect='auto')
            fig.tight_layout()
            filename = f"{out_dir}/{audio_names[i]}_{model_name}_codes.png"
            plt.savefig(filename)
            plt.cla()
        plt.close(fig = fig)
        


def _nested_map(struct, map_fn):
    #function to handle mapping elements of struct to map_fn in a nested way

  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

#Geometric annealing schedule
def geometric(step, r_value, log_slope, goal_beta, warm_up_steps, annealing_steps):
    x = (step-warm_up_steps)/annealing_steps
    beta =  goal_beta* (r_value**(x*log_slope) -1)/(r_value**log_slope - 1)
    if beta > goal_beta:
      return goal_beta
    elif beta > 0:
      return beta
    else:
      return 0

if __name__ == '__main__':
  dataloader = from_maestro(args)