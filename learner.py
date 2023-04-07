import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


from dataset_utils import from_maestro

import time

from model import SpikingAudioDiffusion

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
  learner.restore_from_checkpoint()
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



  def train(self, max_steps=None):
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
          if self.step % self.args.save_model_step  == 0 or self.step == 10000:
            self.save_to_checkpoint()
            t = (time.time() - t0)
            print(f"Saved to checkpoint: {self.step} steps: ")
            print(f"Training Time:  {t/3600} hours")            
        self.step += 1


  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    
    with self.autocast:
      loss, info = self.model(audio)

    rep_loss = info["rep_loss"]
    beta = 1
    if self.args.annealing:
      beta = geometric(self.step,10,4,self.args.goal_beta,10000,40000)

    loss = loss + beta*rep_loss
    
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
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'scaler': self.scaler.state_dict(),
    }
  
  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']
  
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