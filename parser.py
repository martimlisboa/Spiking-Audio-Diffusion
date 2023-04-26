import argparse
import torch
import socket


from params import params
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def override_args_from_dotdict(dotdict_args):
    args = make_scaffold_parser().parse_args() #Create a scaffold parser with all the default arguments
    for k,v in dotdict_args.items():
        setattr(args,k,v) #Add the compulsory arguments and modify whatever is not set         
                          #to default according to dotdict_args  
    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def parse_opt_parameters(parser):
    parser.add_argument('--fp16', action='store_true', default=False,help='use 16-bit floating point operations for training')
    parser.add_argument('--rep_loss_type', nargs='+', default= ["none"], help='encoder loss function')
    parser.add_argument('--annealing',action='store_true',default = False,help="Anneal the representation loss coefficient")
    parser.add_argument('--goal_beta',default = params.goal_beta, type = float, help="Final value of representation loss coefficient")
    #parser.add_argument('--annealing_type', nargs ='+', default=["constant"],help='annealing type')
    #parser.add_argument('--multitask', action='store_true', default=False,help='multitasking module')
    #parser.add_argument('--context', action='store_true', default=False, help='parallel context bottleneck module')

# Training Parameters
    parser.add_argument('--max_steps', default=None, type=int,help='maximum number of training steps')
    parser.add_argument('--batch_size', default=params.batch_size, type=int,help='batch size')
    parser.add_argument('--learning_rate', default=params.learning_rate, type = float,help='learning rate')
    parser.add_argument('--betas', default=params.betas, nargs = '+', type = float,help='AdamW betas')
    parser.add_argument('--eps', default=params.eps, type = float,help='AdamW epsilon')
    parser.add_argument('--w_decay', default=params.w_decay, type = float,help='AdamW weight decay')
    parser.add_argument('--max_grad_norm', default=params.max_grad_norm, type = float,help='maximum allowed gradient norm')
    parser.add_argument('--save_step', default=params.save_step, type=int,help='steps to write summary')
    parser.add_argument('--save_model_step', default=params.save_model_step, type=int,help='steps to save model to checkpoint')

# Dataset Parameters
    parser.add_argument('--dataset_folder',default = params.dataset_params.dataset_folder)
    parser.add_argument('--sequence_length', default=params.dataset_params.sequence_length, type = int,help='number of seconds of audio in each of the samples from the dataset')
    parser.add_argument('--clip_limit', default=params.dataset_params.clip_limit, type = int,help='number clips to pass tot he model')
    parser.add_argument('--data_limit', default=params.dataset_params.data_limit, type = int,help='number of songs to clip from')
    parser.add_argument('--sample_rate', default=params.dataset_params.sample_rate, type = int,help='midi sampling rate')
    parser.add_argument('--midi_sampling_rate', default=params.dataset_params.midi_sampling_rate, type = int,help='midi sampling rate')
    parser.add_argument('--download', action='store_true', default=params.dataset_params.download, help='download the dataset?')
    parser.add_argument('--split', default='train', help='dataset split to work with, ["train","test","validation"]; "train" by default')
    


# Encoder Parameters
    parser.add_argument('--bottleneck_dim', default=params.encoder_params.bottleneck_dim, type=int, help='Default bottleneck dim')
    parser.add_argument('--inject_depth', default=params.encoder_params.inject_depth, type=int, help='Default inject depth')
    parser.add_argument('--lstm', action='store_true', default=params.encoder_params.lstm,help='integrate context with lstm after the spikes')
    parser.add_argument('--transformer', action='store_true', default=params.encoder_params.transformer,help='integrate context with Transformer after the spikes')
    parser.add_argument('--batch_norm', action='store_true', default=params.encoder_params.batch_norm,help='batch norm before spiking')    
    parser.add_argument('--encodec_ratios', nargs='+',default = params.encoder_params.encodec_ratios, type = int,help='Encodec SEANet encoder downsampling factors')

    parser.add_argument('--firing_rate_threshold', default=params.encoder_params.firing_rate_threshold, type = float,help='firing rate threshold in relu loss')
    parser.add_argument('--spike_function', default= params.encoder_params.spike_function, help='alternative spiking functions')
    parser.add_argument('--in_channels', default=params.encoder_params.in_channels, type=int, help='input audio channels')

    parser.add_argument('--encodec_dim', default=params.encoder_params.encodec_dim, type=int, help='Default encodec encoder output dim')
    parser.add_argument('--lstm_hidden_size', default=params.encoder_params.lstm_hidden_size, type=int, help='Default lstm hidden size')
    
    parser.add_argument('--transformer_hidden_dim', default=params.encoder_params.transformer_hidden_dim, type=int, help='Transformer hidden dimension')
    parser.add_argument('--transformer_output_dim', default=params.encoder_params.transformer_output_dim, type=int, help='Transformer output dimension')
    parser.add_argument('--transformer_internal_dim', default=params.encoder_params.transformer_internal_dim, type=int, help='Transformer internal dimension')
    parser.add_argument('--transformer_nhead', default=params.encoder_params.transformer_nhead, type=int, help='Transformer nr of heads')
    parser.add_argument('--transformer_nlayers', default=params.encoder_params.transformer_nlayers, type=int, help='Transformer number of layers')


    parser.add_argument('--binary_quantizer_lr_internal', default=params.encoder_params.binary_quantizer_lr_internal, type=float, help='Internal Learning Rate of Binary Quantizer')



def parse_parameters(parser):
# Model setup parameters
    parser.add_argument('encoder', nargs='+', help='type of encoder: [] + dimension')
    parser.add_argument('model_dir', help='directory in which to store model checkpoints and training logs')
    parser.add_argument('data_dirs', nargs='+',default = "maestro", help='space separated list of directories from which to read .wav files for training')
    parse_opt_parameters(parser)

def make_parser():
    parser = argparse.ArgumentParser(description='train (or resume training) a SpikingMusic model')
    parse_parameters(parser)
    return parser

def make_scaffold_parser():
    parser = argparse.ArgumentParser(description = "Parses only the optional arguments, set to default")
    parse_opt_parameters(parser)
    return parser