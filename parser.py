import argparse
import torch
import socket


from params import params

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_parameters(parser):
# Model setup parameters
    parser.add_argument('encoder', nargs='+', help='type of encoder: [] + dimension')
    parser.add_argument('model_dir', help='directory in which to store model checkpoints and training logs')
    parser.add_argument('data_dirs', nargs='+',default = "maestro", help='space separated list of directories from which to read .wav files for training')
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
    parser.add_argument('--sequence_time', default=params.dataset_params.sequence_time, type = float,help='number of seconds of audio in each of the samples from the dataset')
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
    parser.add_argument('--encodec_ratios', nargs='+',default = params.encoder_params.encodec_ratios, type = int,help='Encodec SEANet encoder downsampling factors')

    parser.add_argument('--firing_rate_threshold', default=params.encoder_params.firing_rate_threshold, type = float,help='firing rate threshold in relu loss')
    parser.add_argument('--in_channels', default=params.encoder_params.in_channels, type=int, help='input audio channels')

    parser.add_argument('--encodec_dim', default=params.encoder_params.encodec_dim, type=int, help='Default encodec encoder output dim')
    parser.add_argument('--lstm_hidden_size', default=params.encoder_params.lstm_hidden_size, type=int, help='Default lstm hidden size')
    

def make_parser():
    parser = argparse.ArgumentParser(description='train (or resume training) a SpikingMusic model')
    parse_parameters(parser)
    return parser