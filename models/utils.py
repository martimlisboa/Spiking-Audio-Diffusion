import torch
import torch.nn as nn
import numpy as np


def to_numpy(data):
    if np.isscalar(data):
        return data

    if data is None:
        return None

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        return data

    if isinstance(data, tuple):
        return tuple([to_numpy(datum) for datum in data])

    if isinstance(data, list):
        return [to_numpy(datum) for datum in data]

    if isinstance(data, dict):
        return dict([(k, to_numpy(v)) for k,v in data.items()])

    raise NotImplementedError("to numpy not implemented for data: {}".format(type(data)))

def get_device(model):
    return next(model.parameters()).device

def fuse_batch_and_time_dimensions(tensor):
    ''' Fused batch and time axis as a single one. It expects a batch major tensor (time is the second axis).'''
    return torch.cat(torch.unbind(tensor, dim=0), dim=0)


def recover_batch_and_time_dimensions(tensor, n_batch, n_time):
    ''' Recover the batch time dimension for a fused tensor. It produces a batch major tensor (time is the second axis).'''
    return torch.stack(torch.split(tensor, n_time, dim=0), dim=0)
