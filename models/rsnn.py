import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import math
import torch.nn.functional as F
from torch.nn.functional import normalize
import torchinfo
import numpy as np
from torch.nn import init

from .utils import to_numpy


class SpikeFunction(nn.Module):
    def __init__(self, dampening_factor=0.3):
        super(SpikeFunction, self).__init__()
        self.dampening_factor = dampening_factor

    def forward(self, x, hidden=None):
        z_forward = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        if not x.requires_grad:
            return z_forward

        z_backward = torch.where(x > 0, - 0.5 * torch.square(x - 1), 0.5 * torch.square(x + 1))
        z_backward = torch.where(torch.abs(x) < 1, z_backward, torch.zeros_like(x))
        z_backward = self.dampening_factor * z_backward

        z = (z_forward - z_backward).detach() + z_backward
        return z


class FullSpikeFunction(nn.Module):
    #Implements a reset on the spikes
    def __init__(self,n_neurons,  reset_alpha = 0.9, thr_alpha=0.95, dampening_factor=0.3, refractory_period=5):
        super(FullSpikeFunction, self).__init__()
        self.dampening_factor = dampening_factor

        self.reset_alpha = reset_alpha
        self.thr_alpha = thr_alpha
        self.refractory_period = refractory_period
        self.n_neurons = n_neurons
        self.reset_gamma = nn.Parameter(data = torch.ones(self.n_neurons), requires_grad = True)
        self.thr_gamma = nn.Parameter(data = torch.ones(self.n_neurons), requires_grad = True)
        self.register_buffer("u0", torch.zeros(self.n_neurons))
        self.register_buffer("b0", torch.zeros(self.n_neurons))

        self.spike = SpikeFunction(dampening_factor = self.dampening_factor)

    def zero_state(self, n_batch, device):
        oo = torch.ones([n_batch,1], device=device)
        return oo * self.u0[None, :], oo * self.b0[None, :]

    def forward(self,x):
        #x in [Batch, Time, Neuron] format
        B,T,N = x.shape
        z_list=[]
        #Initialize z(t=0) and u0
        u,b = self.zero_state(B, x.device)
        q = torch.zeros([B,N], device=x.device, dtype=torch.int32)
        
        last_z = torch.zeros_like(x[:,0,:])

        al = self.reset_alpha
        bet =self.thr_alpha
        #Time Loop:
        for t in range(T):
            xt = x[:,t,:]
            reset = torch.mul(last_z,self.reset_gamma)
            u = al * u + (1- al) * xt - reset
            z = self.spike(u - b)
            z = torch.where(q == 0, z, torch.zeros_like(z))
            #q = torch.maximum(q + z.int() * self.refractory_period, torch.zeros_like(z))
            q = torch.maximum(torch.zeros_like(q),q-1)
            q = q + self.refractory_period*z.int() 
            b = bet*b + (1-bet)*torch.mul(z,self.thr_gamma)
            last_z = z
            z_list.append(z)

        if(self.training):
            mom = 0.95
            self.u0.data = mom * self.u0 + (1-mom) * u.mean(0)
            self.b0.data = mom * self.b0 + (1-mom) * b.mean(0)

        z = torch.stack(z_list,dim=1)
        return z


class ResetSpikeFunction(nn.Module):
    #Implements a reset on the spikes
    def __init__(self,n_neurons,  alpha = 0.9, dampening_factor=0.3):
        super(ResetSpikeFunction, self).__init__()
        self.dampening_factor = dampening_factor
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.reset_gamma = nn.Parameter(data = torch.ones(self.n_neurons), requires_grad = True)
        self.register_buffer("u0", torch.zeros(self.n_neurons))
        self.spike = SpikeFunction(dampening_factor = self.dampening_factor)

    def zero_state(self, n_batch, device):
        oo = torch.ones([n_batch,1], device=device)
        return oo * self.u0[None, :]

    def forward(self,x):
        #x in [Batch, Time, Neuron] format
        B,T,N = x.shape
        z_list=[]
        #Initialize z(t=0) and u0
        z_list.append(self.spike(x[:,0,:]))
        u = self.zero_state(B, x.device)
        al = self.alpha
        #Time Loop:
        for t in range(1,T):
            xt = x[:,t,:]
            reset = torch.mul(z_list[-1],self.reset_gamma)
            u = al * u + (1- al) * xt - reset
            z = self.spike(u)
            z_list.append(z)

        if(self.training):
            mom = 0.95
            self.u0.data = mom * self.u0 + (1-mom) * u.mean(0)

        z = torch.stack(z_list,dim=1)
        return z


class AdaptSpikeFunction(nn.Module):
    def __init__(self,n_neurons, alpha = 0.95, dampening_factor=0.3):
        super(AdaptSpikeFunction, self).__init__()
        self.dampening_factor = dampening_factor
        self.alpha = alpha
        self.n_neurons = n_neurons
        #Learnable gamma parameter
        self.reset_gamma = nn.Parameter(data = torch.ones(self.n_neurons), requires_grad = True)
        self.spike = SpikeFunction(dampening_factor = self.dampening_factor)
        self.register_buffer("b0", torch.zeros(self.n_neurons))
        self.count = 0

    def zero_state(self, n_batch, device):
        oo = torch.ones([n_batch,1], device=device)
        return oo * self.b0[None, :]
    
    def forward(self, x):
        B,T,N = x.shape
        z_list = []
        #Initialize z(t=0) b and u 
        b = self.zero_state(B, x.device)
        #b = torch.ones_like(x[:,0,:])
        z_list.append(self.spike(x[:,0,:]))

        #Time Loop:
        for t in range(1,T):
            b = self.alpha*b + (1-self.alpha)*torch.mul(z_list[-1],self.reset_gamma)
            z_list.append(self.spike(x[:,t,:]-b))

        #Check whether or not the model learns Gamma parameter or not
        if self.count%10000 == 0:
            print(f"Gamma mean: {self.reset_gamma.mean()}")
        self.count +=1

        if(self.training):
            mom = 0.95
            self.b0.data = mom * self.b0 + (1-mom) * b.mean(0)
        
        z = torch.stack(z_list,dim=1)
        
        return z


class RefractorySpikeFunction(nn.Module):
    def __init__(self,n_neurons, refractory_period = 5, dampening_factor=0.3):
        super(RefractorySpikeFunction, self).__init__()
        self.dampening_factor = dampening_factor
        self.n_neurons = n_neurons
        self.refractory_period = refractory_period
        #self.spike = SpikeFunction(dampening_factor = self.dampening_factor)

    def forward(self,x):
        B,T,N = x.shape
        ''' 
        #Initialize z list and q
        z_list = []
        z_list.append(self.spike(x[:,0,:]))
        q = self.refractory_period*z_list[0]


        # time loop
        for t in range(1,T):
            z_list.append(self.spike(x[:,t,:])) #Spike on input first
            cond = torch.logical_and(x[:,t,:] > 0, torch.logical_not(q)) 
            z_list[-1] = torch.where(cond, z_list[-1], torch.zeros_like(q)) # remove the spikes that are there during the refractory period
            q = torch.maximum(torch.zeros_like(q),q-1)
            q = q + self.refractory_period*z_list[-1] #Update the value of q

        z = torch.stack(z_list,dim=1)
        
        '''
        

        # time loop
        with torch.no_grad():
            z_forward=torch.zeros_like(x)
            #Initialize z(0) and q
            z_forward[:,0,:]=torch.where(x[:,0,:] > 0, torch.ones_like(x[:,0,:]), torch.zeros_like(x[:,0,:]))
            q = torch.zeros_like(z_forward[:,0,:])
            q = q + self.refractory_period*z_forward[:,0,:]

            for t in range(1,T):
                cond = torch.logical_and(x[:,t,:] > 0, q == 0)
                z_forward[:,t,:]=torch.where(cond, torch.ones_like(q), torch.zeros_like(q))
                q = torch.maximum(torch.zeros_like(q),q-1)
                q = q + self.refractory_period*z_forward[:,t,:]

        if not x.requires_grad:
            return z_forward

        z_backward = torch.where(x > 0, - 0.5 * torch.square(x - 1), 0.5 * torch.square(x + 1))
        z_backward = torch.where(torch.abs(x) < 1, z_backward, torch.zeros_like(x))
        z_backward = self.dampening_factor * z_backward

        z = (z_forward - z_backward).detach() + z_backward
        
        return z


def logit(prob):
    if not isinstance(prob, Tensor):
        prob = torch.tensor(prob)
    return torch.logit(prob, 1e-8)


def gumbel_binary_sample(u, temp_derivative, temp_noise, hard):
    xi = logit(torch.rand_like(u))
    l = (u - xi * temp_noise) * temp_derivative
    z_diff = torch.sigmoid(l)
    z = (z_diff > 0.5).float()

    if not hard:
        return z_diff, z_diff

    return z_diff + (z - z_diff).detach(), z_diff


class SpikeDefault(nn.Module):

    def __init__(self, temp_noise=0., temp_derivative=1.0, prob_at_zero=0.5, binary=True):
        super(SpikeDefault, self).__init__()
        self.u0 = logit(prob_at_zero)
        self.temp_noise = temp_noise
        self.temp_derivative = temp_derivative
        self.hard = binary

    def forward(self, u, hidden):
        # type: (Tensor, Optional[Tensor]) -> Tensor

        u += self.u0

        if self.temp_noise == 0. and self.temp_derivative == 1.:
            p = torch.sigmoid(u)
            if not self.hard: return p
            z = (p > 0.5).float()
            z = p + (z - p).detach()
        else:
            z,_ = gumbel_binary_sample(u, self.temp_derivative, self.temp_noise, self.hard)

        return z

@jit.script
def permute_or_expand_inputs(x, perm, connectivity):
    # type: (Tensor, Optional[Tensor], str) -> Tensor

    if perm is None:
        return x

    if connectivity == "grouped":
        return x

    x_permuted = x[:, perm]

    if connectivity == "random":
        return x_permuted

    if connectivity == "both":
        # the goal here is to interleave grouped and randomized order
        # concatenation is not an option
        shp = x.shape
        shp[1] = 2 * shp[1]

        x_combined = torch.stack([x, x_permuted], 2)
        return x_combined.reshape(shp)

    raise NotImplementedError()


class ConvOp(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, group_size, connectivity: str = "random", orthogonal_initialization=True):
        super(ConvOp, self).__init__()
        assert connectivity in ["random", "both", "grouped"]

        if connectivity in ["random", "both"]:
            self.register_buffer("_perm", torch.randperm(input_size))
        else:
            self._perm = None

        if connectivity == "both":
            input_size = 2 * input_size
            group_size = 2 * group_size

        self.groups = input_size // group_size
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, groups=self.groups, bias=False)
        self.connectivity = connectivity

        if orthogonal_initialization:
            self.reset_to_orthogonal_weights()

    def __repr__(self):
        return f"ConvOp({self.conv.extra_repr()})"

    def reset_to_orthogonal_weights(self):
        init.orthogonal_(self.conv.weight)

    def forward_manual(self, x, kernel_quantized):
        # type: (Tensor, Tensor) -> Tensor
        x = permute_or_expand_inputs(x, self._perm, self.connectivity)
        return F.conv1d(x, weight=kernel_quantized, bias=None, groups=self.groups, )

    def forward(self, x):
        x = permute_or_expand_inputs(x, self._perm, self.connectivity)
        return self.conv(x)


class RSNN(nn.Module):  # jit.ScriptModule):
    def __init__(self, hidden_size,
                 input_size,
                 min_rec_delay,
                 rec_kernel_size,
                 input_kernel_size,
                 group_size,
                 differentiable_hops=2,
                 causal_padding=True,
                 connectivity="both",):

        super(RSNN, self).__init__()

        # define variables
        self.min_rec_delay = min_rec_delay
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rec_kernel_size = rec_kernel_size
        self.input_kernel_size = input_kernel_size
        self.padding = causal_padding
        self.differentiable_hops = differentiable_hops

        #self.spike_module = SpikeDefault(prob_at_zero=prob_at_zero, temp_noise=temp_noise, temp_derivative=temp_derivative)
        self.spike_module = SpikeFunction()

        self.rec_op = ConvOp(hidden_size, hidden_size, rec_kernel_size, group_size, connectivity,
                             orthogonal_initialization=True)

        in_conv = ConvOp(input_size, hidden_size, input_kernel_size, group_size, "random")

        self.in_op = nn.Sequential(
            in_conv,
            nn.BatchNorm1d(self.hidden_size)
        )

        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def input_buffer_size(self):
        return self.input_kernel_size - 1

    def rec_buffer_size(self):
        return self.rec_kernel_size + self.min_rec_delay - 1

    def pad_start(self, a, n_time):
        n_batch, n_channels, _ = a.shape
        zz = torch.zeros([n_batch, n_channels, n_time - a.size(2)], dtype=a.dtype, device=a.device)
        a = torch.cat([zz, a], 2)  # padded with zeros to get the right size
        return a

    def zero_hidden_buffer(self, n_batch, dtype, device):
        # type: (int, torch.dtype, torch.device) -> Tensor
        return torch.zeros([n_batch, self.hidden_size, self.rec_buffer_size()], dtype=dtype, device=device)

    def zero_input_buffer(self, n_batch, dtype, device):
        # type: (int, torch.dtype, torch.device) -> Tensor
        return torch.zeros([n_batch, self.input_size, self.input_buffer_size()], dtype=dtype, device=device)

    # @jit.script_method
    def forward_recursive(self, x, hidden_buffer, input_buffer):
        # type: (Tensor, Tensor, Tensor) -> Tensor

        with torch.no_grad():
            n_batch, _, n_time = x.shape  # x: (n_batch, channel, n_time)
            x = torch.cat([input_buffer[:, :, -self.input_buffer_size():], x], 2)
            a_in = self.in_op(x)  # a: (n_batch, hidden_size, n_time)

            z_list = jit.annotate(List[Tensor], [])
            d = self.min_rec_delay
            t = 0
            bias = self.bias[:, None]

            while t < n_time:
                a_rec = self.rec_op(hidden_buffer)
                t_end = min(t + d, a_in.size(2))
                a = a_in[:, :, t:t_end] + a_rec[:, :, :t_end - t] + bias

                z = self.spike_module.forward(a, hidden_buffer)

                # proceed
                t += d
                hidden_buffer = torch.cat([hidden_buffer[:, :, d:], z], 2)
                z_list.append(z)

            zs = torch.cat(z_list, 2)

        return zs

    # @jit.script_method
    def dummy_forward(self, x, zs_clamp, hidden, input_buffer):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

        d = self.min_rec_delay
        n_time = x.size(2)

        x = torch.cat([input_buffer[:, :, -self.input_buffer_size():], x], 2)
        a_in = self.in_op(x)
        bias = self.bias[:, None]

        # tensors = [a_in, hidden, self.rec_op.conv.weight, bias]
        # _, a_max, q_a_max = self.tensor_quantize(tensors)
        # tensors_fake_quantized = self.fake_quantize(tensors)
        # a_in_quant, hidden_quant, weigh_quant, bias_quant = tensors_fake_quantized

        zs = zs_clamp
        for i in range(self.differentiable_hops):
            # Note1: gradient trick, clamped in forward but permissive in backward
            # Note2: it is logical to put the gradient trick at end of the loop,
            # but I keep it at the beginning of the loop to have the opportunity
            # for testing numerical difference between dummy and recursive forwards
            zs = zs + (zs_clamp - zs).detach()
            zs = torch.cat([hidden, zs[:, :, :-d]], 2)
            a_rec = self.rec_op(zs)
            a = a_in + a_rec + bias

            zs = self.spike_module(a, hidden)

        # zs = self.causal_padding_to(zs, n_time)

        return zs

    # @jit.script_method
    def forward(self, x, hidden=None, input_buffer=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor

        # x: (batch, time, channel)
        x = x.permute([0, 2, 1])  # x: (batch, channel, time)
        n_batch, n_channels, n_time = x.shape

        if hidden is None:
            hidden = self.zero_hidden_buffer(n_batch, x.dtype, x.device)

        if input_buffer is None:
            input_buffer = self.zero_input_buffer(n_batch, x.dtype, x.device)

        with torch.no_grad():
            zs = self.forward_recursive(x, hidden, input_buffer).detach()

        if self.training:
            zs_dummy = self.dummy_forward(x, zs, hidden, input_buffer)
            zs = zs_dummy + (zs - zs_dummy).detach()

        zs = zs.permute([0, 2, 1])  # output: (batch, time, channel)

        return zs

    def causal_padding_to(self, zs, n_time):
        # type: (Tensor, int) -> Tensor
        n_batch, n_channels, n_time_short = zs.shape
        zz = torch.zeros([n_batch, n_channels, n_time - n_time_short], dtype=zs.dtype, device=zs.device)
        zs = torch.cat([zz, zs], 2)
        return zs


class RSNNLayer(nn.Module):  # jit.ScriptModule):

    def __init__(self, hidden_size, input_size,
                 num_layers=1,
                 kernel_size=8,
                 group_size=32,
                 min_delays=4,
                 bidirectional=False,
                 drop_out=0.3,
                 differentiable_hops=2,
                 connectivity="grouped"):

        super(RSNNLayer, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.bidirectional = bidirectional
        self.differentiable_hops = differentiable_hops
        self.group_size = group_size

        self.min_delay = min_delays
        self.connectivity = connectivity

        self.output_dim = 2 * self.hidden_size if bidirectional else self.hidden_size

        self.dropout = nn.Dropout(drop_out) if drop_out > 0.0 and num_layers > 1 else None
        self.forward_layers = nn.ModuleList(self.make_layers())
        self.backward_layers = nn.ModuleList(self.make_layers()) if self.bidirectional else None

    def make_layers(self):

        layer_list = []

        for i_layer in range(self.num_layers):
            # number of inputs
            if i_layer == 0:
                n_in = self.input_size
            else:
                n_in = self.output_dim

            layer = RSNN(hidden_size=self.hidden_size,
                         input_size=n_in,
                         min_rec_delay=self.min_delay,
                         rec_kernel_size=self.kernel_size,
                         input_kernel_size=self.kernel_size,
                         group_size=self.group_size,
                         differentiable_hops=self.differentiable_hops,
                         connectivity=self.connectivity)

            layer_list.append(layer)
        return layer_list

    def forward(self, x):

        for layer_index in range(self.num_layers):

            if self.dropout is not None and layer_index > 0:
                x = self.dropout(x)

            layer_forward = self.forward_layers[layer_index]
            x_forward = layer_forward(x)

            if self.bidirectional:
                layer_backward = self.backward_layers[layer_index]
                x_backward = layer_backward(torch.flip(x, [1]))
                x_backward = torch.flip(x_backward, [1])
                x = torch.cat([x_forward, x_backward], 2)
            else:
                x = x_forward

        return x


class NaiveSpikeLayer(nn.Module):

    def __init__(self, input_dim, embed_dim,spike_fn):
        super(NaiveSpikeLayer, self).__init__()

        self.w_in = nn.Linear(input_dim, embed_dim)
        if spike_fn == "free":
            self.spike_function = SpikeFunction()
        elif spike_fn == "reset":
            self.spike_function = ResetSpikeFunction(n_neurons=embed_dim)
        elif spike_fn == "adapt":
            self.spike_function = AdaptSpikeFunction(n_neurons=embed_dim)
        elif spike_fn == "refractory":
            self.spike_function = RefractorySpikeFunction(n_neurons=embed_dim)
        elif spike_fn == "full":
            self.spike_function = FullSpikeFunction(n_neurons=embed_dim)
    def forward(self, x):
        x = self.w_in(x)
        x = self.spike_function(x)

        return x


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n_in = 1024
    n_rec = 512
    T = 150
    k = 32
    group_size = 32
    d = 4
    n_batch = 2
    connectivity = "grouped"

    model = RSNN(n_rec, n_in, d, k, k, group_size,
                 differentiable_hops=2,
                 connectivity=connectivity)

    x = torch.randn([n_batch, T, n_in], dtype=torch.float32)
    for _ in range(10):
        model(x)

    torchinfo.summary(model, x.shape[1:])

    big_model = RSNNLayer(n_rec, n_in, 3, k, group_size)
    z_many_layers = big_model(x)
    print("z_many_layers: ", z_many_layers.shape)

    x = x.permute([0, 2, 1])

    hidden = model.zero_hidden_buffer(n_batch, x.dtype, x.device)
    input_buffer = model.zero_input_buffer(n_batch, x.dtype, x.device)
    print('forward recursive')
    z = model.forward_recursive(x, hidden, input_buffer)
    print('forward dummy')
    z_dummy = model.dummy_forward(x, z, hidden, input_buffer)

    l1_norm = lambda x: torch.sum(torch.abs(x))
    relative_diff = lambda x1, x2: l1_norm(x1 - x2) / torch.max(l1_norm(x1), l1_norm(x2))
    diff = relative_diff(z, z_dummy).detach().numpy()

    z_np = to_numpy(z)
    z_dummy_np = to_numpy(z_dummy)
    fig, ax_list = plt.subplots(2)
    ax_list[0].pcolor(z_np[0])
    ax_list[1].pcolor(z_dummy_np[0])
    plt.show()

    print("indices:", np.where(z_np != z_dummy_np))

    print("norm z", l1_norm(z))
    print("norm z_dummy", l1_norm(z_dummy))

    print("diff: ", diff)
    print("z_forward_only: ", z.shape)
    print("x: ", x.shape)

